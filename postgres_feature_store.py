import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import Error
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgresFeatureStore:
    """
    PostgreSQL-based feature store for Karachi AQI data
    """

    def __init__(self, host: str, user: str, password: str, database: str, port: int = 5432,
                 table_name: str = "karachi_aqi_features"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.table_name = table_name
        self.connection = None
        self.is_connected = False

        self._connect()
        self._create_table()

    def _connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            # Set autocommit to true if you want immediate writes without explicit commits
            self.connection.autocommit = True
            self.is_connected = True
            logger.info(f"✅ Connected to PostgreSQL database: {self.database}")

        except Error as e:
            logger.error(f"❌ Error connecting to PostgreSQL: {e}")
            self.is_connected = False

    def _create_table(self):
        """Create feature table if it doesn't exist"""
        if not self.is_connected:
            logger.error("❌ Not connected to PostgreSQL, cannot create table.")
            return

        try:
            cursor = self.connection.cursor()
            create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                timestamp TIMESTAMP PRIMARY KEY,
                city VARCHAR(255),
                temperature REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                visibility REAL,
                clouds REAL,
                weather_main VARCHAR(255),
                weather_description VARCHAR(255),
                aqi REAL,
                aqi_category VARCHAR(255),
                co REAL,
                no REAL,
                no2 REAL,
                o3 REAL,
                so2 REAL,
                pm2_5 REAL,
                pm10 REAL,
                nh3 REAL,
                # Engineered features
                hour INTEGER,
                day_of_week INTEGER,
                day_of_year INTEGER,
                month INTEGER,
                quarter INTEGER,
                year INTEGER,
                aqi_change_rate REAL,
                aqi_ma_3h REAL,
                temp_humidity_interaction REAL
            );"""
            cursor.execute(create_table_query)
            logger.info(f"✅ Table '{self.table_name}' checked/created successfully")

        except Error as e:
            logger.error(f"❌ Error creating table '{self.table_name}': {e}")
            self.is_connected = False # Mark as disconnected if table creation fails

    def insert_data(self, data_df: pd.DataFrame) -> bool:
        """Insert or update data into the feature store"""
        if not self.is_connected:
            logger.error("❌ Not connected to PostgreSQL, cannot insert data.")
            return False

        if data_df.empty:
            logger.warning("⚠️ No data to insert.")
            return True

        try:
            cursor = self.connection.cursor()
            columns = ', '.join(data_df.columns)
            placeholders = ', '.join(['%s'] * len(data_df.columns))

            # PostgreSQL UPSERT syntax
            update_fields = ', '.join([f'{col}=EXCLUDED.{col}' for col in data_df.columns if col != 'timestamp'])
            
            insert_query = f"""INSERT INTO {self.table_name} ({columns})
                            VALUES ({placeholders})
                            ON CONFLICT (timestamp) DO UPDATE SET {update_fields}"""

            records_to_insert = [tuple(row) for row in data_df.values]
            
            cursor.executemany(insert_query, records_to_insert)
            logger.info(f"✅ Inserted/Updated {cursor.rowcount} records into PostgreSQL")
            return True

        except Error as e:
            logger.error(f"❌ Error inserting/updating data into PostgreSQL: {e}")
            return False

    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """Get recent data from feature store"""
        if not self.is_connected:
            logger.error("❌ Not connected to PostgreSQL, cannot retrieve data.")
            return pd.DataFrame()

        try:
            # Use cursor_factory for dictionary-like rows
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                query = f"""SELECT * FROM {self.table_name}
                          WHERE timestamp >= %s AND city = 'Karachi'
                          ORDER BY timestamp DESC"""
                cursor.execute(query, (cutoff_time,))
                records = cursor.fetchall()
                
                if records:
                    df = pd.DataFrame(records)
                    # Convert RealDictRow objects to regular dicts if needed, or pandas handles it
                    df = pd.DataFrame([dict(row) for row in records])
                    logger.info(f"✅ Retrieved {len(df)} recent records from PostgreSQL")
                    return df
                else:
                    logger.warning("⚠️ No recent data found in PostgreSQL")
                    return pd.DataFrame()

        except Error as e:
            logger.error(f"❌ Error retrieving recent data from PostgreSQL: {e}")
            return pd.DataFrame()

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data for model training"""
        if not self.is_connected:
            logger.error("❌ Not connected to PostgreSQL, cannot retrieve historical data.")
            return pd.DataFrame()

        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cutoff_date = datetime.now() - timedelta(days=days)
                query = f"""SELECT * FROM {self.table_name}
                          WHERE timestamp >= %s AND city = 'Karachi'
                          ORDER BY timestamp ASC"""
                cursor.execute(query, (cutoff_date,))
                records = cursor.fetchall()

                if records:
                    df = pd.DataFrame([dict(row) for row in records])
                    logger.info(f"✅ Retrieved {len(df)} historical records from PostgreSQL")
                    return df
                else:
                    logger.warning("⚠️ No historical data found in PostgreSQL")
                    return pd.DataFrame()

        except Error as e:
            logger.error(f"❌ Error retrieving historical data from PostgreSQL: {e}")
            return pd.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        if not self.is_connected:
            logger.error("❌ Not connected to PostgreSQL, cannot get statistics.")
            return {'error': 'Not connected to database'}

        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                
                # Total records
                cursor.execute(f"SELECT COUNT(*) as total_records FROM {self.table_name} WHERE city = 'Karachi'")
                total_records = cursor.fetchone()['total_records']

                # Date range
                cursor.execute(f"SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date FROM {self.table_name} WHERE city = 'Karachi'")
                date_range_result = cursor.fetchone()
                date_range = {
                    'start': date_range_result['start_date'].isoformat() if date_range_result['start_date'] else None,
                    'end': date_range_result['end_date'].isoformat() if date_range_result['end_date'] else None
                }

                # AQI statistics
                cursor.execute(f"SELECT AVG(aqi) as mean_aqi, MIN(aqi) as min_aqi, MAX(aqi) as max_aqi FROM {self.table_name} WHERE city = 'Karachi'")
                aqi_stats_result = cursor.fetchone()
                aqi_stats = {
                    'mean': float(aqi_stats_result['mean_aqi']) if aqi_stats_result['mean_aqi'] else None,
                    'min': float(aqi_stats_result['min_aqi']) if aqi_stats_result['min_aqi'] else None,
                    'max': float(aqi_stats_result['max_aqi']) if aqi_stats_result['max_aqi'] else None
                }

                return {
                    'total_records': total_records,
                    'date_range': date_range,
                    'aqi_stats': aqi_stats,
                    'storage_type': 'PostgreSQL'
                }

        except Error as e:
            logger.error(f"❌ Error getting statistics from PostgreSQL: {e}")
            return {'error': str(e)}

    def clear_data(self) -> bool:
        """Clear all data for Karachi from the feature store"""
        if not self.is_connected:
            logger.error("❌ Not connected to PostgreSQL, cannot clear data.")
            return False

        try:
            cursor = self.connection.cursor()
            cursor.execute(f"DELETE FROM {self.table_name} WHERE city = 'Karachi'")
            logger.info(f"🗑️ Cleared {cursor.rowcount} records for Karachi from PostgreSQL table '{self.table_name}'")
            return True

        except Error as e:
            logger.error(f"❌ Error clearing data from PostgreSQL: {e}")
            return False

    def close_connection(self):
        """Close PostgreSQL connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("🔌 PostgreSQL connection closed")

if __name__ == "__main__":
    # Example usage for testing
    print("🧪 Testing PostgreSQL Feature Store")
    print("=" * 50)

    # Assume a Heroku Postgres instance for testing locally
    # Replace with your actual Heroku Postgres credentials for local testing
    postgres_config = {
        'host': 'YOUR_HEROKU_POSTGRES_HOST',
        'user': 'YOUR_HEROKU_POSTGRES_USER',
        'password': 'YOUR_HEROKU_POSTGRES_PASSWORD',
        'database': 'YOUR_HEROKU_POSTGRES_DATABASE',
        'port': 5432
    }

    try:
        store = PostgresFeatureStore(**postgres_config)

        if store.is_connected:
            print("✅ PostgresFeatureStore initialized successfully.")

            # Generate sample data
            sample_data = pd.DataFrame({
                'timestamp': pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)]),
                'city': ['Karachi'] * 10,
                'temperature': [25.5, 26.0, 25.1, 24.8, 26.2, 27.0, 26.5, 25.9, 24.5, 23.9],
                'humidity': [60.2, 61.5, 60.8, 59.5, 62.1, 63.0, 61.9, 60.5, 58.9, 57.5],
                'pressure': [1000.1, 1000.5, 999.8, 1000.3, 1001.0, 1001.2, 1000.7, 1000.0, 999.5, 999.0],
                'wind_speed': [5.1, 5.5, 5.0, 4.8, 5.3, 5.8, 5.2, 4.9, 4.5, 4.0],
                'wind_direction': [180, 185, 175, 190, 170, 195, 165, 182, 178, 192],
                'visibility': [10000.0, 9500.0, 9800.0, 10000.0, 9000.0, 9200.0, 9700.0, 9900.0, 10000.0, 9500.0],
                'clouds': [20.0, 25.0, 22.0, 18.0, 30.0, 28.0, 24.0, 21.0, 19.0, 17.0],
                'weather_main': ['Clear'] * 10,
                'weather_description': ['clear sky'] * 10,
                'aqi': [85.0, 88.0, 82.0, 80.0, 90.0, 92.0, 87.0, 84.0, 78.0, 75.0],
                'aqi_category': ['Moderate'] * 10,
                'co': [500.0, 510.0, 490.0, 480.0, 520.0, 530.0, 505.0, 495.0, 470.0, 460.0],
                'no': [10.0, 10.5, 9.8, 9.5, 11.0, 11.5, 10.2, 9.9, 9.0, 8.5],
                'no2': [15.0, 15.5, 14.8, 14.5, 16.0, 16.5, 15.2, 14.9, 14.0, 13.5],
                'o3': [30.0, 31.0, 29.5, 28.0, 32.0, 33.0, 30.5, 29.0, 27.5, 26.0],
                'so2': [8.0, 8.2, 7.9, 7.5, 8.5, 8.8, 8.1, 7.7, 7.3, 7.0],
                'pm2_5': [30.0, 31.0, 29.0, 28.0, 32.0, 33.0, 30.5, 29.5, 27.0, 26.0],
                'pm10': [50.0, 52.0, 48.0, 46.0, 54.0, 56.0, 51.0, 49.0, 45.0, 44.0],
                'nh3': [2.0, 2.1, 1.9, 1.8, 2.2, 2.3, 2.0, 1.9, 1.7, 1.6],
                'hour': [dt.hour for dt in pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)])],
                'day_of_week': [dt.dayofweek for dt in pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)])],
                'day_of_year': [dt.dayofyear for dt in pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)])],
                'month': [dt.month for dt in pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)])],
                'quarter': [dt.quarter for dt in pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)])],
                'year': [dt.year for dt in pd.to_datetime([datetime.now() - timedelta(hours=i) for i in range(10)])],
                'aqi_change_rate': [0.0, 3.0, -6.0, -2.0, 10.0, 2.0, -5.0, -3.0, -6.0, -3.0],
                'aqi_ma_3h': [85.0, 86.5, 85.0, 83.33, 84.0, 86.0, 89.67, 86.67, 83.0, 79.33],
                'temp_humidity_interaction': [1533.0, 1599.0, 1526.0, 1476.0, 1627.82, 1701.0, 1622.35, 1568.95, 1438.05, 1374.9]
            })

            # Test insertion
            print("\n📥 Testing data insertion...")
            success = store.insert_data(sample_data)
            if success:
                print("✅ Data insertion successful")
            else:
                print("❌ Data insertion failed")

            # Test retrieval
            print("\n📤 Testing data retrieval...")
            recent_data = store.get_recent_data(hours=12)
            print(f"✅ Retrieved {len(recent_data)} recent records")

            historical_data = store.get_historical_data(days=1)
            print(f"✅ Retrieved {len(historical_data)} historical records")

            # Test statistics
            print("\n📊 Testing statistics...")
            stats = store.get_statistics()
            print(f"✅ Statistics: {stats}")

            # Close connection
            store.close_connection()
            print("\n🎉 PostgreSQL Feature Store test completed!")

        else:
            print("❌ Failed to initialize PostgresFeatureStore.")
    except Exception as e:
        print(f"❌ An error occurred during testing: {e}")
