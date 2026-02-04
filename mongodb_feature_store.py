import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
import sys
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBFeatureStore:
    """
    MongoDB-based feature store for Karachi AQI data.
    Handles connection, data insertion, retrieval, and statistics.
    """

    def __init__(self, connection_string: str, database_name: str = "aqi_data",
                 collection_name: str = "karachi_aqi_features"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

        self._connect()

    def _connect(self):
        """Establish connection to MongoDB Atlas and select database/collection"""
        try:
            logger.info("Attempting to connect to MongoDB Atlas...")
            # Use serverSelectionTimeoutMS to fail fast if MongoDB is unreachable
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            
            # The ping command is cheap and does not require auth.
            # It will fail if the database is unreachable.
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"✅ Connected to MongoDB Atlas. Database: '{self.database_name}', Collection: '{self.collection_name}'")

        except ServerSelectionTimeoutError as err:
            logger.error(f"❌ MongoDB Server Selection Timeout Error: {err}. Please check connection string and network access.")
            self.collection = None # Indicate failure by setting collection to None
            sys.exit(1) # Exit early in CI/CD
        except ConnectionFailure as err:
            logger.error(f"❌ MongoDB Connection Failure: {err}. Please check connection string and network access.")
            self.collection = None
            sys.exit(1) # Exit early in CI/CD
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during MongoDB connection: {e}", exc_info=True)
            self.collection = None
            sys.exit(1) # Exit early in CI/CD

    def insert_data(self, data_df: pd.DataFrame) -> bool:
        """Insert or update data into the feature store"""
        if self.collection is None:
            logger.error("❌ MongoDB connection not established. Cannot insert data.")
            return False

        if data_df.empty:
            logger.warning("⚠️ No data to insert.")
            return True

        # Convert DataFrame to a list of dictionaries for MongoDB insertion
        records = data_df.to_dict(orient='records')

        try:
            # Use update_one with upsert=True to insert if _id (timestamp) doesn't exist, else update
            # MongoDB's default _id is not suitable for upsert by timestamp directly.
            # We will assume 'timestamp' column is unique and use it for upserting.
            # For each record, create an upsert operation based on the timestamp.
            # This is more efficient for single document upserts than bulk_write with many upserts.

            # For potentially large DataFrames, bulk_write is more efficient.
            from pymongo import UpdateOne
            operations = []
            for record in records:
                # MongoDB uses BSON datetime, so ensure the timestamp is a native datetime object
                record['timestamp'] = record['timestamp'].to_pydatetime() if isinstance(record['timestamp'], pd.Timestamp) else record['timestamp']
                
                # Use the timestamp as the unique key for upserting
                operations.append(
                    UpdateOne({'timestamp': record['timestamp']}, {'$set': record}, upsert=True)
                )
            
            if operations:
                result = self.collection.bulk_write(operations)
                logger.info(f"✅ Inserted/Updated {result.upserted_count} new and {result.modified_count} existing records in MongoDB")
            else:
                logger.warning("⚠️ No operations to perform in bulk_write.")

            return True

        except Exception as e:
            logger.error(f"❌ Error inserting/updating data into MongoDB: {e}", exc_info=True)
            return False

    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """Get recent data from feature store (last N hours)"""
        if self.collection is None:
            logger.error("❌ MongoDB connection not established. Cannot retrieve data.")
            return pd.DataFrame()

        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            # Query for data within the last N hours, sorted by timestamp descending
            query = {
                "timestamp": {"$gte": cutoff_time},
                "city": "Karachi" # Assuming city filter
            }
            # Fetch records and convert to DataFrame
            records = list(self.collection.find(query).sort("timestamp", -1))
            
            if records:
                # Convert _id (ObjectId) to string if present, or drop it
                for record in records:
                    record.pop('_id', None)
                df = pd.DataFrame(records)
                logger.info(f"✅ Retrieved {len(df)} recent records from MongoDB")
                return df
            else:
                logger.warning(f"⚠️ No recent data found in MongoDB for the last {hours} hours.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error retrieving recent data from MongoDB: {e}", exc_info=True)
            return pd.DataFrame()

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data for model training (last N days)"""
        if self.collection is None:
            logger.error("❌ MongoDB connection not established. Cannot retrieve historical data.")
            return pd.DataFrame()

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            # Query for data within the last N days, sorted by timestamp ascending
            query = {
                "timestamp": {"$gte": cutoff_date},
                "city": "Karachi" # Assuming city filter
            }
            records = list(self.collection.find(query).sort("timestamp", 1))

            if records:
                for record in records:
                    record.pop('_id', None)
                df = pd.DataFrame(records)
                logger.info(f"✅ Retrieved {len(df)} historical records from MongoDB")
                return df
            else:
                logger.warning(f"⚠️ No historical data found in MongoDB for the last {days} days.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error retrieving historical data from MongoDB: {e}", exc_info=True)
            return pd.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        if self.collection is None:
            logger.error("❌ MongoDB connection not established. Cannot get statistics.")
            return {'error': 'Not connected to database'}

        try:
            # Total records
            total_records = self.collection.count_documents({"city": "Karachi"})

            # Date range (min and max timestamp)
            # Use aggregation pipeline to get min/max dates and avg AQI
            pipeline = [
                {'$match': {"city": "Karachi"}},
                {'$group': {
                    '_id': None,
                    'min_date': {'$min': "$timestamp"},
                    'max_date': {'$max': "$timestamp"},
                    'avg_aqi': {'$avg': "$aqi"},
                    'min_aqi': {'$min': "$aqi"},
                    'max_aqi': {'$max': "$aqi"}
                }}
            ]
            stats_result = list(self.collection.aggregate(pipeline))

            date_range = {'start': None, 'end': None}
            aqi_stats = {'mean': None, 'min': None, 'max': None}

            if stats_result:
                stats = stats_result[0]
                if stats.get('min_date'):
                    date_range['start'] = stats['min_date'].isoformat()
                if stats.get('max_date'):
                    date_range['end'] = stats['max_date'].isoformat()
                if stats.get('avg_aqi') is not None:
                    aqi_stats['mean'] = float(stats['avg_aqi'])
                if stats.get('min_aqi') is not None:
                    aqi_stats['min'] = float(stats['min_aqi'])
                if stats.get('max_aqi') is not None:
                    aqi_stats['max'] = float(stats['max_aqi'])

            return {
                'total_records': total_records,
                'date_range': date_range,
                'aqi_stats': aqi_stats,
                'storage_type': 'MongoDB'
            }

        except Exception as e:
            logger.error(f"❌ Error getting statistics from MongoDB: {e}", exc_info=True)
            return {'error': str(e)}

    def clear_data(self) -> bool:
        """Clear all data for Karachi from the feature store"""
        if self.collection is None:
            logger.error("❌ MongoDB connection not established. Cannot clear data.")
            return False

        try:
            result = self.collection.delete_many({"city": "Karachi"})
            logger.info(f"🗑️ Cleared {result.deleted_count} records for Karachi from MongoDB collection '{self.collection_name}'")
            return True

        except Exception as e:
            logger.error(f"❌ Error clearing data from MongoDB: {e}", exc_info=True)
            return False

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")

if __name__ == "__main__":
    # Example usage for testing locally (requires a running MongoDB instance)
    print("🧪 Testing MongoDB Feature Store")
    print("=" * 50)

    # !!! IMPORTANT: Replace with your actual MongoDB Atlas connection string for local testing !!!
    # Example: "mongodb+srv://<username>:<password>@<cluster-url>/test?retryWrites=true&w=majority"
    test_connection_string = "mongodb://localhost:27017/" # Default local connection for testing
    
    # To test with Atlas, make sure to replace the above with your Atlas connection string
    # And ensure your IP is whitelisted in Atlas
    # Example of Atlas connection string:
    # test_connection_string = "mongodb+srv://<USERNAME>:<PASSWORD>@cluster0.abcde.mongodb.net/aqi_data?retryWrites=true&w=majority"

    store = MongoDBFeatureStore(connection_string=test_connection_string)

    if store.collection is not None:
        print("✅ MongoDBFeatureStore initialized successfully.")

        # Generate sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)]),
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
            'hour': [dt.hour for dt in pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)])],
            'day_of_week': [dt.dayofweek for dt in pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)])],
            'day_of_year': [dt.dayofyear for dt in pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)])],
            'month': [dt.month for dt in pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)])],
            'quarter': [dt.quarter for dt in pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)])],
            'year': [dt.year for dt in pd.to_datetime([datetime.utcnow() - timedelta(hours=i) for i in range(10)])],
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
        print("\n🎉 MongoDB Feature Store test completed!")

    else:
        print("❌ Failed to initialize MongoDBFeatureStore.")
