import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from mysql_feature_store import MySQLFeatureStore # Changed import
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CIFeatureEngineer:
    def __init__(self):
        # MySQL setup - retrieve connection details from environment variables
        self.mysql_host = os.getenv('MYSQL_HOST', 'localhost')
        self.mysql_user = os.getenv('MYSQL_USER', 'root')
        self.mysql_password = os.getenv('MYSQL_PASSWORD', '') # No password by default for local setup
        self.mysql_database = os.getenv('MYSQL_DATABASE', 'aqi_data')
        self.mysql_port = int(os.getenv('MYSQL_PORT', 3306))

        logger.info(f"Attempting to connect to MySQL for feature engineering using database: {self.mysql_database} on {self.mysql_host}:{self.mysql_port}...")
        self.feature_store = MySQLFeatureStore(
            host=self.mysql_host,
            user=self.mysql_user,
            password=self.mysql_password,
            database=self.mysql_database,
            port=self.mysql_port
        )
        
        if not self.feature_store.is_connected:
            logger.error("❌ MySQL connection failed during CIFeatureEngineer initialization. Feature engineering will likely fail.")
            sys.exit(1)

    def load_raw_data(self) -> pd.DataFrame:
        """Load recent raw data from MySQL."""
        logger.info("🔍 Loading recent raw data from MySQL for feature engineering...")
        try:
            df = self.feature_store.get_recent_data(hours=1) 
            if df.empty:
                logger.warning("⚠️ No recent raw data found in MySQL for feature engineering. Returning empty DataFrame.")
                return pd.DataFrame()
            logger.info(f"📊 Loaded {len(df)} recent raw records from MySQL.")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading raw data from MySQL: {e}", exc_info=True)
            sys.exit(1)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("⚠️ No data in DataFrame to compute features from. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            df = df.copy() 
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['year'] = df['timestamp'].dt.year

            df['aqi_change_rate'] = df['aqi'].diff().fillna(0) 
            df['aqi_ma_3h'] = df['aqi'].rolling(window=3, min_periods=1).mean().fillna(df['aqi'])
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

            logger.info("✅ Computed features: hour, day_of_week, day_of_year, month, quarter, year, aqi_change_rate, aqi_ma_3h, temp_humidity_interaction.")
            return df
        except Exception as e:
            logger.error(f"❌ Error computing features: {e}", exc_info=True)
            sys.exit(1)

    def store_features(self, df: pd.DataFrame) -> bool:
        if df.empty:
            logger.warning("⚠️ No features to store.")
            return False

        logger.info(f"💾 Storing {len(df)} engineered features into MySQL...")
        
        if 'city' not in df.columns:
            df['city'] = 'Karachi' 
        
        try:
            success = self.feature_store.insert_data(df)
            if success:
                logger.info("✅ Engineered features stored successfully in MySQL.")
            else:
                logger.error("❌ Failed to store engineered features in MySQL.")
            return success
        except Exception as e:
            logger.error(f"❌ Error storing engineered features in MySQL: {e}", exc_info=True)
            sys.exit(1)

def main():
    logger.info("🚀 Starting CI/CD feature engineering pipeline...")
    
    engineer = CIFeatureEngineer()
    
    raw_data_df = engineer.load_raw_data()
    if raw_data_df.empty:
        logger.info("No raw data to process. Exiting feature engineering pipeline gracefully.")
        sys.exit(0)

    features_df = engineer.compute_features(raw_data_df)
    if features_df.empty:
        logger.error("❌ Feature computation resulted in empty DataFrame. Exiting.")
        sys.exit(1)

    success = engineer.store_features(features_df)
    
    if success:
        logger.info("🎉 Feature engineering pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Feature engineering pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
