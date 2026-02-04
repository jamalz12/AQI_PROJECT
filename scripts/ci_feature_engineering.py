import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from postgres_feature_store import PostgresFeatureStore # Changed import
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CIFeatureEngineer:
    def __init__(self):
        # PostgreSQL setup - retrieve connection details from environment variables
        self.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', '')
        self.postgres_database = os.getenv('POSTGRES_DATABASE', 'aqi_data')
        self.postgres_port = int(os.getenv('POSTGRES_PORT', 5432))

        logger.info(f"Attempting to connect to PostgreSQL for feature engineering using database: {self.postgres_database} on {self.postgres_host}:{self.postgres_port}...")
        self.feature_store = PostgresFeatureStore(
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            database=self.postgres_database,
            port=self.postgres_port
        )
        
        if not self.feature_store.is_connected:
            logger.error("❌ PostgreSQL connection failed during CIFeatureEngineer initialization. Feature engineering will likely fail.")
            sys.exit(1)

    def load_raw_data(self) -> pd.DataFrame:
        """Load recent raw data from PostgreSQL."""
        logger.info("🔍 Loading recent raw data from PostgreSQL for feature engineering...")
        try:
            df = self.feature_store.get_recent_data(hours=1) 
            if df.empty:
                logger.warning("⚠️ No recent raw data found in PostgreSQL for feature engineering. Returning empty DataFrame.")
                return pd.DataFrame()
            logger.info(f"📊 Loaded {len(df)} recent raw records from PostgreSQL.")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading raw data from PostgreSQL: {e}", exc_info=True)
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

        logger.info(f"💾 Storing {len(df)} engineered features into PostgreSQL...")
        
        if 'city' not in df.columns:
            df['city'] = 'Karachi' 
        
        try:
            success = self.feature_store.insert_data(df)
            if success:
                logger.info("✅ Engineered features stored successfully in PostgreSQL.")
            else:
                logger.error("❌ Failed to store engineered features in PostgreSQL.")
            return success
        except Exception as e:
            logger.error(f"❌ Error storing engineered features in PostgreSQL: {e}", exc_info=True)
            sys.exit(1)

def main():
    logger.info("🚀 Starting CI/CD feature engineering pipeline...")
    
    engineer = CIFeatureEngineer()
    
    raw_data_df = engineer.load_raw_data()
    if raw_data_df.empty:
        logger.info("No raw data to process. Exiting feature engineering pipeline gracefully.")
    # No longer exiting on empty raw data, just logging and proceeding for subsequent runs.
    # sys.exit(0) # Removed exit on empty raw data for graceful handling.

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
