import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from mongodb_feature_store import MongoDBFeatureStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIFeatureEngineer:
    def __init__(self):
        self.mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING', "mongodb://localhost:27017/")
        self.feature_store = MongoDBFeatureStore(connection_string=self.mongodb_connection_string)

    def load_raw_data(self) -> pd.DataFrame:
        """Load recent raw data from MongoDB."""
        logger.info("🔍 Loading recent raw data from MongoDB for feature engineering...")
        # Load data collected in the last hour, as per the pipeline schedule
        df = self.feature_store.get_recent_data(hours=1) 
        if df.empty:
            logger.warning("⚠️ No recent raw data found in MongoDB for feature engineering.")
            return pd.DataFrame()
        logger.info(f"📊 Loaded {len(df)} recent raw records from MongoDB.")
        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-based and derived features.
        
        Args:
            df: DataFrame with raw AQI and weather data.
            
        Returns:
            DataFrame with computed features.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy() # Avoid SettingWithCopyWarning
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year

        # Derived features
        # AQI change rate (difference from previous hour)
        df['aqi_change_rate'] = df['aqi'].diff().fillna(0) 
        
        # Simple moving average of AQI (e.g., last 3 hours)
        df['aqi_ma_3h'] = df['aqi'].rolling(window=3, min_periods=1).mean().fillna(df['aqi'])
        
        # Interaction features (example)
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

        logger.info("✅ Computed features: hour, day_of_week, day_of_year, month, quarter, year, aqi_change_rate, aqi_ma_3h, temp_humidity_interaction.")
        return df

    def store_features(self, df: pd.DataFrame) -> bool:
        """Store engineered features back into MongoDB."""
        if df.empty:
            logger.warning("⚠️ No features to store.")
            return False

        logger.info(f"💾 Storing {len(df)} engineered features into MongoDB...")
        # We will insert these as new documents or update existing ones if _id is present
        # Assuming we want to update the existing raw data documents with new feature fields
        
        # Convert DataFrame to list of dictionaries for MongoDB
        records = df.to_dict('records')
        
        # MongoDBFeatureStore.insert_data already handles insertion and metadata update.
        # It also handles duplicate timestamps. So, we can reuse it.
        # Ensure 'city' column is present as expected by insert_data
        if 'city' not in df.columns:
            df['city'] = 'Karachi' # Assuming Karachi if not present
        
        success = self.feature_store.insert_data(df)
        if success:
            logger.info("✅ Engineered features stored successfully in MongoDB.")
        else:
            logger.error("❌ Failed to store engineered features in MongoDB.")
        return success

def main():
    logger.info("🚀 Starting CI/CD feature engineering pipeline...")
    
    engineer = CIFeatureEngineer()
    
    # 1. Load raw data
    raw_data_df = engineer.load_raw_data()
    if raw_data_df.empty:
        sys.exit(0) # Exit gracefully if no raw data to process

    # 2. Compute features
    features_df = engineer.compute_features(raw_data_df)
    if features_df.empty:
        logger.error("❌ Feature computation resulted in empty DataFrame.")
        sys.exit(1)

    # 3. Store features
    success = engineer.store_features(features_df)
    
    if success:
        logger.info("🎉 Feature engineering pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Feature engineering pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
