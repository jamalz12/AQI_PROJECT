import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from postgres_feature_store import PostgresFeatureStore # Changed import
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CIModelTrainer:
    def __init__(self):
        self.models_dir = 'src/models/saved_models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # PostgreSQL setup - retrieve connection details from environment variables
        self.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', '')
        self.postgres_database = os.getenv('POSTGRES_DATABASE', 'aqi_data')
        self.postgres_port = int(os.getenv('POSTGRES_PORT', 5432))

        logger.info(f"Attempting to connect to PostgreSQL for model training using database: {self.postgres_database} on {self.postgres_host}:{self.postgres_port}...")
        self.feature_store = PostgresFeatureStore(
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            database=self.postgres_database,
            port=self.postgres_port
        )
        
        if not self.feature_store.is_connected:
            logger.error("❌ PostgreSQL connection failed during CIModelTrainer initialization. Model training will likely fail.")
            sys.exit(1)

    def load_data(self):
        logger.info("🔍 Loading historical engineered features from PostgreSQL...")
        try:
            df = self.feature_store.get_historical_data(days=30)
            if df.empty:
                logger.error("❌ No engineered features found in PostgreSQL for training. Exiting.")
                return None
            logger.info(f"📊 Loaded {len(df)} engineered feature records from PostgreSQL.")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading historical data from PostgreSQL: {e}", exc_info=True)
            sys.exit(1)

    def prepare_features(self, df):
        try:
            feature_columns = [
                'temperature', 'humidity', 'pressure', 'wind_speed',
                'wind_direction', 'visibility', 'clouds',
                'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
                'hour', 'day_of_week', 'day_of_year', 'month', 'quarter', 'year',
                'aqi_change_rate', 'aqi_ma_3h', 'temp_humidity_interaction'
            ]

            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"⚠️ Warning: Missing columns for feature preparation: {missing_cols}. Attempting to use available features.")
                feature_columns = [col for col in feature_columns if col in df.columns]
                if not feature_columns:
                    logger.error("❌ No valid feature columns found after filtering missing ones. Cannot proceed with training.")
                    return None, None

            X = df[feature_columns].copy()
            y = df['aqi'].copy()

            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) == 0:
                logger.error("❌ No valid data for training after handling missing values.")
                return None, None

            logger.info(f"📈 Prepared {len(X)} samples with {len(feature_columns)} engineered features")
            return X, y

        except Exception as e:
            logger.error(f"❌ Error preparing features for training: {e}", exc_info=True)
            sys.exit(1)

    def train_models(self, X, y):
        logger.info("🤖 Starting model training...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgboost': XGBRegressor(n_estimators=100, random_state=42)
            }

            results = {}

            for name, model in models.items():
                logger.info(f"  Training {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred[:10].tolist(),
                    'actual': y_test[:10].tolist()
                }

                logger.info(f"  {name}: MAE = {mae:.3f}, RMSE = {rmse:.3f}, R² = {r2:.3f}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            for name, result in results.items():
                model_filename = f"{name}_karachi_{timestamp}.joblib"
                model_path = os.path.join(self.models_dir, model_filename)
                joblib.dump(result['model'], model_path)
                logger.info(f"  Saved model: {model_filename}")

                perf_filename = f"performance_{name}_karachi_{timestamp}.json"
                perf_path = os.path.join(self.models_dir, perf_filename)

                perf_data = {
                    'model_name': name,
                    'timestamp': timestamp,
                    'test_mae': result['mae'],
                    'test_rmse': result['rmse'],
                    'test_r2': result['r2'],
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }

                with open(perf_path, 'w') as f:
                    json.dump(perf_data, f, indent=2)
                logger.info(f"  Saved performance metrics: {perf_filename}")

            scaler_filename = f"scaler_karachi_{timestamp}.joblib"
            scaler_path = os.path.join(self.models_dir, scaler_filename)
            joblib.dump(scaler, scaler_path)
            logger.info(f"  Saved scaler: {scaler_filename}")

            return results

        except Exception as e:
            logger.error(f"❌ Error training models: {e}", exc_info=True)
            sys.exit(1)

    def run_training_pipeline(self):
        logger.info("🚀 Starting CI/CD model training pipeline...")

        df = self.load_data()
        if df is None:
            return False

        X, y = self.prepare_features(df)
        if X is None or y is None:
            return False

        results = self.train_models(X, y)
        if results is None:
            return False

        logger.info("\n📊 Training Summary:")
        best_model = max(results.items(), key=lambda x: x[1]['r2'])

        for name, result in results.items():
            is_best = "🏆 BEST" if name == best_model[0] else ""
            logger.info(f"   {name}: R² = {result['r2']:.3f}, MAE = {result['mae']:.3f} {is_best}")

        logger.info("🎉 Model training completed successfully!")
        return True

def main():
    logger.info("🚀 Starting CI/CD model training for Karachi AQI...")

    trainer = CIModelTrainer()
    success = trainer.run_training_pipeline()

    if success:
        logger.info("✅ Model training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Model training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
