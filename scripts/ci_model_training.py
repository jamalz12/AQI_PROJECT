#!/usr/bin/env python3
"""
CI/CD Model Training Script for Karachi AQI
This script is specifically designed for GitHub Actions workflow
"""

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
from mongodb_feature_store import MongoDBFeatureStore

class CIModelTrainer:
    def __init__(self):
        self.models_dir = 'src/models/saved_models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING', "mongodb://localhost:27017/")
        self.feature_store = MongoDBFeatureStore(connection_string=self.mongodb_connection_string)

    def load_data(self):
        """Load and prepare training data"""
        print("🔍 Loading historical engineered features from MongoDB...")
        df = self.feature_store.get_historical_data(days=30) # Get last 30 days of engineered data

        if df.empty:
            print("❌ No engineered features found in MongoDB for training. Exiting.")
            return None

        print(f"📊 Loaded {len(df)} engineered feature records from MongoDB.")
        return df


    def prepare_features(self, df):
        """Prepare features for training"""
        try:
            # Select features
            feature_columns = [
                'temperature', 'humidity', 'pressure', 'wind_speed',
                'wind_direction', 'visibility', 'clouds',
                'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
                # Engineered features
                'hour', 'day_of_week', 'day_of_year', 'month', 'quarter', 'year',
                'aqi_change_rate', 'aqi_ma_3h', 'temp_humidity_interaction'
            ]

            # Check if all required columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Warning: Missing columns: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in df.columns]

            # Prepare features and target
            X = df[feature_columns].copy()
            y = df['aqi'].copy()

            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Remove rows where target is NaN
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) == 0:
                print("❌ No valid data for training")
                return None, None

            print(f"📈 Prepared {len(X)} samples with {len(feature_columns)} engineered features")
            return X, y

        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None, None

    def train_models(self, X, y):
        """Train multiple models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
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
                print(f"🤖 Training {name}...")

                # Train model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred[:10].tolist(),  # Sample predictions
                    'actual': y_test[:10].tolist()
                }

                print(f"   {name}: MAE = {mae:.3f}, RMSE = {rmse:.3f}, R² = {r2:.3f}")
            # Save models and scaler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            for name, result in results.items():
                model_filename = f"{name}_karachi_{timestamp}.joblib"
                model_path = os.path.join(self.models_dir, model_filename)
                joblib.dump(result['model'], model_path)

                # Save performance metrics
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

            # Save scaler
            scaler_filename = f"scaler_karachi_{timestamp}.joblib"
            scaler_path = os.path.join(self.models_dir, scaler_filename)
            joblib.dump(scaler, scaler_path)

            return results

        except Exception as e:
            print(f"❌ Error training models: {e}")
            return None

    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 Starting CI/CD model training pipeline...")

        # Load data
        df = self.load_data()
        if df is None:
            return False

        # Prepare features
        X, y = self.prepare_features(df)
        if X is None or y is None:
            return False

        # Train models
        results = self.train_models(X, y)
        if results is None:
            return False

        # Print summary
        print("\n📊 Training Summary:")
        best_model = max(results.items(), key=lambda x: x[1]['r2'])

        for name, result in results.items():
            is_best = "🏆 BEST" if name == best_model[0] else ""
            print(f"   {name}: R² = {result['r2']:.3f}, MAE = {result['mae']:.3f} {is_best}")

        print("\n🎉 Model training completed successfully!")
        return True

def main():
    """Main function for CI/CD model training"""
    print("🚀 Starting CI/CD model training for Karachi AQI...")

    trainer = CIModelTrainer()
    success = trainer.run_training_pipeline()

    if success:
        print("✅ Model training completed successfully!")
        sys.exit(0)
    else:
        print("❌ Model training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
