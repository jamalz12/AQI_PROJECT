import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from mongodb_feature_store import MongoDBFeatureStore
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICIDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.air_quality_url = "http://api.openweathermap.org/data/2.5/air_pollution"

        self.lat = 24.8607
        self.lon = 67.0011

        # MongoDB setup - retrieve connection string from environment variable
        self.mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        
        if not self.mongodb_connection_string:
            logger.error("❌ MONGODB_CONNECTION_STRING environment variable not set. Please configure GitHub Secret.")
            sys.exit(1) # Exit immediately if connection string is missing

        logger.info(f"Attempting to connect to MongoDB using connection string: {self.mongodb_connection_string[:30]}...") # Log first 30 chars for security
        self.feature_store = MongoDBFeatureStore(connection_string=self.mongodb_connection_string)
        
        if self.feature_store.collection is None:
            logger.error("❌ MongoDB connection failed during CICIDataCollector initialization. Data collection will likely fail.")
            sys.exit(1) # Exit early if MongoDB connection fails

    def get_current_weather(self):
        logger.info("Attempting to get current weather data...")
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info("Successfully fetched weather data.")

            weather_data = {
                'timestamp': datetime.now(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 10000),
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'clouds': data['clouds']['all']
            }

            return weather_data

        except requests.exceptions.HTTPError as errh:
            logger.error(f"❌ HTTP Error for weather data: {errh}")
        except requests.exceptions.ConnectionError as errc:
            logger.error(f"❌ Error Connecting for weather data: {errc}")
        except requests.exceptions.Timeout as errt:
            logger.error(f"❌ Timeout Error for weather data: {errt}")
        except requests.exceptions.RequestException as err:
            logger.error(f"❌ Something went wrong with the weather API request: {err}")
        except Exception as e:
            logger.error(f"❌ Unexpected error getting weather data: {e}", exc_info=True)
        return None

    def get_air_quality(self):
        logger.info("Attempting to get air quality data...")
        try:
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key
            }

            response = requests.get(self.air_quality_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info("Successfully fetched air quality data.")

            aqi_categories = {
                1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"
            }

            aqi_data = data['list'][0]
            main_aqi = aqi_data['main']['aqi']

            pollution_data = {
                'aqi': main_aqi,
                'aqi_category': aqi_categories.get(main_aqi, "Unknown"),
                'co': aqi_data['components']['co'],
                'no': aqi_data['components']['no'],
                'no2': aqi_data['components']['no2'],
                'o3': aqi_data['components']['o3'],
                'so2': aqi_data['components']['so2'],
                'pm2_5': aqi_data['components']['pm2_5'],
                'pm10': aqi_data['components']['pm10'],
                'nh3': aqi_data['components']['nh3']
            }

            return pollution_data

        except requests.exceptions.HTTPError as errh:
            logger.error(f"❌ HTTP Error for air quality data: {errh}")
        except requests.exceptions.ConnectionError as errc:
            logger.error(f"❌ Error Connecting for weather data: {errc}")
        except requests.exceptions.Timeout as errt:
            logger.error(f"❌ Timeout Error for weather data: {errt}")
        except requests.exceptions.RequestException as err:
            logger.error(f"❌ Something went wrong with the weather API request: {err}")
        except Exception as e:
            logger.error(f"❌ Unexpected error getting weather data: {e}", exc_info=True)
        return None

    def collect_and_store_data(self):
        logger.info("🌤️ Collecting current weather data for Karachi...")

        weather_data = self.get_current_weather()
        if not weather_data:
            logger.error("❌ Failed to get weather data. Cannot proceed with collection.")
            return False

        aqi_data = self.get_air_quality()
        if not aqi_data:
            logger.error("❌ Failed to get air quality data. Cannot proceed with collection.")
            return False

        complete_data = {**weather_data, **aqi_data}

        df = pd.DataFrame([complete_data])
        logger.info(f"DataFrame created with {len(df)} record(s).")

        logger.info("Attempting to insert data into MongoDB...")
        insertion_success = self.feature_store.insert_data(df)
        if not insertion_success:
            logger.error("❌ Failed to insert data into MongoDB. Exiting.")
            sys.exit(1)

        logger.info(f"✅ Successfully collected and stored data!")
        logger.info(f"   AQI: {complete_data['aqi']} ({complete_data['aqi_category']})")
        logger.info(f"   Temperature: {complete_data['temperature']}°C")
        logger.info(f"   Humidity: {complete_data['humidity']}%")
        
        stats = self.feature_store.get_statistics()
        if stats and 'total_records' in stats:
            logger.info(f"   Total records in MongoDB: {stats['total_records']}")
        else:
            logger.warning("⚠️ Could not retrieve total records from MongoDB.")

        return True

def main():
    logger.info("🚀 Starting CI/CD data collection for Karachi AQI...")
    
    api_key = os.getenv('OPENWEATHER_API_KEY')

    if not api_key:
        logger.error("❌ OPENWEATHER_API_KEY environment variable not set. Please configure GitHub Secret.")
        sys.exit(1)

    logger.info("✅ OPENWEATHER_API_KEY is set.")
    
    collector = CICIDataCollector(api_key)
    success = collector.collect_and_store_data()

    if success:
        logger.info("🎉 Data collection completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Data collection failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
