"""
BLS LAUS (Local Area Unemployment Statistics) Ingester
Downloads unemployment rates by state from Bureau of Labor Statistics
"""

import os
import logging
import pandas as pd
import requests
from typing import Dict, List
from datetime import datetime, timedelta
from .base import BaseIngester
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from geo import get_all_states

logger = logging.getLogger(__name__)

class BLSIngester(BaseIngester):
    """Ingester for BLS LAUS unemployment data by state"""
    
    def __init__(self):
        super().__init__("BLS")
        self.api_key = os.getenv("BLS_API_KEY")
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        
        if not self.api_key:
            logger.warning("BLS_API_KEY not found, using public API (limited)")
    
    def get_state_series_ids(self) -> Dict[str, str]:
        """Get BLS series IDs for unemployment rate by state"""
        states = get_all_states()
        series_ids = {}
        
        # BLS LAUS series ID format: LASST{state_fips}0000000003
        # where state_fips is 2-digit FIPS code and 03 = unemployment rate
        for state in states:
            fips = state['fips'].zfill(2)  # Ensure 2 digits
            series_id = f"LASST{fips}0000000000003"
            series_ids[series_id] = state['abbr']
        
        return series_ids
    
    def fetch_data(self, start_year: int = 2020) -> pd.DataFrame:
        """Fetch unemployment data from BLS API"""
        series_ids = self.get_state_series_ids()
        all_data = []
        
        # BLS API allows max 50 series per request with registration
        series_list = list(series_ids.keys())
        batch_size = 50 if self.api_key else 25
        
        for i in range(0, len(series_list), batch_size):
            batch = series_list[i:i + batch_size]
            logger.info(f"Fetching BLS batch {i//batch_size + 1}, series: {len(batch)}")
            
            payload = {
                "seriesid": batch,  # BLS expects list format for v1 API
                "startyear": str(start_year),
                "endyear": str(datetime.now().year),
                "registrationkey": self.api_key
            }
            
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") != "REQUEST_SUCCEEDED":
                    logger.error(f"BLS API error: {data.get('message', 'Unknown error')}")
                    continue
                
                # Parse response
                for series in data.get("Results", {}).get("series", []):
                    series_id = series["seriesID"]
                    state_abbr = series_ids.get(series_id, "UNK")
                    
                    for obs in series.get("data", []):
                        # BLS date format: YYYY + period (M01-M12)
                        year = obs["year"]
                        period = obs["period"]
                        
                        if period.startswith("M"):
                            month = period[1:].zfill(2)
                            date_str = f"{year}-{month}-01"
                            
                            try:
                                date = pd.to_datetime(date_str)
                                value = float(obs["value"]) if obs["value"] != "." else None
                                
                                if value is not None:
                                    all_data.append({
                                        "date": date,
                                        "series_id": series_id,
                                        "state_abbr": state_abbr,
                                        "value": value
                                    })
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error parsing BLS data point: {e}")
                                continue
                
                # Rate limiting
                time.sleep(1)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching BLS data batch {i//batch_size + 1}: {e}")
                continue
        
        if not all_data:
            logger.error("No BLS data retrieved")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logger.info(f"Retrieved {len(df)} BLS unemployment records")
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform BLS data to standard format"""
        if df.empty:
            return df
        
        # Create a copy to avoid warnings
        df = df.copy()
        
        # Map state abbreviations to geo keys
        state_fips_map = {state['abbr']: state['fips'] for state in get_all_states()}
        df['fips'] = df['state_abbr'].map(state_fips_map)
        df['geo_key'] = df['fips'].apply(lambda x: f"STATE_{x}" if pd.notna(x) else None)
        
        # Remove rows where geo mapping failed
        df = df.dropna(subset=['geo_key'])
        
        # Standardize columns
        df['geo_level'] = 'STATE'
        df['metric'] = 'UNEMPLOYMENT_RATE'
        df['source'] = 'BLS_LAUS'
        df['unit'] = 'percent'
        df['freq'] = 'monthly'
        
        # Select final columns
        result = df[['date', 'geo_key', 'geo_level', 'metric', 'value', 'source', 'unit', 'freq']].copy()
        
        # Sort by date and geo_key
        result = result.sort_values(['geo_key', 'date'])
        
        logger.info(f"Transformed {len(result)} BLS records for {df['geo_key'].nunique()} states")
        return result
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate BLS data quality"""
        if df.empty:
            logger.warning("BLS dataset is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'geo_key', 'metric', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"BLS data missing columns: {missing_cols}")
            return False
        
        # Check value ranges (unemployment rate should be 0-30%)
        invalid_values = df[(df['value'] < 0) | (df['value'] > 30)]
        if not invalid_values.empty:
            logger.warning(f"BLS data has {len(invalid_values)} records with invalid unemployment rates")
        
        # Check for recent data
        latest_date = df['date'].max()
        if pd.to_datetime(latest_date) < pd.to_datetime(datetime.now()) - timedelta(days=90):
            logger.warning(f"BLS data may be stale, latest date: {latest_date}")
        
        # Check state coverage
        unique_states = df['geo_key'].nunique()
        if unique_states < 45:  # Allow some missing states/territories
            logger.warning(f"BLS data covers only {unique_states} states/territories")
        
        logger.info(f"BLS validation passed: {len(df)} records, {unique_states} states")
        return True

    def run(self) -> bool:
        """Run the full BLS ingestion process"""
        try:
            logger.info("Starting BLS LAUS unemployment data ingestion")
            
            # Fetch data
            raw_data = self.fetch_data()
            if raw_data.empty:
                logger.error("No BLS data fetched")
                return False
            
            # Transform data
            clean_data = self.transform_data(raw_data)
            if clean_data.empty:
                logger.error("No BLS data after transformation")
                return False
            
            # Validate data
            if not self.validate_data(clean_data):
                logger.error("BLS data validation failed")
                return False
            
            # Save to database
            success = self.save_to_database(
                clean_data, 
                'fact_metric',
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                logger.info(f"Successfully ingested {len(clean_data)} BLS unemployment records")
                return True
            else:
                logger.error("Failed to save BLS data to database")
                return False
                
        except Exception as e:
            logger.error(f"BLS ingestion failed: {e}")
            return False
