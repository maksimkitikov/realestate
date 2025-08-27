"""
Redfin Data Center Ingester
Downloads real estate market data by state from Redfin public data
"""

import os
import logging
import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import io
from .base import BaseIngester
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from geo import get_all_states

logger = logging.getLogger(__name__)

class RedfinIngester(BaseIngester):
    """Ingester for Redfin market data by state"""
    
    def __init__(self):
        super().__init__("REDFIN")
        # Redfin Data Center provides public CSV downloads
        self.base_url = "https://redfin-public-data.s3.us-west-2.amazonaws.com"
        
    def get_state_mapping(self) -> Dict[str, str]:
        """Get mapping of state abbreviations to FIPS codes"""
        states = get_all_states()
        abbr_to_fips = {state['abbr']: state['fips'] for state in states}
        return abbr_to_fips
    
    def fetch_redfin_data(self) -> pd.DataFrame:
        """Fetch Redfin state-level data"""
        logger.info("Fetching Redfin market data")
        
        # Redfin publishes state-level data in this CSV format
        csv_urls = [
            f"{self.base_url}/redfin_market_tracker/state_market_tracker.tsv000.gz",
            f"{self.base_url}/redfin_market_tracker/state_market_tracker.tsv"
        ]
        
        for url in csv_urls:
            try:
                logger.info(f"Trying Redfin URL: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                # Handle gzipped or plain text
                if url.endswith('.gz'):
                    import gzip
                    content = gzip.decompress(response.content).decode('utf-8')
                else:
                    content = response.text
                
                # Read TSV data
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data, sep='\t')
                
                if not df.empty:
                    logger.info(f"Retrieved Redfin data with {len(df)} rows from {url}")
                    return df
                    
            except (requests.exceptions.RequestException, Exception) as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
                continue
        
        # If all URLs fail, create sample data
        logger.warning("All Redfin URLs failed, using sample data")
        return self._create_sample_redfin_data()
    
    def _create_sample_redfin_data(self) -> pd.DataFrame:
        """Create sample Redfin data structure"""
        logger.warning("Using sample Redfin data structure")
        
        states = get_all_states()
        sample_data = []
        
        import random
        random.seed(42)
        
        # Generate sample data for last 3 years, monthly
        start_date = datetime(2021, 1, 1)
        current_date = datetime.now()
        
        for state in states[:15]:  # Limit for sample
            base_price = 300000 + random.uniform(-100000, 200000)
            base_inventory = 1000 + random.uniform(-500, 1000)
            
            date = start_date
            while date <= current_date:
                # Monthly data
                if date.day == 1:
                    # Simulate market trends
                    price_change = random.uniform(-0.02, 0.03)
                    inventory_change = random.uniform(-0.1, 0.1)
                    
                    base_price *= (1 + price_change)
                    base_inventory *= (1 + inventory_change)
                    
                    sample_data.append({
                        'period_begin': date.strftime('%Y-%m-%d'),
                        'period_end': date.strftime('%Y-%m-%d'),
                        'period_duration': 1,
                        'region_type': 'state',
                        'region_type_id': 2,
                        'table_id': 1,
                        'is_seasonally_adjusted': 'False',
                        'region': state['name'],
                        'state': state['abbr'],
                        'state_code': state['abbr'],
                        'property_type': 'All Residential',
                        'property_type_id': 5,
                        'median_sale_price': round(base_price),
                        'median_list_price': round(base_price * 1.05),
                        'median_ppsf': round(base_price / 2000, 2),
                        'homes_sold': int(base_inventory * 0.1),
                        'inventory': int(base_inventory),
                        'months_of_supply': round(random.uniform(1.5, 6.0), 1),
                        'median_dom': int(random.uniform(20, 80)),
                        'avg_sale_to_list': round(random.uniform(0.95, 1.02), 3),
                        'sold_above_list': round(random.uniform(0.2, 0.8), 3),
                        'price_drops': round(random.uniform(0.1, 0.4), 3),
                        'off_market_in_two_weeks': round(random.uniform(0.3, 0.7), 3)
                    })
                
                # Move to next month
                if date.month == 12:
                    date = date.replace(year=date.year + 1, month=1)
                else:
                    date = date.replace(month=date.month + 1)
        
        return pd.DataFrame(sample_data)
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch Redfin housing market data via RapidAPI"""
        logger.info("Fetching Redfin housing data via RapidAPI")
        
        # RapidAPI Redfin endpoint
        url = "https://redfin-public-data.p.rapidapi.com/stats"
        
        headers = {
            "X-RapidAPI-Key": "db9467b6b5msh01439dd71b85f61p19c624jsn06c77981462c",
            "X-RapidAPI-Host": "redfin-public-data.p.rapidapi.com"
        }
        
        states = get_all_states()
        all_data = []
        
        for state in states:
            try:
                # Query Redfin API for each state
                params = {
                    "region_id": state['abbr'],
                    "region_type": "state",
                    "property_type": "All Residential"
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process Redfin data
                    for record in data.get('records', []):
                        try:
                            date = pd.to_datetime(record.get('period_begin', ''))
                            if pd.isna(date):
                                continue
                            
                            # Extract metrics
                            median_sale_price = record.get('median_sale_price')
                            inventory = record.get('inventory')
                            median_dom = record.get('median_dom')
                            
                            if median_sale_price:
                                all_data.append({
                                    "date": date,
                                    "state_fips": state['fips'],
                                    "metric": "MEDIAN_SALE_PRICE",
                                    "value": float(median_sale_price),
                                    "unit": "dollars",
                                    "freq": "monthly"
                                })
                            
                            if inventory:
                                all_data.append({
                                    "date": date,
                                    "state_fips": state['fips'],
                                    "metric": "INVENTORY",
                                    "value": float(inventory),
                                    "unit": "count",
                                    "freq": "monthly"
                                })
                            
                            if median_dom:
                                all_data.append({
                                    "date": date,
                                    "state_fips": state['fips'],
                                    "metric": "DAYS_ON_MARKET",
                                    "value": float(median_dom),
                                    "unit": "days",
                                    "freq": "monthly"
                                })
                        
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing Redfin record for {state['abbr']}: {e}")
                            continue
                
                # Rate limiting
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching Redfin data for {state['abbr']}: {e}")
                continue
        
        if not all_data:
            logger.error("No Redfin data retrieved")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logger.info(f"Retrieved {len(df)} Redfin records")
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic Redfin data as fallback"""
        logger.info("Generating synthetic Redfin housing data")
        
        states = get_all_states()
        all_data = []
        
        # Generate data for last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        for state in states:
            # Base values vary by state
            base_price = 200000 + (hash(state['abbr']) % 400000)  # $200K-$600K
            base_inventory = 1000 + (hash(state['abbr']) % 5000)   # 1K-6K homes
            
            current_date = start_date
            while current_date <= end_date:
                # Add some monthly variation
                monthly_variation = (hash(f"{state['abbr']}{current_date.month}") % 20 - 10) / 100.0
                
                # Median sale price
                price = base_price * (1 + monthly_variation)
                all_data.append({
                    "date": current_date,
                    "state_fips": state['fips'],
                    "metric": "MEDIAN_SALE_PRICE",
                    "value": price,
                    "unit": "dollars",
                    "freq": "monthly"
                })
                
                # Inventory
                inventory = base_inventory * (1 + monthly_variation * 0.5)
                all_data.append({
                    "date": current_date,
                    "state_fips": state['fips'],
                    "metric": "INVENTORY",
                    "value": inventory,
                    "unit": "count",
                    "freq": "monthly"
                })
                
                # Days on market
                dom = 30 + (hash(f"{state['abbr']}{current_date.month}") % 60)
                all_data.append({
                    "date": current_date,
                    "state_fips": state['fips'],
                    "metric": "DAYS_ON_MARKET",
                    "value": dom,
                    "unit": "days",
                    "freq": "monthly"
                })
                
                current_date += timedelta(days=30)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} synthetic Redfin records")
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Redfin data to standard format"""
        if df.empty:
            return df
        
        logger.info(f"Transforming Redfin data with columns: {list(df.columns)}")
        
        # Create a copy
        df = df.copy()
        
        records = []
        
        for _, row in df.iterrows():
            try:
                date = row['date']
                state_fips = row['state_fips']
                geo_key = f"STATE_{state_fips}"
                
                records.append({
                    'date': date,
                    'geo_key': geo_key,
                    'geo_level': 'STATE',
                    'metric': row['metric'],
                    'value': float(row['value']),
                    'source': 'REDFIN',
                    'unit': row['unit'],
                    'freq': row['freq']
                })
            
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing Redfin row: {e}")
                continue
        
        if not records:
            logger.warning("No valid Redfin records after transformation")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(records)
        
        # Sort by metric, geo_key, and date
        result_df = result_df.sort_values(['metric', 'geo_key', 'date'])
        
        logger.info(f"Transformed {len(result_df)} Redfin records for {result_df['geo_key'].nunique()} states")
        return result_df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from Redfin data"""
        derived_records = []
        
        # Calculate price growth YoY for median sale price
        price_df = df[df['metric'] == 'MEDIAN_SALE_PRICE'].copy()
        if not price_df.empty:
            price_df = price_df.sort_values(['geo_key', 'date'])
            
            # Calculate YoY growth (12 months ago for monthly data)
            price_df['prev_year_value'] = price_df.groupby('geo_key')['value'].shift(12)
            price_df['price_growth_yoy'] = (
                (price_df['value'] - price_df['prev_year_value']) / 
                price_df['prev_year_value'] * 100
            )
            
            # Add growth records
            valid_growth = price_df[price_df['price_growth_yoy'].notna()]
            for _, row in valid_growth.iterrows():
                derived_records.append({
                    'date': row['date'],
                    'geo_key': row['geo_key'],
                    'geo_level': 'STATE',
                    'metric': 'PRICE_GROWTH_YOY',
                    'value': row['price_growth_yoy'],
                    'source': 'REDFIN',
                    'unit': 'percent',
                    'freq': 'monthly'
                })
        
        return pd.DataFrame(derived_records) if derived_records else pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate Redfin data quality"""
        if df.empty:
            logger.warning("Redfin dataset is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'geo_key', 'metric', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Redfin data missing columns: {missing_cols}")
            return False
        
        # Check price values (should be reasonable for US real estate)
        price_data = df[df['metric'] == 'MEDIAN_SALE_PRICE']
        if not price_data.empty:
            invalid_prices = price_data[(price_data['value'] < 50000) | (price_data['value'] > 5000000)]
            if not invalid_prices.empty:
                logger.warning(f"Redfin data has {len(invalid_prices)} records with unusual price values")
        
        # Check for recent data
        latest_date = df['date'].max()
        if latest_date < datetime.now() - timedelta(days=90):
            logger.warning(f"Redfin data may be stale, latest date: {latest_date}")
        
        # Check state coverage
        unique_states = df['geo_key'].nunique()
        if unique_states < 30:  # Redfin doesn't cover all states equally
            logger.warning(f"Redfin data covers only {unique_states} states")
        
        # Check metrics
        available_metrics = df['metric'].unique()
        expected_metrics = ['MEDIAN_SALE_PRICE', 'INVENTORY', 'MEDIAN_DAYS_ON_MARKET']
        missing_metrics = [m for m in expected_metrics if m not in available_metrics]
        if missing_metrics:
            logger.warning(f"Redfin data missing metrics: {missing_metrics}")
        
        logger.info(f"Redfin validation passed: {len(df)} records, {unique_states} states, metrics: {len(available_metrics)}")
        return True
    
    def run(self) -> bool:
        """Run the full Redfin ingestion process"""
        try:
            logger.info("Starting Redfin market data ingestion")
            
            # Fetch data
            raw_data = self.fetch_data()
            if raw_data.empty:
                logger.error("No Redfin data fetched")
                return False
            
            # Transform data
            clean_data = self.transform_data(raw_data)
            if clean_data.empty:
                logger.error("No Redfin data after transformation")
                return False
            
            # Validate data
            if not self.validate_data(clean_data):
                logger.error("Redfin data validation failed")
                return False
            
            # Save to database
            success = self.save_to_database(
                clean_data, 
                'fact_metric',
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                logger.info(f"Successfully ingested {len(clean_data)} Redfin market records")
                return True
            else:
                logger.error("Failed to save Redfin data to database")
                return False
                
        except Exception as e:
            logger.error(f"Redfin ingestion failed: {e}")
            return False
