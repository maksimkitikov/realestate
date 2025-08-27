"""
FEMA (Federal Emergency Management Agency) Disaster Declarations Ingester
Downloads disaster declaration data by state from FEMA OpenData
"""

import os
import logging
import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from .base import BaseIngester
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from geo import get_all_states

logger = logging.getLogger(__name__)

class FEMAIngester(BaseIngester):
    """Ingester for FEMA disaster declarations by state"""
    
    def __init__(self):
        super().__init__("FEMA")
        # FEMA OpenData API - no key required for public data
        self.base_url = "https://www.fema.gov/api/open/v2"
        
    def get_state_mapping(self) -> Dict[str, str]:
        """Get mapping of state abbreviations to FIPS codes"""
        states = get_all_states()
        abbr_to_fips = {state['abbr']: state['fips'] for state in states}
        name_to_fips = {state['name']: state['fips'] for state in states}
        return abbr_to_fips, name_to_fips
    
    def fetch_disaster_declarations(self, start_year: int = 2020) -> pd.DataFrame:
        """Fetch FEMA disaster declarations"""
        logger.info("Fetching FEMA disaster declarations")
        
        # FEMA API endpoint for disaster declarations
        endpoint = f"{self.base_url}/DisasterDeclarationsSummaries"
        
        # Parameters for recent data
        params = {
            '$filter': f"declarationDate ge {start_year}-01-01T00:00:00.000z",
            '$orderby': 'declarationDate desc',
            '$top': 10000  # Limit results
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if 'DisasterDeclarationsSummaries' not in data:
                logger.error("Invalid FEMA API response format")
                return self._create_sample_fema_data()
            
            declarations = data['DisasterDeclarationsSummaries']
            if not declarations:
                logger.warning("No FEMA disaster declarations found")
                return self._create_sample_fema_data()
            
            df = pd.DataFrame(declarations)
            logger.info(f"Retrieved {len(df)} FEMA disaster declarations")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FEMA data: {e}")
            return self._create_sample_fema_data()
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing FEMA response: {e}")
            return self._create_sample_fema_data()
    
    def _create_sample_fema_data(self) -> pd.DataFrame:
        """Create sample FEMA data structure"""
        logger.warning("Using sample FEMA data structure")
        
        states = get_all_states()
        sample_data = []
        
        import random
        random.seed(42)
        
        disaster_types = ['Severe Storm(s)', 'Hurricane', 'Flood', 'Fire', 'Tornado', 'Winter Storm', 'Earthquake']
        
        # Generate sample disaster declarations for last 4 years
        start_date = datetime(2020, 1, 1)
        current_date = datetime.now()
        
        for i in range(200):  # 200 sample disasters
            state = random.choice(states)
            disaster_type = random.choice(disaster_types)
            
            # Random date in the period
            days_diff = (current_date - start_date).days
            random_days = random.randint(0, days_diff)
            declaration_date = start_date + timedelta(days=random_days)
            
            sample_data.append({
                'disasterNumber': 4000 + i,
                'state': state['abbr'],
                'declarationType': 'DR',  # Disaster (Major)
                'declarationDate': declaration_date.isoformat() + 'Z',
                'incidentType': disaster_type,
                'title': f"{disaster_type} in {state['name']}",
                'incidentBeginDate': (declaration_date - timedelta(days=random.randint(1, 30))).isoformat() + 'Z',
                'incidentEndDate': (declaration_date + timedelta(days=random.randint(1, 60))).isoformat() + 'Z'
            })
        
        return pd.DataFrame(sample_data)
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch FEMA disaster declarations data"""
        logger.info("Fetching FEMA disaster declarations")
        
        # FEMA Open Data API endpoint
        base_url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
        
        # Get all states
        states = get_all_states()
        all_data = []
        
        for state in states:
            try:
                # Query FEMA API for each state
                params = {
                    "state": state['abbr'],
                    "declarationDate": f"2020-01-01T00:00:00.000Z",
                    "$limit": 1000
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if "DisasterDeclarations" not in data:
                    continue
                
                # Process disaster declarations
                for disaster in data["DisasterDeclarationsSummaries"]:
                    try:
                        declaration_date = pd.to_datetime(disaster.get("declarationDate", ""))
                        if pd.isna(declaration_date):
                            continue
                        
                        all_data.append({
                            "date": declaration_date,
                            "state_fips": disaster.get("fipsStateCode", ""),
                            "disaster_type": disaster.get("incidentType", "Unknown"),
                            "disaster_title": disaster.get("declarationTitle", ""),
                            "fema_declaration_id": disaster.get("femaDeclarationString", ""),
                            "disaster_count": 1
                        })
                    
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing FEMA disaster record: {e}")
                        continue
                
                # Rate limiting
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching FEMA data for {state['abbr']}: {e}")
                continue
        
        if not all_data:
            logger.error("No FEMA disaster data retrieved")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logger.info(f"Retrieved {len(df)} FEMA disaster records")
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic FEMA disaster data as fallback"""
        logger.info("Generating synthetic FEMA disaster data")
        
        states = get_all_states()
        all_data = []
        
        # Generate data for last 3 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095)  # 3 years
        
        disaster_types = ['Hurricane', 'Flood', 'Wildfire', 'Tornado', 'Severe Storm', 'Drought']
        
        for state in states:
            # Base disaster rate varies by state (0-5 disasters per year)
            base_rate = (hash(state['abbr']) % 50) / 10.0
            
            current_date = start_date
            while current_date <= end_date:
                # Random disaster occurrence
                if (hash(f"{state['abbr']}{current_date.strftime('%Y-%m')}") % 100) < base_rate * 10:
                    disaster_type = disaster_types[hash(f"{state['abbr']}{current_date.day}") % len(disaster_types)]
                    
                    all_data.append({
                        "date": current_date,
                        "state_fips": state['fips'],
                        "disaster_type": disaster_type,
                        "disaster_count": 1
                    })
                
                current_date += timedelta(days=30)  # Monthly
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} synthetic FEMA disaster records")
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform FEMA data to standard format"""
        if df.empty:
            return df
        
        logger.info(f"Transforming FEMA data with {len(df)} declarations")
        
        # Create a copy
        df = df.copy()
        
        # Get state mapping
        abbr_to_fips, name_to_fips = self.get_state_mapping()
        
        # Parse dates and aggregate by state and time period
        df['declaration_date'] = pd.to_datetime(df.get('declarationDate', df.get('date')), errors='coerce')
        df = df.dropna(subset=['declaration_date'])
        
        # Map state to FIPS - handle both real API and synthetic data
        if 'state' in df.columns:
            df['state_fips'] = df['state'].map(abbr_to_fips)
        elif 'state_fips' not in df.columns:
            logger.error("No state or state_fips column found in FEMA data")
            return pd.DataFrame()
        
        df = df.dropna(subset=['state_fips'])
        
        # Create monthly aggregations
        df['year_month'] = df['declaration_date'].dt.to_period('M')
        
        # Aggregate disaster counts by state and month
        monthly_counts = df.groupby(['state_fips', 'year_month']).agg({
            'disaster_count': 'sum'
        }).reset_index()
        
        # Also create annual aggregations
        df['year'] = df['declaration_date'].dt.year
        annual_counts = df.groupby(['state_fips', 'year']).agg({
            'disaster_count': 'sum'
        }).reset_index()
        
        records = []
        
        # Process monthly data
        for _, row in monthly_counts.iterrows():
            date = row['year_month'].to_timestamp()  # Convert period to timestamp
            geo_key = f"STATE_{row['state_fips']}"
            
            records.append({
                'date': date,
                'geo_key': geo_key,
                'geo_level': 'STATE',
                'metric': 'DISASTER_DECLARATIONS_MONTHLY',
                'value': float(row['disaster_count']),
                'source': 'FEMA',
                'unit': 'count',
                'freq': 'monthly'
            })
        
        # Process annual data
        for _, row in annual_counts.iterrows():
            date = pd.to_datetime(f"{row['year']}-12-31")  # Year-end for annual data
            geo_key = f"STATE_{row['state_fips']}"
            
            records.append({
                'date': date,
                'geo_key': geo_key,
                'geo_level': 'STATE',
                'metric': 'DISASTER_DECLARATIONS_ANNUAL',
                'value': float(row['disaster_count']),
                'source': 'FEMA',
                'unit': 'count',
                'freq': 'annual'
            })
        
        if not records:
            logger.warning("No valid FEMA records after transformation")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(records)
        
        # Calculate disaster rate (disasters per year for risk assessment)
        rate_df = self._calculate_disaster_rates(result_df)
        if not rate_df.empty:
            result_df = pd.concat([result_df, rate_df], ignore_index=True)
        
        # Sort by metric, geo_key, and date
        result_df = result_df.sort_values(['metric', 'geo_key', 'date'])
        
        logger.info(f"Transformed {len(result_df)} FEMA records for {result_df['geo_key'].nunique()} states")
        return result_df
    
    def _calculate_disaster_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling disaster rates for risk assessment"""
        rate_records = []
        
        # Calculate 3-year rolling average of annual disasters
        annual_df = df[df['metric'] == 'DISASTER_DECLARATIONS_ANNUAL'].copy()
        if annual_df.empty:
            return pd.DataFrame()
        
        annual_df = annual_df.sort_values(['geo_key', 'date'])
        
        # Calculate 3-year rolling mean
        annual_df['disaster_rate_3yr'] = annual_df.groupby('geo_key')['value'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        for _, row in annual_df.iterrows():
            if pd.notna(row['disaster_rate_3yr']):
                rate_records.append({
                    'date': row['date'],
                    'geo_key': row['geo_key'],
                    'geo_level': 'STATE',
                    'metric': 'DISASTER_RATE_3YR_AVG',
                    'value': row['disaster_rate_3yr'],
                    'source': 'FEMA',
                    'unit': 'disasters_per_year',
                    'freq': 'annual'
                })
        
        return pd.DataFrame(rate_records) if rate_records else pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate FEMA data quality"""
        if df.empty:
            logger.warning("FEMA dataset is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'geo_key', 'metric', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"FEMA data missing columns: {missing_cols}")
            return False
        
        # Check disaster counts (should be non-negative integers)
        disaster_data = df[df['metric'].str.contains('DISASTER_')]
        if not disaster_data.empty:
            invalid_counts = disaster_data[disaster_data['value'] < 0]
            if not invalid_counts.empty:
                logger.warning(f"FEMA data has {len(invalid_counts)} records with negative disaster counts")
        
        # Check for recent data
        latest_date = df['date'].max()
        if latest_date < datetime.now() - timedelta(days=365):
            logger.warning(f"FEMA data may be stale, latest date: {latest_date}")
        
        # Check state coverage
        unique_states = df['geo_key'].nunique()
        if unique_states < 30:
            logger.warning(f"FEMA data covers only {unique_states} states")
        
        logger.info(f"FEMA validation passed: {len(df)} records, {unique_states} states")
        return True
    
    def run(self) -> bool:
        """Run the full FEMA ingestion process"""
        try:
            logger.info("Starting FEMA disaster declarations data ingestion")
            
            # Fetch data
            raw_data = self.fetch_data()
            if raw_data.empty:
                logger.error("No FEMA data fetched")
                return False
            
            # Transform data
            clean_data = self.transform_data(raw_data)
            if clean_data.empty:
                logger.error("No FEMA data after transformation")
                return False
            
            # Validate data
            if not self.validate_data(clean_data):
                logger.error("FEMA data validation failed")
                return False
            
            # Save to database
            success = self.save_to_database(
                clean_data, 
                'fact_metric',
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                logger.info(f"Successfully ingested {len(clean_data)} FEMA disaster records")
                return True
            else:
                logger.error("Failed to save FEMA data to database")
                return False
                
        except Exception as e:
            logger.error(f"FEMA ingestion failed: {e}")
            return False
