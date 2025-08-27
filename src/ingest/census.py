"""
Census ACS (American Community Survey) Ingester
Downloads demographic and economic data by state from US Census Bureau
"""

import os
import logging
import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .base import BaseIngester
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from geo import get_all_states

logger = logging.getLogger(__name__)

class CensusIngester(BaseIngester):
    """Ingester for Census ACS demographic data by state"""
    
    def __init__(self):
        super().__init__("CENSUS")
        self.api_key = os.getenv("CENSUS_API_KEY")
        self.base_url = "https://api.census.gov/data"
        
        if not self.api_key:
            raise ValueError("CENSUS_API_KEY is required")
    
    def get_state_fips_list(self) -> List[str]:
        """Get list of state FIPS codes for Census API"""
        states = get_all_states()
        return [state['fips'] for state in states if state['abbr'] not in ['PR', 'VI', 'GU', 'AS', 'MP']]
    
    def fetch_acs_data(self, year: int = 2022) -> pd.DataFrame:
        """Fetch ACS data from Census API"""
        logger.info(f"Fetching Census ACS data for {year}")
        
        # ACS 5-year estimates variables we want
        variables = {
            "B01003_001E": "TOTAL_POPULATION",          # Total population
            "B19013_001E": "MEDIAN_HOUSEHOLD_INCOME",   # Median household income
            "B25077_001E": "MEDIAN_HOME_VALUE",         # Median home value
            "B08303_001E": "TOTAL_COMMUTERS",           # Total commuters
            "B08301_010E": "PUBLIC_TRANSPORT_COMMUTERS", # Public transportation commuters
            "B15003_022E": "BACHELORS_DEGREE",          # Bachelor's degree
            "B15003_001E": "TOTAL_EDUCATION_UNIVERSE",  # Total education universe
            "B12001_007E": "DIVORCED_MALE",             # Divorced males
            "B12001_016E": "DIVORCED_FEMALE",           # Divorced females
            "B12001_001E": "TOTAL_MARITAL_STATUS",      # Total marital status universe
            "B25003_002E": "OWNER_OCCUPIED_HOUSING",    # Owner-occupied housing units
            "B25003_001E": "TOTAL_HOUSING_UNITS",       # Total housing units
        }
        
        variable_list = ",".join(variables.keys())
        state_list = ",".join(self.get_state_fips_list())
        
        url = f"{self.base_url}/{year}/acs/acs5"
        params = {
            "get": variable_list,
            "for": "state:*",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or len(data) < 2:  # Should have header + data rows
                logger.error("Invalid Census API response format")
                return pd.DataFrame()
            
            # Convert to DataFrame
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            
            # Clean up data
            records = []
            for _, row in df.iterrows():
                state_fips = row.get('state', '').zfill(2)
                if not state_fips or len(state_fips) != 2:
                    continue
                
                date = pd.to_datetime(f"{year}-12-31")  # Annual data, use year-end
                
                # Process each variable
                for var_code, metric_name in variables.items():
                    value_str = row.get(var_code, '')
                    
                    try:
                        if value_str and value_str not in ['', 'null', '-', '(X)']:
                            value = float(value_str)
                            
                            records.append({
                                "date": date,
                                "state_fips": state_fips,
                                "metric": metric_name,
                                "value": value,
                                "var_code": var_code
                            })
                    
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse Census value for {metric_name}: {value_str}")
                        continue
            
            if not records:
                logger.warning("No valid Census records parsed")
                return pd.DataFrame()
            
            result_df = pd.DataFrame(records)
            logger.info(f"Retrieved {len(result_df)} Census ACS records for {year}")
            return result_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Census ACS data: {e}")
            return pd.DataFrame()
    
    def fetch_data(self, start_year: int = 2019) -> pd.DataFrame:
        """Fetch Census data for multiple years"""
        all_data = []
        current_year = datetime.now().year
        
        # Census ACS 5-year estimates are usually available with 1-2 year lag
        end_year = min(current_year - 1, 2022)
        
        for year in range(start_year, end_year + 1):
            year_data = self.fetch_acs_data(year)
            if not year_data.empty:
                all_data.append(year_data)
            
            # Rate limiting
            time.sleep(1)
        
        if not all_data:
            logger.warning("No Census data retrieved")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total Census records")
        return combined_df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Census data to standard format"""
        if df.empty:
            return df
        
        # Create a copy
        df = df.copy()
        
        # Map state FIPS to geo keys
        state_fips_map = {state['fips']: state['fips'] for state in get_all_states()}
        df['geo_key'] = df['state_fips'].apply(lambda x: f"STATE_{x}" if x in state_fips_map else None)
        
        # Remove rows where geo mapping failed
        df = df.dropna(subset=['geo_key'])
        
        # Standardize columns
        df['geo_level'] = 'STATE'
        df['source'] = 'CENSUS_ACS'
        df['freq'] = 'annual'
        
        # Set units based on metric type
        unit_mapping = {
            'TOTAL_POPULATION': 'count',
            'MEDIAN_HOUSEHOLD_INCOME': 'dollars',
            'MEDIAN_HOME_VALUE': 'dollars',
            'TOTAL_COMMUTERS': 'count',
            'PUBLIC_TRANSPORT_COMMUTERS': 'count',
            'BACHELORS_DEGREE': 'count',
            'TOTAL_EDUCATION_UNIVERSE': 'count',
            'DIVORCED_MALE': 'count',
            'DIVORCED_FEMALE': 'count',
            'TOTAL_MARITAL_STATUS': 'count',
            'OWNER_OCCUPIED_HOUSING': 'count',
            'TOTAL_HOUSING_UNITS': 'count'
        }
        
        df['unit'] = df['metric'].map(unit_mapping).fillna('count')
        
        # Calculate derived metrics
        derived_df = self._calculate_derived_metrics(df)
        
        # Combine original and derived data
        if not derived_df.empty:
            df = pd.concat([df, derived_df], ignore_index=True)
        
        # Select final columns
        result = df[['date', 'geo_key', 'geo_level', 'metric', 'value', 'source', 'unit', 'freq']].copy()
        
        # Sort by metric, geo_key, and date
        result = result.sort_values(['metric', 'geo_key', 'date'])
        
        logger.info(f"Transformed {len(result)} Census records for {df['geo_key'].nunique()} states")
        return result
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from Census data"""
        derived_records = []
        
        # Group by state and date for calculations
        for (geo_key, date), group in df.groupby(['geo_key', 'date']):
            metrics = {row['metric']: row['value'] for _, row in group.iterrows()}
            
            # Education rate (% with bachelor's degree)
            if 'BACHELORS_DEGREE' in metrics and 'TOTAL_EDUCATION_UNIVERSE' in metrics:
                total_edu = metrics['TOTAL_EDUCATION_UNIVERSE']
                if total_edu > 0:
                    edu_rate = (metrics['BACHELORS_DEGREE'] / total_edu) * 100
                    derived_records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'EDUCATION_RATE',
                        'value': edu_rate,
                        'source': 'CENSUS_ACS',
                        'unit': 'percent',
                        'freq': 'annual'
                    })
            
            # Divorce rate (% divorced)
            if all(m in metrics for m in ['DIVORCED_MALE', 'DIVORCED_FEMALE', 'TOTAL_MARITAL_STATUS']):
                total_divorced = metrics['DIVORCED_MALE'] + metrics['DIVORCED_FEMALE']
                total_marital = metrics['TOTAL_MARITAL_STATUS']
                if total_marital > 0:
                    divorce_rate = (total_divorced / total_marital) * 100
                    derived_records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'DIVORCE_RATE',
                        'value': divorce_rate,
                        'source': 'CENSUS_ACS',
                        'unit': 'percent',
                        'freq': 'annual'
                    })
            
            # Homeownership rate
            if 'OWNER_OCCUPIED_HOUSING' in metrics and 'TOTAL_HOUSING_UNITS' in metrics:
                total_housing = metrics['TOTAL_HOUSING_UNITS']
                if total_housing > 0:
                    ownership_rate = (metrics['OWNER_OCCUPIED_HOUSING'] / total_housing) * 100
                    derived_records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'HOMEOWNERSHIP_RATE',
                        'value': ownership_rate,
                        'source': 'CENSUS_ACS',
                        'unit': 'percent',
                        'freq': 'annual'
                    })
            
            # Public transport usage rate
            if 'PUBLIC_TRANSPORT_COMMUTERS' in metrics and 'TOTAL_COMMUTERS' in metrics:
                total_commuters = metrics['TOTAL_COMMUTERS']
                if total_commuters > 0:
                    transit_rate = (metrics['PUBLIC_TRANSPORT_COMMUTERS'] / total_commuters) * 100
                    derived_records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'PUBLIC_TRANSIT_RATE',
                        'value': transit_rate,
                        'source': 'CENSUS_ACS',
                        'unit': 'percent',
                        'freq': 'annual'
                    })
        
        if derived_records:
            return pd.DataFrame(derived_records)
        else:
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate Census data quality"""
        if df.empty:
            logger.warning("Census dataset is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'geo_key', 'metric', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Census data missing columns: {missing_cols}")
            return False
        
        # Check for reasonable population values
        pop_data = df[df['metric'] == 'TOTAL_POPULATION']
        if not pop_data.empty:
            invalid_pop = pop_data[(pop_data['value'] < 100000) | (pop_data['value'] > 50000000)]
            if not invalid_pop.empty:
                logger.warning(f"Census data has {len(invalid_pop)} records with unusual population values")
        
        # Check for recent data
        latest_date = df['date'].max()
        if latest_date < datetime.now() - timedelta(days=730):  # 2 years
            logger.warning(f"Census data may be stale, latest date: {latest_date}")
        
        # Check state coverage
        unique_states = df['geo_key'].nunique()
        if unique_states < 45:
            logger.warning(f"Census data covers only {unique_states} states")
        
        # Check metrics coverage
        available_metrics = df['metric'].unique()
        expected_core_metrics = ['TOTAL_POPULATION', 'MEDIAN_HOUSEHOLD_INCOME', 'MEDIAN_HOME_VALUE']
        missing_core = [m for m in expected_core_metrics if m not in available_metrics]
        if missing_core:
            logger.warning(f"Census data missing core metrics: {missing_core}")
        
        logger.info(f"Census validation passed: {len(df)} records, {unique_states} states, metrics: {len(available_metrics)}")
        return True
    
    def run(self) -> bool:
        """Run the full Census ingestion process"""
        try:
            logger.info("Starting Census ACS demographic data ingestion")
            
            # Fetch data
            raw_data = self.fetch_data()
            if raw_data.empty:
                logger.error("No Census data fetched")
                return False
            
            # Transform data
            clean_data = self.transform_data(raw_data)
            if clean_data.empty:
                logger.error("No Census data after transformation")
                return False
            
            # Validate data
            if not self.validate_data(clean_data):
                logger.error("Census data validation failed")
                return False
            
            # Save to database
            success = self.save_to_database(
                clean_data, 
                'fact_metric',
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                logger.info(f"Successfully ingested {len(clean_data)} Census demographic records")
                return True
            else:
                logger.error("Failed to save Census data to database")
                return False
                
        except Exception as e:
            logger.error(f"Census ingestion failed: {e}")
            return False
