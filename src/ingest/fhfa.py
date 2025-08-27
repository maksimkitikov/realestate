"""
FHFA (Federal Housing Finance Agency) House Price Index Ingester
Downloads HPI data by state from FHFA public CSV files
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
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from geo import get_all_states

logger = logging.getLogger(__name__)

class FHFAIngester(BaseIngester):
    """Ingester for FHFA House Price Index data by state"""
    
    def __init__(self):
        super().__init__("FHFA")
        # FHFA publishes public CSV files, no API key needed
        self.base_url = "https://www.fhfa.gov/DataTools/Downloads"
        
    def get_state_mapping(self) -> Dict[str, str]:
        """Get mapping of state names to FIPS codes"""
        states = get_all_states()
        # Create mapping from state name to FIPS and abbreviation
        name_to_fips = {}
        name_to_abbr = {}
        
        for state in states:
            name_to_fips[state['name']] = state['fips']
            name_to_abbr[state['name']] = state['abbr']
        
        return name_to_fips, name_to_abbr
    
    def fetch_hpi_data(self) -> pd.DataFrame:
        """Fetch FHFA HPI data from CSV"""
        logger.info("Fetching FHFA HPI data")
        
        # FHFA State HPI CSV URL (this URL structure is based on FHFA's actual data distribution)
        csv_url = "https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_state.csv"
        
        try:
            response = requests.get(csv_url, timeout=60)
            response.raise_for_status()
            
            # Read CSV data
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            if df.empty:
                logger.error("FHFA CSV file is empty")
                return pd.DataFrame()
            
            logger.info(f"Retrieved FHFA CSV with {len(df)} rows and columns: {list(df.columns)}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FHFA HPI data: {e}")
            # If the primary URL fails, try alternative approach with sample data structure
            return self._create_sample_hpi_data()
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing FHFA CSV: {e}")
            return self._create_sample_hpi_data()
    
    def _create_sample_hpi_data(self) -> pd.DataFrame:
        """Create sample HPI data structure for testing"""
        logger.warning("Using sample FHFA HPI data structure")
        
        states = get_all_states()
        sample_data = []
        
        # Generate sample data for last 5 years
        import random
        random.seed(42)  # For consistent sample data
        
        start_date = datetime(2019, 1, 1)
        current_date = datetime.now()
        
        for state in states[:10]:  # Limit to 10 states for sample
            base_index = 200 + random.uniform(-50, 100)
            date = start_date
            
            while date <= current_date:
                # Simulate quarterly HPI data
                if date.month in [1, 4, 7, 10]:  # Quarterly
                    growth_rate = random.uniform(-0.02, 0.03)  # -2% to +3% quarterly
                    base_index *= (1 + growth_rate)
                    
                    sample_data.append({
                        'yr': date.year,
                        'qtr': (date.month - 1) // 3 + 1,
                        'state': state['name'],
                        'index_nsa': round(base_index, 2),  # Not seasonally adjusted
                        'index_sa': round(base_index * 1.02, 2)  # Seasonally adjusted (slightly higher)
                    })
                
                # Move to next month
                if date.month == 12:
                    date = date.replace(year=date.year + 1, month=1)
                else:
                    date = date.replace(month=date.month + 1)
        
        return pd.DataFrame(sample_data)
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch FHFA HPI data - currently not implemented"""
        logger.error("FHFA HPI data fetching not implemented - no real API available")
        return pd.DataFrame()
        
        states = get_all_states()
        all_data = []
        
        # Generate data for last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1825)  # 5 years
        
        for state in states:
            # Base HPI varies by state (100-300)
            base_hpi = 150 + (hash(state['abbr']) % 200)
            
            current_date = start_date
            while current_date <= end_date:
                # Add some quarterly variation and growth trend
                quarter = (current_date.month - 1) // 3 + 1
                quarterly_variation = (hash(f"{state['abbr']}{quarter}") % 20 - 10) / 100.0
                time_trend = (current_date - start_date).days / 365.0 * 0.05  # 5% annual growth
                
                hpi_value = base_hpi * (1 + quarterly_variation + time_trend)
                
                all_data.append({
                    "date": current_date,
                    "state_fips": state['fips'],
                    "hpi_value": hpi_value,
                    "hpi_growth_yoy": quarterly_variation * 4 + time_trend * 20  # Annualized growth
                })
                
                current_date += timedelta(days=90)  # Quarterly
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} synthetic FHFA HPI records")
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform FHFA data to standard format"""
        if df.empty:
            return df
        
        logger.info(f"Transforming FHFA data with columns: {list(df.columns)}")
        
        # Create a copy
        df = df.copy()
        
        records = []
        
        for _, row in df.iterrows():
            try:
                date = row['date']
                state_fips = row['state_fips']
                geo_key = f"STATE_{state_fips}"
                
                # HPI Index
                records.append({
                    'date': date,
                    'geo_key': geo_key,
                    'geo_level': 'STATE',
                    'metric': 'HPI_INDEX',
                    'value': float(row['hpi_value']),
                    'source': 'FHFA',
                    'unit': 'index',
                    'freq': 'quarterly'
                })
                
                # HPI Growth YoY
                records.append({
                    'date': date,
                    'geo_key': geo_key,
                    'geo_level': 'STATE',
                    'metric': 'HPI_GROWTH_YOY',
                    'value': float(row['hpi_growth_yoy']),
                    'source': 'FHFA',
                    'unit': 'percent',
                    'freq': 'quarterly'
                })
            
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing FHFA row: {e}")
                continue
        
        if not records:
            logger.warning("No valid FHFA records after transformation")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(records)
        
        # Sort by metric, geo_key, and date
        result_df = result_df.sort_values(['metric', 'geo_key', 'date'])
        
        logger.info(f"Transformed {len(result_df)} FHFA HPI records for {result_df['geo_key'].nunique()} states")
        return result_df
    
    def _calculate_hpi_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate HPI year-over-year growth rates"""
        growth_records = []
        
        # Calculate YoY growth for both NSA and SA indices
        for metric in ['HPI_NSA', 'HPI_SA']:
            metric_df = df[df['metric'] == metric].copy()
            if metric_df.empty:
                continue
            
            # Sort by geo_key and date
            metric_df = metric_df.sort_values(['geo_key', 'date'])
            
            # Calculate YoY growth (4 quarters ago for quarterly data)
            metric_df['prev_year_value'] = metric_df.groupby('geo_key')['value'].shift(4)
            metric_df['yoy_growth'] = (
                (metric_df['value'] - metric_df['prev_year_value']) / 
                metric_df['prev_year_value'] * 100
            )
            
            # Add growth records
            valid_growth = metric_df[metric_df['yoy_growth'].notna()]
            for _, row in valid_growth.iterrows():
                growth_metric = metric.replace('HPI_', 'HPI_GROWTH_YOY_')
                growth_records.append({
                    'date': row['date'],
                    'geo_key': row['geo_key'],
                    'geo_level': 'STATE',
                    'metric': growth_metric,
                    'value': row['yoy_growth'],
                    'source': 'FHFA',
                    'unit': 'percent',
                    'freq': 'quarterly'
                })
        
        return pd.DataFrame(growth_records) if growth_records else pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate FHFA data quality"""
        if df.empty:
            logger.warning("FHFA dataset is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'geo_key', 'metric', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"FHFA data missing columns: {missing_cols}")
            return False
        
        # Check HPI values (should be positive indices, typically 100-500)
        hpi_data = df[df['metric'].str.contains('HPI_')]
        if not hpi_data.empty:
            invalid_hpi = hpi_data[(hpi_data['value'] < 10) | (hpi_data['value'] > 1000)]
            if not invalid_hpi.empty:
                logger.warning(f"FHFA data has {len(invalid_hpi)} records with unusual HPI values")
        
        # Check for recent data
        latest_date = df['date'].max()
        if latest_date < datetime.now() - timedelta(days=180):  # 6 months
            logger.warning(f"FHFA data may be stale, latest date: {latest_date}")
        
        # Check state coverage
        unique_states = df['geo_key'].nunique()
        if unique_states < 40:  # Some states might not have HPI data
            logger.warning(f"FHFA data covers only {unique_states} states")
        
        # Check metrics
        available_metrics = df['metric'].unique()
        expected_metrics = ['HPI_NSA', 'HPI_SA']
        missing_metrics = [m for m in expected_metrics if m not in available_metrics]
        if missing_metrics:
            logger.warning(f"FHFA data missing metrics: {missing_metrics}")
        
        logger.info(f"FHFA validation passed: {len(df)} records, {unique_states} states, metrics: {list(available_metrics)}")
        return True
    
    def run(self) -> bool:
        """Run the full FHFA ingestion process"""
        try:
            logger.info("Starting FHFA House Price Index data ingestion")
            
            # Fetch data
            raw_data = self.fetch_data()
            if raw_data.empty:
                logger.error("No FHFA data fetched")
                return False
            
            # Transform data
            clean_data = self.transform_data(raw_data)
            if clean_data.empty:
                logger.error("No FHFA data after transformation")
                return False
            
            # Validate data
            if not self.validate_data(clean_data):
                logger.error("FHFA data validation failed")
                return False
            
            # Save to database
            success = self.save_to_database(
                clean_data, 
                'fact_metric',
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                logger.info(f"Successfully ingested {len(clean_data)} FHFA HPI records")
                return True
            else:
                logger.error("Failed to save FHFA data to database")
                return False
                
        except Exception as e:
            logger.error(f"FHFA ingestion failed: {e}")
            return False
