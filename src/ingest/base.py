"""
Base class for data ingestion
Provides common functionality for all data sources
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseIngester(ABC):
    """Base class for all data ingesters"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from source"""
        pass
    
    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data to standard format"""
        pass
    
    def ingest(self, **kwargs) -> pd.DataFrame:
        """Main ingestion method"""
        try:
            self.logger.info(f"Starting ingestion from {self.source_name}")
            raw_data = self.fetch_data(**kwargs)
            transformed_data = self.transform_data(raw_data)
            self.logger.info(f"Successfully ingested {len(transformed_data)} records from {self.source_name}")
            return transformed_data
        except Exception as e:
            self.logger.error(f"Error ingesting from {self.source_name}: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Response:
        """Make HTTP request with retry logic"""
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Request failed for {url}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """Validate data has required columns"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types"""
        # Standard column mapping
        column_mapping = {
            'date': 'date',
            'Date': 'date',
            'DATE': 'date',
            'value': 'value',
            'Value': 'value',
            'VALUE': 'value',
            'metric': 'metric',
            'Metric': 'metric',
            'METRIC': 'metric',
            'geo_key': 'geo_key',
            'geo': 'geo_key',
            'Geo': 'geo_key',
            'GEO': 'geo_key'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Ensure value column is numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df
    
    def save_to_database(self, df: pd.DataFrame, table_name: str, unique_columns: list) -> bool:
        """Save data to database using DatabaseManager"""
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from database import DatabaseManager
            
            db = DatabaseManager()
            success = db.upsert_fact_data(table_name, df, unique_columns)
            
            if success:
                self.logger.info(f"Successfully saved {len(df)} records to {table_name}")
                return True
            else:
                self.logger.error(f"Failed to save data to {table_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            return False
