"""
FRED (Federal Reserve Economic Data) data ingestion
Handles economic indicators from FRED API
"""

import pandas as pd
import os
from typing import List, Dict
from .base import BaseIngester
import json

class FREDIngester(BaseIngester):
    """Ingester for FRED economic data"""
    
    def __init__(self):
        super().__init__("FRED")
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY not found in environment variables")
        
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Series to fetch
        self.series = [
            "MORTGAGE30US",  # 30-Year Fixed Rate Mortgage Average
            "CPIAUCSL",      # Consumer Price Index
            "DGS10",         # 10-Year Treasury Rate
            "T10Y2Y",        # 10-Year minus 2-Year Treasury Spread
            "UNRATE"         # Unemployment Rate
        ]
    
    def fetch_data(self, series_ids: List[str] = None) -> pd.DataFrame:
        """Fetch data for specified series"""
        if series_ids is None:
            series_ids = self.series
        
        all_data = []
        
        for series_id in series_ids:
            try:
                self.logger.info(f"Fetching data for series: {series_id}")
                
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'file_type': 'json',
                    'sort_order': 'asc'
                }
                
                response = self.make_request(self.base_url, params=params)
                data = response.json()
                
                if 'observations' in data:
                    for obs in data['observations']:
                        all_data.append({
                            'series_id': series_id,
                            'date': obs['date'],
                            'value': obs['value'],
                            'realtime_start': obs.get('realtime_start'),
                            'realtime_end': obs.get('realtime_end')
                        })
                
            except Exception as e:
                self.logger.error(f"Error fetching series {series_id}: {str(e)}")
                continue
        
        return pd.DataFrame(all_data)
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform FRED data to standard format"""
        if df.empty:
            return df
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert values to numeric, handling '.' (missing values)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Filter out missing values
        df = df.dropna(subset=['value'])
        
        # Add standard columns
        df['geo_key'] = 'US_00'  # FRED data is national
        df['geo_level'] = 'US'
        df['metric'] = df['series_id']
        df['source'] = 'FRED'
        
        # Apply unit and frequency mapping
        df['unit'] = df['series_id'].map(self._get_unit_mapping())
        df['freq'] = df['series_id'].map(self._get_frequency_mapping())
        
        # Select final columns
        result = df[['date', 'geo_key', 'geo_level', 'metric', 'value', 'source', 'unit', 'freq']].copy()
        
        return result
    
    def _get_unit_mapping(self) -> Dict[str, str]:
        """Get unit mapping for series"""
        return {
            'MORTGAGE30US': 'percent',
            'CPIAUCSL': 'index',
            'DGS10': 'percent',
            'T10Y2Y': 'percent',
            'UNRATE': 'percent'
        }
    
    def _get_frequency_mapping(self) -> Dict[str, str]:
        """Get frequency mapping for series"""
        return {
            'MORTGAGE30US': 'weekly',
            'CPIAUCSL': 'monthly',
            'DGS10': 'daily',
            'T10Y2Y': 'daily',
            'UNRATE': 'monthly'
        }
    
    def _get_unit_for_series(self, series_id: str) -> str:
        """Get unit for series (deprecated, use _get_unit_mapping)"""
        units = self._get_unit_mapping()
        return units.get(series_id, 'unknown')
    
    def _get_frequency_for_series(self, series_id: str) -> str:
        """Get frequency for series (deprecated, use _get_frequency_mapping)"""
        frequencies = self._get_frequency_mapping()
        return frequencies.get(series_id, 'unknown')
    
    def run(self) -> bool:
        """Run the full FRED ingestion process"""
        try:
            self.logger.info("Starting FRED data ingestion")
            
            # Fetch data
            clean_data = self.ingest()
            if clean_data.empty:
                self.logger.error("No FRED data after ingestion")
                return False
            
            # Save to database using DatabaseManager
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from database import DatabaseManager
            
            db = DatabaseManager()
            success = db.upsert_fact_data(
                'fact_metric', 
                clean_data,
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                self.logger.info(f"Successfully ingested {len(clean_data)} FRED records")
                return True
            else:
                self.logger.error("Failed to save FRED data to database")
                return False
                
        except Exception as e:
            self.logger.error(f"FRED ingestion failed: {e}")
            return False
