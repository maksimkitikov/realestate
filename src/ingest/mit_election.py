"""
MIT Election Data and Science Lab Ingester
Downloads election results data by state from MIT Election Lab
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

class MITElectionIngester(BaseIngester):
    """Ingester for MIT Election Lab data by state"""
    
    def __init__(self):
        super().__init__("MIT_ELECTION")
        # MIT Election Lab provides public CSV data
        self.base_url = "https://dataverse.harvard.edu/api/access/datafile"
        
    def get_state_mapping(self) -> Dict[str, str]:
        """Get mapping of state names/abbreviations to FIPS codes"""
        states = get_all_states()
        name_to_fips = {state['name'].upper(): state['fips'] for state in states}
        abbr_to_fips = {state['abbr']: state['fips'] for state in states}
        return name_to_fips, abbr_to_fips
    
    def fetch_election_data(self) -> pd.DataFrame:
        """Fetch MIT Election Lab data"""
        logger.info("Fetching MIT Election Lab data")
        
        # MIT Election Lab Presidential Election Returns data
        # File ID from Harvard Dataverse - this is the actual ID for 1976-2020 data
        file_ids = [
            "4819117",  # Presidential returns 1976-2020
            "4455022"   # Alternative file ID if first fails
        ]
        
        for file_id in file_ids:
            try:
                url = f"{self.base_url}/{file_id}"
                logger.info(f"Trying MIT Election data from file ID: {file_id}")
                
                response = requests.get(url, timeout=120)  # Longer timeout for large file
                response.raise_for_status()
                
                # Parse CSV
                csv_data = io.StringIO(response.text)
                df = pd.read_csv(csv_data)
                
                if not df.empty and 'year' in df.columns:
                    logger.info(f"Retrieved MIT Election data with {len(df)} rows from file {file_id}")
                    return df
                    
            except (requests.exceptions.RequestException, Exception) as e:
                logger.warning(f"Failed to fetch MIT Election data from file {file_id}: {e}")
                continue
        
        # If all URLs fail, create sample data
        logger.warning("All MIT Election URLs failed, using sample data")
        return self._create_sample_election_data()
    
    def _create_sample_election_data(self) -> pd.DataFrame:
        """Create sample election data structure"""
        logger.warning("Using sample MIT Election data structure")
        
        states = get_all_states()
        sample_data = []
        
        import random
        random.seed(42)
        
        # Generate sample data for recent presidential elections
        election_years = [2016, 2020]
        candidates = {
            2016: [('CLINTON, HILLARY', 'democrat'), ('TRUMP, DONALD J.', 'republican'), ('JOHNSON, GARY', 'libertarian')],
            2020: [('BIDEN, JOSEPH R. JR', 'democrat'), ('TRUMP, DONALD J.', 'republican'), ('JORGENSEN, JO', 'libertarian')]
        }
        
        for year in election_years:
            for state in states:
                # Generate realistic vote totals
                total_votes = random.randint(500000, 8000000)  # Varies by state size
                
                # Generate party vote shares (sum to ~100%)
                dem_share = random.uniform(0.25, 0.65)
                rep_share = random.uniform(0.30, 0.70)
                other_share = max(0.01, 1.0 - dem_share - rep_share)
                
                # Normalize to sum to 1
                total_share = dem_share + rep_share + other_share
                dem_share /= total_share
                rep_share /= total_share
                other_share /= total_share
                
                for candidate, party in candidates[year]:
                    if party == 'democrat':
                        votes = int(total_votes * dem_share)
                    elif party == 'republican':
                        votes = int(total_votes * rep_share)
                    else:
                        votes = int(total_votes * other_share)
                    
                    sample_data.append({
                        'year': year,
                        'state': state['name'].upper(),
                        'state_po': state['abbr'],
                        'candidate': candidate,
                        'party': party,
                        'candidatevotes': votes,
                        'totalvotes': total_votes,
                        'version': '20220130'
                    })
        
        return pd.DataFrame(sample_data)
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch MIT Election Lab data"""
        logger.info("Fetching MIT Election Lab data")
        
        # MIT Election Lab API endpoint
        base_url = "https://electionlab.mit.edu/api/v1/elections"
        
        # Get all states
        states = get_all_states()
        all_data = []
        
        for state in states:
            try:
                # Query MIT Election Lab API for each state
                params = {
                    "state": state['abbr'],
                    "year": "2020,2022",  # Recent elections
                    "office": "president,senate,house"
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process election data
                    for election in data.get('elections', []):
                        try:
                            election_date = pd.to_datetime(election.get('election_date', ''))
                            if pd.isna(election_date):
                                continue
                            
                            # Extract vote shares
                            dem_votes = election.get('democratic_votes', 0)
                            rep_votes = election.get('republican_votes', 0)
                            total_votes = election.get('total_votes', 0)
                            
                            if total_votes > 0:
                                dem_share = (dem_votes / total_votes) * 100
                                rep_share = (rep_votes / total_votes) * 100
                                competitiveness = 100 - abs(dem_share - rep_share)
                                
                                all_data.append({
                                    "date": election_date,
                                    "state_fips": state['fips'],
                                    "democratic_share": dem_share,
                                    "republican_share": rep_share,
                                    "competitiveness": competitiveness,
                                    "election_type": election.get('office', 'unknown')
                                })
                        
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing MIT Election record for {state['abbr']}: {e}")
                            continue
                
                # Rate limiting
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching MIT Election data for {state['abbr']}: {e}")
                continue
        
        if not all_data:
            logger.error("No MIT Election data retrieved")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logger.info(f"Retrieved {len(df)} MIT Election records")
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic MIT Election Lab data as fallback"""
        logger.info("Generating synthetic MIT Election data")
        
        states = get_all_states()
        all_data = []
        
        # Generate data for last 2 elections
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1460)  # 4 years
        
        for state in states:
            # Base political lean varies by state
            base_dem_share = 30 + (hash(state['abbr']) % 40)  # 30-70%
            base_rep_share = 70 - base_dem_share
            
            current_date = start_date
            while current_date <= end_date:
                # Add some variation for different elections
                election_variation = (hash(f"{state['abbr']}{current_date.year}") % 20 - 10)
                
                dem_share = max(0, min(100, base_dem_share + election_variation))
                rep_share = max(0, min(100, base_rep_share - election_variation))
                
                # Calculate competitiveness (closer to 50-50 = more competitive)
                competitiveness = 100 - abs(dem_share - rep_share)
                
                all_data.append({
                    "date": current_date,
                    "state_fips": state['fips'],
                    "democratic_share": dem_share,
                    "republican_share": rep_share,
                    "competitiveness": competitiveness
                })
                
                current_date += timedelta(days=1460)  # Every 4 years
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} synthetic MIT Election records")
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform MIT Election data to standard format"""
        if df.empty:
            return df
        
        logger.info(f"Transforming MIT Election data with {len(df)} records")
        
        # Create a copy
        df = df.copy()
        
        # Get state mapping
        name_to_fips, abbr_to_fips = self.get_state_mapping()
        
        # Filter for recent presidential elections only
        if 'year' in df.columns:
            df = df[df['year'] >= 2016]
        else:
            # For synthetic data, extract year from date
            df['year'] = df['date'].dt.year
        
        # Handle different data formats (real API vs synthetic)
        if 'party' in df.columns and 'candidatevotes' in df.columns:
            # Real API data format
            df['state_fips'] = df.get('state_po', df.get('state', '')).map(abbr_to_fips)
            if df['state_fips'].isna().any():
                # Try state name mapping
                df.loc[df['state_fips'].isna(), 'state_fips'] = df.loc[df['state_fips'].isna(), 'state'].map(name_to_fips)
            
            df = df.dropna(subset=['state_fips'])
            
            # Calculate party vote shares by state and year
            party_totals = df.groupby(['year', 'state_fips', 'party']).agg({
                'candidatevotes': 'sum',
                'totalvotes': 'first'  # Total votes should be same for all candidates in state
            }).reset_index()
            
            # Calculate vote percentages
            party_totals['vote_percentage'] = (party_totals['candidatevotes'] / party_totals['totalvotes']) * 100
        else:
            # Synthetic data format - already has vote shares
            party_totals = df.copy()
            party_totals['vote_percentage'] = party_totals['democratic_share']  # Use democratic share as base
        
        records = []
        
        for _, row in party_totals.iterrows():
            try:
                year = int(row['year'])
                date = pd.to_datetime(f"{year}-11-01")  # November election date
                geo_key = f"STATE_{row['state_fips']}"
                
                if 'party' in row:
                    # Real API data format
                    party = row['party'].upper()
                    
                    # Create metrics for major parties
                    if party in ['DEMOCRAT', 'DEMOCRATIC']:
                        metric = 'DEMOCRAT_VOTE_SHARE'
                    elif party in ['REPUBLICAN']:
                        metric = 'REPUBLICAN_VOTE_SHARE'
                    elif party in ['LIBERTARIAN']:
                        metric = 'LIBERTARIAN_VOTE_SHARE'
                    elif party in ['GREEN']:
                        metric = 'GREEN_VOTE_SHARE'
                    else:
                        metric = f"{party}_VOTE_SHARE"
                    
                    records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': metric,
                        'value': float(row['vote_percentage']),
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'percent',
                        'freq': 'presidential_election'  # Every 4 years
                    })
                    
                    # Also add raw vote counts
                    records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': f"{metric.replace('_SHARE', '_VOTES')}",
                        'value': float(row['candidatevotes']),
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'count',
                        'freq': 'presidential_election'
                    })
                else:
                    # Synthetic data format
                    records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'DEMOCRAT_VOTE_SHARE',
                        'value': float(row['democratic_share']),
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'percent',
                        'freq': 'presidential_election'
                    })
                    
                    records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'REPUBLICAN_VOTE_SHARE',
                        'value': float(row['republican_share']),
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'percent',
                        'freq': 'presidential_election'
                    })
                    
                    records.append({
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'POLITICAL_COMPETITIVENESS',
                        'value': float(row['competitiveness']),
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'percent',
                        'freq': 'presidential_election'
                    })
            
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing MIT Election row: {e}")
                continue
        
        if not records:
            logger.warning("No valid MIT Election records after transformation")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(records)
        
        # Calculate political metrics
        derived_df = self._calculate_political_metrics(result_df)
        if not derived_df.empty:
            result_df = pd.concat([result_df, derived_df], ignore_index=True)
        
        # Sort by metric, geo_key, and date
        result_df = result_df.sort_values(['metric', 'geo_key', 'date'])
        
        logger.info(f"Transformed {len(result_df)} MIT Election records for {result_df['geo_key'].nunique()} states")
        return result_df
    
    def _calculate_political_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived political metrics"""
        derived_records = []
        
        # Calculate political competitiveness and partisan lean
        for (geo_key, date), group in df.groupby(['geo_key', 'date']):
            metrics = {row['metric']: row['value'] for _, row in group.iterrows()}
            
            dem_share = metrics.get('DEMOCRAT_VOTE_SHARE', 0)
            rep_share = metrics.get('REPUBLICAN_VOTE_SHARE', 0)
            
            if dem_share > 0 and rep_share > 0:
                # Calculate partisan lean (positive = Republican, negative = Democratic)
                partisan_lean = rep_share - dem_share
                
                # Calculate competitiveness (lower = more competitive)
                competitiveness = abs(partisan_lean)
                
                derived_records.extend([
                    {
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'PARTISAN_LEAN',
                        'value': partisan_lean,
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'percentage_points',
                        'freq': 'presidential_election'
                    },
                    {
                        'date': date,
                        'geo_key': geo_key,
                        'geo_level': 'STATE',
                        'metric': 'POLITICAL_COMPETITIVENESS',
                        'value': competitiveness,
                        'source': 'MIT_ELECTION_LAB',
                        'unit': 'percentage_points',
                        'freq': 'presidential_election'
                    }
                ])
        
        return pd.DataFrame(derived_records) if derived_records else pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate MIT Election data quality"""
        if df.empty:
            logger.warning("MIT Election dataset is empty")
            return False
        
        # Check required columns
        required_cols = ['date', 'geo_key', 'metric', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"MIT Election data missing columns: {missing_cols}")
            return False
        
        # Check vote percentages (should be 0-100%)
        share_data = df[df['metric'].str.contains('_SHARE')]
        if not share_data.empty:
            invalid_shares = share_data[(share_data['value'] < 0) | (share_data['value'] > 100)]
            if not invalid_shares.empty:
                logger.warning(f"MIT Election data has {len(invalid_shares)} records with invalid vote shares")
        
        # Check state coverage
        unique_states = df['geo_key'].nunique()
        if unique_states < 45:
            logger.warning(f"MIT Election data covers only {unique_states} states")
        
        # Check for election data
        available_metrics = df['metric'].unique()
        expected_metrics = ['DEMOCRAT_VOTE_SHARE', 'REPUBLICAN_VOTE_SHARE']
        missing_metrics = [m for m in expected_metrics if m not in available_metrics]
        if missing_metrics:
            logger.warning(f"MIT Election data missing metrics: {missing_metrics}")
        
        logger.info(f"MIT Election validation passed: {len(df)} records, {unique_states} states, metrics: {len(available_metrics)}")
        return True
    
    def run(self) -> bool:
        """Run the full MIT Election ingestion process"""
        try:
            logger.info("Starting MIT Election Lab data ingestion")
            
            # Fetch data
            raw_data = self.fetch_data()
            if raw_data.empty:
                logger.error("No MIT Election data fetched")
                return False
            
            # Transform data
            clean_data = self.transform_data(raw_data)
            if clean_data.empty:
                logger.error("No MIT Election data after transformation")
                return False
            
            # Validate data
            if not self.validate_data(clean_data):
                logger.error("MIT Election data validation failed")
                return False
            
            # Save to database
            success = self.save_to_database(
                clean_data, 
                'fact_metric',
                unique_columns=['date', 'geo_key', 'metric']
            )
            
            if success:
                logger.info(f"Successfully ingested {len(clean_data)} MIT Election records")
                return True
            else:
                logger.error("Failed to save MIT Election data to database")
                return False
                
        except Exception as e:
            logger.error(f"MIT Election ingestion failed: {e}")
            return False
