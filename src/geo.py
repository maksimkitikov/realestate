"""
Geographic data handling and normalization module
Handles state FIPS codes, abbreviations, and geographic key generation
"""

import pandas as pd
from typing import Dict, List, Optional
import us

# State FIPS to abbreviation mapping
STATE_FIPS_TO_ABBR = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
    '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
    '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY'
}

# State abbreviation to FIPS mapping
STATE_ABBR_TO_FIPS = {v: k for k, v in STATE_FIPS_TO_ABBR.items()}

# State names mapping
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

def get_state_fips(state_abbr: str) -> Optional[str]:
    """Get FIPS code for state abbreviation"""
    return STATE_ABBR_TO_FIPS.get(state_abbr.upper())

def get_state_abbr(state_fips: str) -> Optional[str]:
    """Get state abbreviation for FIPS code"""
    return STATE_FIPS_TO_ABBR.get(state_fips)

def get_state_name(state_abbr: str) -> Optional[str]:
    """Get state name for abbreviation"""
    return STATE_NAMES.get(state_abbr.upper())

def get_all_states() -> List[Dict]:
    """Get list of all US states with FIPS and abbreviation"""
    states = []
    for fips, abbr in STATE_FIPS_TO_ABBR.items():
        state_name = STATE_NAMES.get(abbr, abbr)
        states.append({
            'fips': fips,
            'abbr': abbr,
            'name': state_name,
            'geo_key': f"STATE_{fips}"
        })
    return states

def normalize_geo_key(geo_level: str, state_fips: Optional[str] = None, 
                     county_fips: Optional[str] = None, msa: Optional[str] = None,
                     zip_code: Optional[str] = None) -> str:
    """
    Generate normalized geographic key
    
    Args:
        geo_level: Geographic level (US, STATE, MSA, COUNTY, ZIP)
        state_fips: State FIPS code
        county_fips: County FIPS code
        msa: MSA code
        zip_code: ZIP code
        
    Returns:
        Normalized geographic key
    """
    if geo_level == 'US':
        return 'US_00'
    elif geo_level == 'STATE' and state_fips:
        return f'STATE_{state_fips}'
    elif geo_level == 'COUNTY' and state_fips and county_fips:
        return f'COUNTY_{state_fips}_{county_fips}'
    elif geo_level == 'MSA' and msa:
        return f'MSA_{msa}'
    elif geo_level == 'ZIP' and zip_code:
        return f'ZIP_{zip_code}'
    else:
        raise ValueError(f"Invalid geographic parameters for level {geo_level}")

def create_geo_dimension_data() -> pd.DataFrame:
    """Create dimension data for geographic entities"""
    states = get_all_states()
    
    # Create US record
    us_record = {
        'geo_key': 'US_00',
        'level': 'US',
        'state_fips': None,
        'county_fips': None,
        'msa': None,
        'zip': None,
        'state_abbr': None,
        'name': 'United States'
    }
    
    # Create state records
    state_records = []
    for state in states:
        state_records.append({
            'geo_key': state['geo_key'],
            'level': 'STATE',
            'state_fips': state['fips'],
            'county_fips': None,
            'msa': None,
            'zip': None,
            'state_abbr': state['abbr'],
            'name': state['name']
        })
    
    return pd.DataFrame([us_record] + state_records)

def validate_state_fips(fips: str) -> bool:
    """Validate state FIPS code"""
    return fips in STATE_FIPS_TO_ABBR

def validate_state_abbr(abbr: str) -> bool:
    """Validate state abbreviation"""
    return abbr.upper() in STATE_ABBR_TO_FIPS
