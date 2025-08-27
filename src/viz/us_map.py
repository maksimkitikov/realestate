"""
Interactive US States Map Module
Creates choropleth maps for US states with various metrics
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class USMapVisualizer:
    """Interactive US states map visualizer"""
    
    def __init__(self):
        self.states_data = self._load_states_data()
    
    def _load_states_data(self) -> pd.DataFrame:
        """Load US states geographic data"""
        # Create states data with FIPS codes
        states = [
            {'state': 'Alabama', 'abbr': 'AL', 'fips': '01'},
            {'state': 'Alaska', 'abbr': 'AK', 'fips': '02'},
            {'state': 'Arizona', 'abbr': 'AZ', 'fips': '04'},
            {'state': 'Arkansas', 'abbr': 'AR', 'fips': '05'},
            {'state': 'California', 'abbr': 'CA', 'fips': '06'},
            {'state': 'Colorado', 'abbr': 'CO', 'fips': '08'},
            {'state': 'Connecticut', 'abbr': 'CT', 'fips': '09'},
            {'state': 'Delaware', 'abbr': 'DE', 'fips': '10'},
            {'state': 'District of Columbia', 'abbr': 'DC', 'fips': '11'},
            {'state': 'Florida', 'abbr': 'FL', 'fips': '12'},
            {'state': 'Georgia', 'abbr': 'GA', 'fips': '13'},
            {'state': 'Hawaii', 'abbr': 'HI', 'fips': '15'},
            {'state': 'Idaho', 'abbr': 'ID', 'fips': '16'},
            {'state': 'Illinois', 'abbr': 'IL', 'fips': '17'},
            {'state': 'Indiana', 'abbr': 'IN', 'fips': '18'},
            {'state': 'Iowa', 'abbr': 'IA', 'fips': '19'},
            {'state': 'Kansas', 'abbr': 'KS', 'fips': '20'},
            {'state': 'Kentucky', 'abbr': 'KY', 'fips': '21'},
            {'state': 'Louisiana', 'abbr': 'LA', 'fips': '22'},
            {'state': 'Maine', 'abbr': 'ME', 'fips': '23'},
            {'state': 'Maryland', 'abbr': 'MD', 'fips': '24'},
            {'state': 'Massachusetts', 'abbr': 'MA', 'fips': '25'},
            {'state': 'Michigan', 'abbr': 'MI', 'fips': '26'},
            {'state': 'Minnesota', 'abbr': 'MN', 'fips': '27'},
            {'state': 'Mississippi', 'abbr': 'MS', 'fips': '28'},
            {'state': 'Missouri', 'abbr': 'MO', 'fips': '29'},
            {'state': 'Montana', 'abbr': 'MT', 'fips': '30'},
            {'state': 'Nebraska', 'abbr': 'NE', 'fips': '31'},
            {'state': 'Nevada', 'abbr': 'NV', 'fips': '32'},
            {'state': 'New Hampshire', 'abbr': 'NH', 'fips': '33'},
            {'state': 'New Jersey', 'abbr': 'NJ', 'fips': '34'},
            {'state': 'New Mexico', 'abbr': 'NM', 'fips': '35'},
            {'state': 'New York', 'abbr': 'NY', 'fips': '36'},
            {'state': 'North Carolina', 'abbr': 'NC', 'fips': '37'},
            {'state': 'North Dakota', 'abbr': 'ND', 'fips': '38'},
            {'state': 'Ohio', 'abbr': 'OH', 'fips': '39'},
            {'state': 'Oklahoma', 'abbr': 'OK', 'fips': '40'},
            {'state': 'Oregon', 'abbr': 'OR', 'fips': '41'},
            {'state': 'Pennsylvania', 'abbr': 'PA', 'fips': '42'},
            {'state': 'Rhode Island', 'abbr': 'RI', 'fips': '44'},
            {'state': 'South Carolina', 'abbr': 'SC', 'fips': '45'},
            {'state': 'South Dakota', 'abbr': 'SD', 'fips': '46'},
            {'state': 'Tennessee', 'abbr': 'TN', 'fips': '47'},
            {'state': 'Texas', 'abbr': 'TX', 'fips': '48'},
            {'state': 'Utah', 'abbr': 'UT', 'fips': '49'},
            {'state': 'Vermont', 'abbr': 'VT', 'fips': '50'},
            {'state': 'Virginia', 'abbr': 'VA', 'fips': '51'},
            {'state': 'Washington', 'abbr': 'WA', 'fips': '53'},
            {'state': 'West Virginia', 'abbr': 'WV', 'fips': '54'},
            {'state': 'Wisconsin', 'abbr': 'WI', 'fips': '55'},
            {'state': 'Wyoming', 'abbr': 'WY', 'fips': '56'}
        ]
        
        return pd.DataFrame(states)
    
    def create_choropleth_map(self, data: pd.DataFrame, metric_column: str, 
                             title: str = "US States Map", 
                             color_scale: str = "Viridis") -> go.Figure:
        """
        Create interactive choropleth map for US states
        
        Args:
            data: DataFrame with state data and metric values
            metric_column: Column name containing metric values
            title: Chart title
            color_scale: Color scale for the map
            
        Returns:
            Plotly figure object
        """
        try:
            # Merge with states data
            if 'state' not in data.columns and 'abbr' in data.columns:
                merged_data = data.merge(self.states_data, on='abbr', how='left')
            elif 'state' in data.columns:
                merged_data = data.merge(self.states_data, on='state', how='left')
            else:
                merged_data = data
            
            # Create choropleth map
            fig = px.choropleth(
                merged_data,
                locations='fips',
                locationmode='fips',
                color=metric_column,
                hover_name='state',
                hover_data=[metric_column, 'abbr'],
                color_continuous_scale=color_scale,
                scope='usa',
                title=title,
                labels={metric_column: title}
            )
            
            # Update layout
            fig.update_layout(
                geo=dict(
                    showlakes=True,
                    lakecolor='rgb(255, 255, 255)',
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    showocean=True,
                    oceancolor='rgb(204, 229, 255)',
                    showcoastlines=True,
                    coastlinecolor='rgb(80, 80, 80)',
                    showframe=False,
                    framewidth=0,
                    projection=dict(type='albers usa')
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating choropleth map: {e}")
            # Return empty map
            fig = go.Figure()
            fig.update_layout(
                title="Error creating map",
                geo=dict(scope='usa', projection=dict(type='albers usa'))
            )
            return fig
    
    def create_sample_map(self) -> go.Figure:
        """Create sample map with dummy data"""
        # Create sample data
        sample_data = self.states_data.copy()
        sample_data['sample_metric'] = sample_data.index * 2.5  # Dummy metric
        
        return self.create_choropleth_map(
            sample_data, 
            'sample_metric', 
            "Sample US States Map"
        )
