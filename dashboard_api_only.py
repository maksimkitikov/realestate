#!/usr/bin/env python3
"""
API-Only Real Estate Dashboard for Render
Developed by Maksim Kitikov - Upside Analytics
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import logging
from datetime import datetime, timedelta
import requests
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
BEA_API_KEY = os.getenv("BEA_API_KEY", "")
BLS_API_KEY = os.getenv("BLS_API_KEY", "")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

def get_fred_data(series_id):
    """Get data from FRED API"""
    if not FRED_API_KEY:
        return None
    
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': 100
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if 'observations' in data:
            observations = data['observations']
            if observations:
                latest = observations[0]
                return {
                    'value': float(latest['value']) if latest['value'] != '.' else None,
                    'date': latest['date']
                }
    except Exception as e:
        logger.error(f"FRED API error for {series_id}: {e}")
    
    return None

def get_latest_metrics():
    """Get latest economic metrics from APIs"""
    metrics = {}
    
    # FRED Series IDs
    fred_series = {
        'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average',
        'CPIAUCSL': 'Consumer Price Index',
        'UNRATE': 'Unemployment Rate',
        'DGS10': '10-Year Treasury Constant Maturity Rate'
    }
    
    for series_id, description in fred_series.items():
        data = get_fred_data(series_id)
        if data and data['value'] is not None:
            metrics[series_id] = {
                'value': data['value'],
                'date': data['date'],
                'description': description
            }
    
    return metrics

def get_metric_history(metric_name, days=365):
    """Get historical data for specific metric"""
    if not FRED_API_KEY:
        return pd.DataFrame()
    
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': metric_name,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': days
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if 'observations' in data:
            observations = data['observations']
            df_data = []
            for obs in observations:
                if obs['value'] != '.':
                    df_data.append({
                        'date': obs['date'],
                        'value': float(obs['value'])
                    })
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')
    
    except Exception as e:
        logger.error(f"Error getting history for {metric_name}: {e}")
    
    return pd.DataFrame()

def get_states_data():
    """Get sample state data (since we don't have full API access)"""
    states = [
        {'state': 'CA', 'name': 'California', 'home_value': 750000, 'unemployment': 4.2},
        {'state': 'TX', 'name': 'Texas', 'home_value': 350000, 'unemployment': 3.8},
        {'state': 'FL', 'name': 'Florida', 'home_value': 450000, 'unemployment': 3.1},
        {'state': 'NY', 'name': 'New York', 'home_value': 650000, 'unemployment': 4.5},
        {'state': 'IL', 'name': 'Illinois', 'home_value': 280000, 'unemployment': 4.8},
        {'state': 'PA', 'name': 'Pennsylvania', 'home_value': 320000, 'unemployment': 4.1},
        {'state': 'OH', 'name': 'Ohio', 'home_value': 220000, 'unemployment': 3.9},
        {'state': 'GA', 'name': 'Georgia', 'home_value': 380000, 'unemployment': 3.2},
        {'state': 'NC', 'name': 'North Carolina', 'home_value': 340000, 'unemployment': 3.5},
        {'state': 'MI', 'name': 'Michigan', 'home_value': 240000, 'unemployment': 4.3}
    ]
    
    return pd.DataFrame(states)

# Create Dash app
app = dash.Dash(__name__, title="API-Only Real Estate Dashboard")

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏠 API-Only Real Estate Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H3("Real-time Data from Federal APIs", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 10}),
        html.H4("Developed by Maksim Kitikov - Upside Analytics", 
                style={'textAlign': 'center', 'color': '#000000', 'marginBottom': 10, 'fontWeight': 'bold'}),
        html.P("📊 Live data from FRED API (Federal Reserve Economic Data)", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'marginBottom': 30})
    ]),
    
    # Key Metrics Row
    html.Div([
        html.Div([
            html.H4("📈 Mortgage Rate", id='mortgage-rate-title'),
            html.H2(id='mortgage-rate-value', style={'color': '#e74c3c'}),
            html.P(id='mortgage-rate-date', style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
        
        html.Div([
            html.H4("💰 CPI Index", id='cpi-title'),
            html.H2(id='cpi-value', style={'color': '#3498db'}),
            html.P(id='cpi-date', style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
        
        html.Div([
            html.H4("📊 Unemployment", id='unemployment-title'),
            html.H2(id='unemployment-value', style={'color': '#f39c12'}),
            html.P(id='unemployment-date', style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
        
        html.Div([
            html.H4("📈 Treasury 10Y", id='treasury-title'),
            html.H2(id='treasury-value', style={'color': '#27ae60'}),
            html.P(id='treasury-date', style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
    # Map Controls
    html.Div([
        html.H3("🗺️ US States Sample Data", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.Label("Select Metric for Map:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='map-metric-selector',
                    options=[
                        {'label': 'Home Value ($)', 'value': 'home_value'},
                        {'label': 'Unemployment Rate (%)', 'value': 'unemployment'}
                    ],
                    value='home_value',
                    style={'width': '300px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Color Scale:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='color-scale-selector',
                    options=[
                        {'label': 'Viridis', 'value': 'Viridis'},
                        {'label': 'Plasma', 'value': 'Plasma'},
                        {'label': 'Inferno', 'value': 'Inferno'},
                        {'label': 'RdBu', 'value': 'RdBu'}
                    ],
                    value='Viridis',
                    style={'width': '200px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'marginBottom': '20px'}),
        
        # Map
        dcc.Graph(id='us-map')
    ]),
    
    # Historical Chart
    html.Div([
        html.H3("📈 Historical Data", style={'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Dropdown(
            id='metric-selector',
            options=[
                {'label': '30-Year Mortgage Rate', 'value': 'MORTGAGE30US'},
                {'label': 'Consumer Price Index', 'value': 'CPIAUCSL'},
                {'label': 'Unemployment Rate', 'value': 'UNRATE'},
                {'label': '10-Year Treasury Rate', 'value': 'DGS10'}
            ],
            value='MORTGAGE30US',
            style={'width': '400px', 'margin': '0 auto 20px auto'}
        ),
        dcc.Graph(id='historical-chart')
    ], style={'marginTop': '30px'}),
    
    # Status
    html.Div([
        html.H4("System Status", style={'marginBottom': '20px'}),
        html.Div(id='system-status')
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
])

@app.callback(
    [Output('mortgage-rate-value', 'children'),
     Output('mortgage-rate-date', 'children'),
     Output('cpi-value', 'children'),
     Output('cpi-date', 'children'),
     Output('unemployment-value', 'children'),
     Output('unemployment-date', 'children'),
     Output('treasury-value', 'children'),
     Output('treasury-date', 'children')],
    [Input('mortgage-rate-value', 'id')]
)
def update_metrics(_):
    """Update key metrics from FRED API"""
    metrics = get_latest_metrics()
    
    mortgage_data = metrics.get('MORTGAGE30US', {})
    cpi_data = metrics.get('CPIAUCSL', {})
    unemployment_data = metrics.get('UNRATE', {})
    treasury_data = metrics.get('DGS10', {})
    
    return (
        f"{mortgage_data.get('value', 'N/A'):.2f}%" if mortgage_data.get('value') else "N/A",
        mortgage_data.get('date', 'No data'),
        f"{cpi_data.get('value', 'N/A'):.1f}" if cpi_data.get('value') else "N/A",
        cpi_data.get('date', 'No data'),
        f"{unemployment_data.get('value', 'N/A'):.1f}%" if unemployment_data.get('value') else "N/A",
        unemployment_data.get('date', 'No data'),
        f"{treasury_data.get('value', 'N/A'):.2f}%" if treasury_data.get('value') else "N/A",
        treasury_data.get('date', 'No data')
    )

@app.callback(
    Output('us-map', 'figure'),
    [Input('map-metric-selector', 'value'),
     Input('color-scale-selector', 'value')]
)
def update_map(selected_metric, color_scale):
    """Update US states map"""
    df = get_states_data()
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            template="plotly_white"
        )
        return fig
    
    fig = px.choropleth(
        df,
        locations='state',
        locationmode='USA-states',
        color=selected_metric,
        scope='usa',
        color_continuous_scale=color_scale,
        title=f"US States - {selected_metric.replace('_', ' ').title()}",
        labels={selected_metric: selected_metric.replace('_', ' ').title()}
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    
    return fig

@app.callback(
    Output('historical-chart', 'figure'),
    [Input('metric-selector', 'value')]
)
def update_historical_chart(selected_metric):
    """Update historical data chart"""
    df = get_metric_history(selected_metric, 365)
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {selected_metric}",
            template="plotly_white"
        )
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines',
        name=selected_metric,
        line=dict(color='#3498db', width=2)
    ))
    
    metric_names = {
        'MORTGAGE30US': '30-Year Mortgage Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'UNRATE': 'Unemployment Rate',
        'DGS10': '10-Year Treasury Rate'
    }
    
    fig.update_layout(
        title=f"{metric_names.get(selected_metric, selected_metric)} - Historical Data",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        height=400
    )
    
    return fig

@app.callback(
    Output('system-status', 'children'),
    [Input('system-status', 'id')]
)
def update_system_status(_):
    """Update system status"""
    status_html = [
        html.Div([
            html.Strong("Status: "), "✅ Running",
            html.Br(),
            html.Strong("Data Source: "), "FRED API (Federal Reserve)",
            html.Br(),
            html.Strong("API Keys: "), f"{'✅ Available' if FRED_API_KEY else '❌ Missing'}",
            html.Br(),
            html.Strong("Developer: "), "Maksim Kitikov - Upside Analytics",
            html.Br(),
            html.Strong("Last Update: "), datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            html.Br(),
            html.Strong("Mode: "), "API-Only (No Database Required)",
        ], style={'textAlign': 'left'})
    ]
    
    return status_html

# Export server for Gunicorn
server = app.server

if __name__ == '__main__':
    print("🚀 Starting API-Only Real Estate Analytics Dashboard...")
    print("🌐 Developed by Maksim Kitikov - Upside Analytics")
    print("📊 Using FRED API for real-time data")
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)
