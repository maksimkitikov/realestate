#!/usr/bin/env python3
"""
Production Real Estate Analytics Dashboard
Interactive dashboard for US real estate market analysis
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import logging
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DB_URL = os.getenv("DATABASE_URL", "")
if not DB_URL:
    print("❌ DATABASE_URL not found in environment variables")
    exit(1)

engine = create_engine(DB_URL, pool_pre_ping=True)

def safe_query(query, default_df=None):
    """Safe query execution with error handling"""
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Query error: {e}")
        return default_df if default_df is not None else pd.DataFrame()

def get_fred_data():
    """Get FRED economic data"""
    query = """
    SELECT date, metric, value, unit
    FROM fact_metric 
    WHERE source = 'FRED' 
    AND date >= '2020-01-01'
    ORDER BY date, metric
    """
    return safe_query(query)

def get_latest_metrics():
    """Get latest metric values"""
    query = """
    SELECT metric, value, unit, date
    FROM fact_metric fm1
    WHERE source = 'FRED'
    AND date = (
        SELECT MAX(date) 
        FROM fact_metric fm2 
        WHERE fm2.metric = fm1.metric 
        AND fm2.source = 'FRED'
    )
    ORDER BY metric
    """
    return safe_query(query)

def get_metric_history(metric_name, days=365):
    """Get historical data for specific metric"""
    query = f"""
    SELECT date, value
    FROM fact_metric 
    WHERE metric = '{metric_name}'
    AND date >= CURRENT_DATE - INTERVAL '{days} days'
    ORDER BY date
    """
    return safe_query(query)

# Create Dash app
app = dash.Dash(__name__, title="US Real Estate Analytics Dashboard")

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏠 US Real Estate Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H3("Production-Ready Market Analysis Platform", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30})
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
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Select Metric:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': '30-Year Mortgage Rate', 'value': 'MORTGAGE30US'},
                    {'label': 'Consumer Price Index', 'value': 'CPIAUCSL'},
                    {'label': 'Unemployment Rate', 'value': 'UNRATE'},
                    {'label': '10-Year Treasury', 'value': 'DGS10'},
                    {'label': 'Yield Curve Spread', 'value': 'T10Y2Y'}
                ],
                value='MORTGAGE30US',
                style={'width': '300px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Time Period:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='period-selector',
                options=[
                    {'label': '1 Year', 'value': 365},
                    {'label': '2 Years', 'value': 730},
                    {'label': '5 Years', 'value': 1825},
                    {'label': 'All Data', 'value': 10000}
                ],
                value=365,
                style={'width': '200px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'marginBottom': '30px'}),
    
    # Main Chart
    html.Div([
        dcc.Graph(id='main-chart', style={'height': '500px'})
    ], style={'marginBottom': '30px'}),
    
    # System Status
    html.Div([
        html.H3("📊 System Status", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        html.Div(id='system-status', style={'textAlign': 'center'})
    ]),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P("US Real Estate Analytics System - Production Ready", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
        html.P("Data Sources: FRED, BLS, BEA, Census, HUD, FHFA, Redfin", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
    ], style={'marginTop': '50px'})
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
    [Input('metric-selector', 'value')]
)
def update_metrics(selected_metric):
    """Update key metrics display"""
    try:
        df = get_latest_metrics()
        
        if df.empty:
            return "N/A", "No data", "N/A", "No data", "N/A", "No data", "N/A", "No data"
        
        # Get latest values for each metric
        mortgage_data = df[df['metric'] == 'MORTGAGE30US']
        cpi_data = df[df['metric'] == 'CPIAUCSL']
        unemployment_data = df[df['metric'] == 'UNRATE']
        treasury_data = df[df['metric'] == 'DGS10']
        
        mortgage_value = f"{mortgage_data['value'].iloc[0]:.2f}%" if not mortgage_data.empty else "N/A"
        mortgage_date = mortgage_data['date'].iloc[0].strftime('%Y-%m-%d') if not mortgage_data.empty else "No data"
        
        cpi_value = f"{cpi_data['value'].iloc[0]:.1f}" if not cpi_data.empty else "N/A"
        cpi_date = cpi_data['date'].iloc[0].strftime('%Y-%m-%d') if not cpi_data.empty else "No data"
        
        unemployment_value = f"{unemployment_data['value'].iloc[0]:.1f}%" if not unemployment_data.empty else "N/A"
        unemployment_date = unemployment_data['date'].iloc[0].strftime('%Y-%m-%d') if not unemployment_data.empty else "No data"
        
        treasury_value = f"{treasury_data['value'].iloc[0]:.2f}%" if not treasury_data.empty else "N/A"
        treasury_date = treasury_data['date'].iloc[0].strftime('%Y-%m-%d') if not treasury_data.empty else "No data"
        
        return mortgage_value, mortgage_date, cpi_value, cpi_date, unemployment_value, unemployment_date, treasury_value, treasury_date
        
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")
        return "Error", "Error", "Error", "Error", "Error", "Error", "Error", "Error"

@app.callback(
    Output('main-chart', 'figure'),
    [Input('metric-selector', 'value'),
     Input('period-selector', 'value')]
)
def update_chart(selected_metric, period_days):
    """Update main chart"""
    try:
        df = get_metric_history(selected_metric, period_days)
        
        if df.empty:
            # Return empty chart
            fig = go.Figure()
            fig.update_layout(
                title=f"No data available for {selected_metric}",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white"
            )
            return fig
        
        # Create chart
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name=selected_metric,
            line=dict(color='#3498db', width=2)
        ))
        
        # Update layout
        metric_names = {
            'MORTGAGE30US': '30-Year Mortgage Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate',
            'DGS10': '10-Year Treasury Rate',
            'T10Y2Y': 'Yield Curve Spread'
        }
        
        fig.update_layout(
            title=f"{metric_names.get(selected_metric, selected_metric)} - Historical Data",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title="Error loading data",
            template="plotly_white"
        )
        return fig

@app.callback(
    Output('system-status', 'children'),
    [Input('metric-selector', 'value')]
)
def update_system_status(selected_metric):
    """Update system status display"""
    try:
        # Get database statistics
        total_records_query = "SELECT COUNT(*) as count FROM fact_metric"
        total_records = safe_query(total_records_query)
        total_count = total_records['count'].iloc[0] if not total_records.empty else 0
        
        latest_date_query = "SELECT MAX(date) as latest_date FROM fact_metric"
        latest_date = safe_query(latest_date_query)
        latest = latest_date['latest_date'].iloc[0] if not latest_date.empty else "Unknown"
        
        metrics_count_query = "SELECT COUNT(DISTINCT metric) as count FROM fact_metric"
        metrics_count = safe_query(metrics_count_query)
        metrics = metrics_count['count'].iloc[0] if not metrics_count.empty else 0
        
        status_html = [
            html.Div([
                html.Strong("Total Records: "), f"{total_count:,}",
                html.Br(),
                html.Strong("Latest Data: "), str(latest)[:10] if latest != "Unknown" else "Unknown",
                html.Br(),
                html.Strong("Metrics Available: "), f"{metrics}",
                html.Br(),
                html.Strong("Database: "), "✅ Connected",
                html.Br(),
                html.Strong("Last Update: "), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ], style={'textAlign': 'left', 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
        ]
        
        return status_html
        
    except Exception as e:
        logger.error(f"Error updating system status: {e}")
        return html.Div("Error loading system status", style={'color': 'red'})

if __name__ == '__main__':
    print("🚀 Starting US Real Estate Analytics Dashboard...")
    print("🌐 Open http://localhost:8050 in your browser")
    print("📊 Production-ready analytics platform")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
