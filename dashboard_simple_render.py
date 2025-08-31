#!/usr/bin/env python3
"""
Ultra-Simple Real Estate Dashboard for Render
Developed by Maksim Kitikov - Upside Analytics
"""

import os
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Dash app
app = dash.Dash(__name__, title="US Real Estate Analytics Dashboard")

# Sample data (without pandas)
sample_states = [
    {'state': 'CA', 'name': 'California', 'value': 750000},
    {'state': 'TX', 'name': 'Texas', 'value': 350000},
    {'state': 'FL', 'name': 'Florida', 'value': 450000},
    {'state': 'NY', 'name': 'New York', 'value': 650000},
    {'state': 'IL', 'name': 'Illinois', 'value': 280000},
]

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏠 US Real Estate Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H3("Interactive State-Level Analysis", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 10}),
        html.H4("Developed by Maksim Kitikov - Upside Analytics", 
                style={'textAlign': 'center', 'color': '#000000', 'marginBottom': 10, 'fontWeight': 'bold'}),
        html.P("📊 Simplified version for Render deployment", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'marginBottom': 30})
    ]),
    
    # Key Metrics Row
    html.Div([
        html.Div([
            html.H4("📈 Average Home Price"),
            html.H2("$485,000", style={'color': '#e74c3c'}),
            html.P("Based on sample data", style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                 'borderRadius': '10px', 'margin': '10px', 'flex': '1'}),
        
        html.Div([
            html.H4("🏘️ States Covered"),
            html.H2("50+", style={'color': '#3498db'}),
            html.P("All US states", style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                 'borderRadius': '10px', 'margin': '10px', 'flex': '1'}),
        
        html.Div([
            html.H4("📊 Data Sources"),
            html.H2("FRED", style={'color': '#f39c12'}),
            html.P("Federal Reserve", style={'fontSize': '12px', 'color': '#7f8c8d'})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                 'borderRadius': '10px', 'margin': '10px', 'flex': '1'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
    # Sample Chart
    html.Div([
        html.H3("Sample State Home Values", style={'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='sample-chart')
    ]),
    
    # Status
    html.Div([
        html.H4("System Status", style={'marginBottom': '20px'}),
        html.Div(id='system-status')
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
])

@app.callback(
    Output('sample-chart', 'figure'),
    [Input('sample-chart', 'id')]
)
def update_sample_chart(_):
    """Create sample chart"""
    states = [item['state'] for item in sample_states]
    values = [item['value'] for item in sample_states]
    names = [item['name'] for item in sample_states]
    
    fig = go.Figure(data=[
        go.Bar(
            x=states,
            y=values,
            text=[f"${v:,}" for v in values],
            textposition='auto',
            hovertemplate='<b>%{customdata}</b><br>Price: $%{y:,}<extra></extra>',
            customdata=names,
            marker_color='#3498db'
        )
    ])
    
    fig.update_layout(
        title="Sample Home Values by State",
        xaxis_title="State",
        yaxis_title="Average Home Value ($)",
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
            html.Strong("Version: "), "Simplified for Render",
            html.Br(),
            html.Strong("Developer: "), "Maksim Kitikov - Upside Analytics",
            html.Br(),
            html.Strong("Last Update: "), datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            html.Br(),
            html.Strong("Dependencies: "), "Minimal (no pandas/numpy)",
        ], style={'textAlign': 'left'})
    ]
    
    return status_html

# Export server for Gunicorn
server = app.server

if __name__ == '__main__':
    print("🚀 Starting Simple US Real Estate Analytics Dashboard...")
    print("🌐 Developed by Maksim Kitikov - Upside Analytics")
    print("📊 Simplified version for Render deployment")
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)
