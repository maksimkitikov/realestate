#!/usr/bin/env python3
"""
Advanced US Real Estate Analytics Dashboard with Regression Analysis

Developed by Maksim Kitikov - Upside Analytics
Interactive dashboard for US real estate market analysis with machine learning models
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from dotenv import load_dotenv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import logging
from datetime import datetime, timedelta
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection (optional)
DB_URL = os.getenv("DATABASE_URL", "")
engine = None

if DB_URL:
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        print("✅ Database connection established")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        engine = None
else:
    print("⚠️ DATABASE_URL not found - running in demo mode")

def safe_query(query, default_df=None):
    """Safe query execution with error handling"""
    if engine is None:
        logger.warning("No database connection - returning empty DataFrame")
        return default_df if default_df is not None else pd.DataFrame()
    
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Query error: {e}")
        return default_df if default_df is not None else pd.DataFrame()

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

def get_states_data(time_period='latest'):
    """Вернуть реальное сечение по штатам на выбранный период (без моков).

    Собираем метрики напрямую из fact_metric, берём последнее значение по каждому штату,
    дата которого не позже граничной даты (cutoff). Для производительности используем DISTINCT ON.
    """
    try:
        # Определяем граничную дату
        interval_map = {
            'latest': None,
            '1m': "1 month",
            '3m': "3 months",
            '6m': "6 months",
            '1y': "1 year",
            '2y': "2 years",
        }
        interval = interval_map.get(time_period, None)

        def latest_per_state(metric_key: str, cutoff_interval: str):
            base = f"""
                SELECT dg.geo_key, dg.state_abbr AS state, dg.name AS state_name, fm.value, fm.date
                FROM dim_geo dg
                LEFT JOIN LATERAL (
                    SELECT value, date
                    FROM fact_metric f
                    WHERE f.geo_level='STATE' AND f.geo_key = dg.geo_key AND f.metric = '{metric_key}'
                    {{cutoff_clause}}
                    ORDER BY date DESC
                    LIMIT 1
                ) fm ON TRUE
                WHERE dg.level='STATE'
            """
            cutoff_clause = ""
            if cutoff_interval:
                cutoff_clause = f"AND date <= CURRENT_DATE - INTERVAL '{cutoff_interval}'"
            q = base.replace("{cutoff_clause}", cutoff_clause)
            return safe_query(q)

        # Базовая таблица штатов
        states = safe_query("SELECT geo_key, state_abbr AS state, name AS state_name FROM dim_geo WHERE level='STATE'")
        if states.empty:
            logger.warning("No states in dim_geo")
            return pd.DataFrame()

        # Реальные метрики
        home_value_df = latest_per_state('MEDIAN_SALE_PRICE', interval)
        if home_value_df['value'].isna().all():
            # fallback на ACS home value если отсутствует Redfin
            home_value_df = latest_per_state('MEDIAN_HOME_VALUE', interval)

        hpi_yoy_df = latest_per_state('HPI_GROWTH_YOY', interval)
        unemp_df = latest_per_state('UNEMPLOYMENT_RATE', interval)
        dom_df = latest_per_state('DAYS_ON_MARKET', interval)
        months_supply_df = latest_per_state('MONTHS_OF_SUPPLY', interval)
        homeowners_df = latest_per_state('HOMEOWNERSHIP_RATE', interval)
        edu_df = latest_per_state('EDUCATION_RATE', interval)
        divorce_df = latest_per_state('DIVORCE_RATE', interval)
        disaster_df = latest_per_state('DISASTER_RATE_3YR_AVG', interval)
        income_df = latest_per_state('MEDIAN_HOUSEHOLD_INCOME', interval)
        pop_df = latest_per_state('TOTAL_POPULATION', interval)
        gdp_df = latest_per_state('GDP_TOTAL', interval)

        # Сборка датафрейма
        df = states.copy()
        def join_metric(src: pd.DataFrame, col: str):
            # Всегда создаём столбец; если данных нет, он будет заполнен NaN
            if src is None or src.empty:
                df[col] = np.nan
                return
            df[col] = df.merge(src[['geo_key','value']], on='geo_key', how='left')['value']

        join_metric(home_value_df, 'home_value')
        join_metric(hpi_yoy_df, 'price_growth_yoy')
        join_metric(unemp_df, 'unemployment_rate')
        join_metric(dom_df, 'median_days_on_market')
        if not months_supply_df.empty and months_supply_df['value'].notna().any():
            join_metric(months_supply_df, 'months_of_supply')
        join_metric(homeowners_df, 'homeownership_rate')
        join_metric(edu_df, 'education_rate')
        join_metric(divorce_df, 'divorce_rate')
        join_metric(disaster_df, 'disaster_rate_3yr')
        join_metric(pop_df, 'population')
        join_metric(gdp_df, 'gdp_total')
        join_metric(income_df, 'median_household_income')

        # GDP per capita и value_to_income_ratio
        if 'gdp_total' in df.columns and 'population' in df.columns:
            df['gdp_per_capita'] = np.where((df['gdp_total'].notna()) & (df['population'].notna()) & (df['population']>0),
                                            df['gdp_total'] / df['population'], np.nan)
        if 'median_household_income' in df.columns and 'home_value' in df.columns:
            df['value_to_income_ratio'] = np.where((df['median_household_income'].notna()) & (df['median_household_income']>0) & (df['home_value'].notna()),
                                                   df['home_value'] / df['median_household_income'], np.nan)

        # Income growth YoY от median_household_income если доступно, иначе от PERSONAL_INCOME
        # median income YoY
        # корректный cutoff для предыдущего года:
        # if latest: cutoff_prev = CURRENT_DATE - INTERVAL '1 year'
        # if interval=X: cutoff_prev = CURRENT_DATE - INTERVAL '1 year' - INTERVAL 'X'
        def latest_per_state_with_expr(metric_key: str, cutoff_expr_sql: str | None):
            base = f"""
                SELECT dg.geo_key, dg.state_abbr AS state, dg.name AS state_name, fm.value, fm.date
                FROM dim_geo dg
                LEFT JOIN LATERAL (
                    SELECT value, date
                    FROM fact_metric f
                    WHERE f.geo_level='STATE' AND f.geo_key = dg.geo_key AND f.metric = '{metric_key}'
                    {{cutoff_clause}}
                    ORDER BY date DESC
                    LIMIT 1
                ) fm ON TRUE
                WHERE dg.level='STATE'
            """
            cutoff_clause = ""
            if cutoff_expr_sql:
                cutoff_clause = f"AND date <= {cutoff_expr_sql}"
            q = base.replace("{cutoff_clause}", cutoff_clause)
            return safe_query(q)

        if interval is None:
            cutoff_prev_expr = "CURRENT_DATE - INTERVAL '1 year'"
        else:
            cutoff_prev_expr = f"CURRENT_DATE - INTERVAL '1 year' - INTERVAL '{interval}'"
        income_prev = latest_per_state_with_expr('MEDIAN_HOUSEHOLD_INCOME', cutoff_prev_expr)
        if not income_df.empty and not income_prev.empty:
            tmp = df[['geo_key']].merge(income_df[['geo_key','value']].rename(columns={'value':'cur_inc'}), on='geo_key', how='left')
            tmp = tmp.merge(income_prev[['geo_key','value']].rename(columns={'value':'prev_inc'}), on='geo_key', how='left')
            df['income_growth'] = np.where((tmp['cur_inc'].notna()) & (tmp['prev_inc'].notna()) & (tmp['prev_inc']!=0),
                                           (tmp['cur_inc']/tmp['prev_inc'] - 1)*100, np.nan)

        logger.info(f"Retrieved real state data for {len(df)} states from database")
        return df

    except Exception as e:
        logger.error(f"Error fetching real state data: {e}")
        return pd.DataFrame()

def get_fallback_states_data():
    """Fallback sample data if database is not available"""
    logger.warning("Using fallback sample state data")
    
    states_data = [
        {'state': 'AL', 'name': 'Alabama'},
        {'state': 'AK', 'name': 'Alaska'},
        {'state': 'AZ', 'name': 'Arizona'},
        {'state': 'AR', 'name': 'Arkansas'},
        {'state': 'CA', 'name': 'California'},
        {'state': 'CO', 'name': 'Colorado'},
        {'state': 'CT', 'name': 'Connecticut'},
        {'state': 'DE', 'name': 'Delaware'},
        {'state': 'FL', 'name': 'Florida'},
        {'state': 'GA', 'name': 'Georgia'},
        {'state': 'HI', 'name': 'Hawaii'},
        {'state': 'ID', 'name': 'Idaho'},
        {'state': 'IL', 'name': 'Illinois'},
        {'state': 'IN', 'name': 'Indiana'},
        {'state': 'IA', 'name': 'Iowa'},
        {'state': 'KS', 'name': 'Kansas'},
        {'state': 'KY', 'name': 'Kentucky'},
        {'state': 'LA', 'name': 'Louisiana'},
        {'state': 'ME', 'name': 'Maine'},
        {'state': 'MD', 'name': 'Maryland'},
        {'state': 'MA', 'name': 'Massachusetts'},
        {'state': 'MI', 'name': 'Michigan'},
        {'state': 'MN', 'name': 'Minnesota'},
        {'state': 'MS', 'name': 'Mississippi'},
        {'state': 'MO', 'name': 'Missouri'},
        {'state': 'MT', 'name': 'Montana'},
        {'state': 'NE', 'name': 'Nebraska'},
        {'state': 'NV', 'name': 'Nevada'},
        {'state': 'NH', 'name': 'New Hampshire'},
        {'state': 'NJ', 'name': 'New Jersey'},
        {'state': 'NM', 'name': 'New Mexico'},
        {'state': 'NY', 'name': 'New York'},
        {'state': 'NC', 'name': 'North Carolina'},
        {'state': 'ND', 'name': 'North Dakota'},
        {'state': 'OH', 'name': 'Ohio'},
        {'state': 'OK', 'name': 'Oklahoma'},
        {'state': 'OR', 'name': 'Oregon'},
        {'state': 'PA', 'name': 'Pennsylvania'},
        {'state': 'RI', 'name': 'Rhode Island'},
        {'state': 'SC', 'name': 'South Carolina'},
        {'state': 'SD', 'name': 'South Dakota'},
        {'state': 'TN', 'name': 'Tennessee'},
        {'state': 'TX', 'name': 'Texas'},
        {'state': 'UT', 'name': 'Utah'},
        {'state': 'VT', 'name': 'Vermont'},
        {'state': 'VA', 'name': 'Virginia'},
        {'state': 'WA', 'name': 'Washington'},
        {'state': 'WV', 'name': 'West Virginia'},
        {'state': 'WI', 'name': 'Wisconsin'},
        {'state': 'WY', 'name': 'Wyoming'}
    ]
    
    # Set fixed seed for consistent data
    random.seed(42)
    
    data = []
    for i, state_info in enumerate(states_data):
        data.append({
            'state': state_info['state'],
            'state_name': state_info['name'],
            'home_value': 200000 + (i * 15000) + random.uniform(-50000, 50000),
            'price_growth_yoy': -5.0 + (i * 0.5) + random.uniform(-5.0, 5.0),
            'unemployment_rate': 3.0 + (i * 0.1) + random.uniform(-1.0, 1.0),
            'income_growth': -1.0 + (i * 0.1) + random.uniform(-1.0, 1.0),
            'population': 1000000 + (i * 500000) + random.uniform(-200000, 200000),
            'gdp_per_capita': 50000 + (i * 2000) + random.uniform(-5000, 5000),
            'median_days_on_market': 30 + random.uniform(-15, 30),
            'months_of_supply': 2.0 + random.uniform(-1.0, 2.0),
            'homeownership_rate': 60.0 + random.uniform(-10.0, 15.0),
            'education_rate': 25.0 + random.uniform(-10.0, 20.0),
            'divorce_rate': 8.0 + random.uniform(-3.0, 7.0),
            'disaster_rate_3yr': random.uniform(0.5, 3.0),
            'political_competitiveness': random.uniform(5.0, 30.0),
            'risk_score': random.uniform(10.0, 50.0),
            'value_to_income_ratio': 3.0 + random.uniform(-1.0, 3.0),
            'market_temperature': random.choice(['Hot', 'Warm', 'Balanced', 'Cool'])
        })
    
    return pd.DataFrame(data)



def calculate_real_estate_risk_score(states_data):
    """Calculate comprehensive real estate risk score for each state"""
    risk_data = states_data.copy()
    
    # Risk factors for real estate companies (based on available real data)
    risk_factors = {}
    
    for idx, row in risk_data.iterrows():
        risk_score = 0
        risk_details = {}
        
        # 1. Economic Risk (30% weight)
        # High unemployment = higher risk
        unemployment_risk = min(row.get('unemployment_rate', 0) / 10.0, 1.0) * 30
        risk_details['unemployment_risk'] = unemployment_risk
        
        # Low GDP per capita = higher risk
        gdp_risk = max(0, (50000 - row.get('gdp_per_capita', 50000)) / 50000) * 15
        risk_details['gdp_risk'] = gdp_risk
        
        # 2. Market Risk (25% weight)
        # Low median income = higher risk (using value_to_income_ratio as proxy)
        income_risk = max(0, (10 - row.get('value_to_income_ratio', 10)) / 10) * 25
        risk_details['income_risk'] = income_risk
        
        # 3. Population Risk (20% weight)
        # Small population = higher risk (less market)
        population_risk = max(0, (5000000 - row.get('population', 5000000)) / 5000000) * 20
        risk_details['population_risk'] = population_risk
        
        # 4. Education Risk (15% weight)
        # Low education rate = higher risk
        education_risk = max(0, (30 - row.get('education_rate', 30)) / 30) * 15
        risk_details['education_risk'] = education_risk
        
        # 5. Housing Market Risk (10% weight)
        # Low homeownership rate = higher risk
        homeownership_risk = max(0, (65 - row.get('homeownership_rate', 65)) / 65) * 10
        risk_details['homeownership_risk'] = homeownership_risk
        
        # Calculate total risk score (0-100, higher = more risky)
        total_risk = unemployment_risk + gdp_risk + income_risk + population_risk + education_risk + homeownership_risk
        risk_score = min(total_risk, 100)
        
        # Risk categories
        if risk_score < 20:
            risk_category = "Low Risk"
            risk_color = "#27ae60"
        elif risk_score < 40:
            risk_category = "Moderate Risk"
            risk_color = "#f39c12"
        elif risk_score < 60:
            risk_category = "High Risk"
            risk_color = "#e67e22"
        else:
            risk_category = "Very High Risk"
            risk_color = "#e74c3c"
        
        risk_factors[idx] = {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'risk_color': risk_color,
            'risk_details': risk_details
        }
    
    return risk_factors

def perform_regression_analysis():
    """Perform comprehensive regression analysis on state data"""
    states_data = get_states_data('latest')  # Use latest data for regression
    
    # Prepare data for regression - use available columns
    available_features = ['unemployment_rate', 'population', 'gdp_per_capita', 'value_to_income_ratio']
    X = states_data[available_features].fillna(0)
    y = states_data['home_value']
    
    # Multiple regression models for comparison
    models = {}
    results = {}
    
    # 1. Linear Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr_model = LinearRegression()
    lr_model.fit(X_scaled, y)
    lr_pred = lr_model.predict(X_scaled)
    
    models['Linear Regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'predictions': lr_pred,
        'r2': r2_score(y, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y, lr_pred)),
        'mae': mean_absolute_error(y, lr_pred)
    }
    
    # 2. Ridge Regression (L2 regularization)
    from sklearn.linear_model import Ridge
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scaled, y)
    ridge_pred = ridge_model.predict(X_scaled)
    
    models['Ridge Regression'] = {
        'model': ridge_model,
        'scaler': scaler,
        'predictions': ridge_pred,
        'r2': r2_score(y, ridge_pred),
        'rmse': np.sqrt(mean_squared_error(y, ridge_pred)),
        'mae': mean_absolute_error(y, ridge_pred)
    }
    
    # 3. Random Forest (non-linear relationships)
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict(X)
    
    models['Random Forest'] = {
        'model': rf_model,
        'scaler': None,
        'predictions': rf_pred,
        'r2': r2_score(y, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y, rf_pred)),
        'mae': mean_absolute_error(y, rf_pred)
    }
    
    # Find best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
    best_model = models[best_model_name]
    
    # Feature importance for best model
    if best_model_name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model['model'].feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': best_model['model'].coef_
        }).sort_values('coefficient', key=abs, ascending=False)
    
    # Cross-validation scores
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(best_model['model'], X_scaled if best_model['scaler'] else X, y, cv=5, scoring='r2')
    
    return {
        'models': models,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'feature_importance': feature_importance,
        'predictions': best_model['predictions'],
        'actual': y.values,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'all_metrics': {
            name: {
                'r2': model['r2'],
                'rmse': model['rmse'],
                'mae': model['mae']
            } for name, model in models.items()
        }
    }

def create_global_analysis_charts():
    """Create global analysis charts"""
    states_data = get_states_data('latest')  # Use latest data for global analysis
    
    # Correlation matrix - use available columns
    available_numeric_cols = ['home_value', 'unemployment_rate', 'price_growth_yoy', 'population', 'gdp_per_capita', 'value_to_income_ratio']
    corr_matrix = states_data[available_numeric_cols].corr()
    
    # Create correlation heatmap
    corr_fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix - Market Metrics",
        color_continuous_scale='RdBu'
    )
    
    # Distribution plots
    dist_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Home Value Distribution', 'Mortgage Rate Distribution', 
                       'Unemployment Rate Distribution', 'Price Growth Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    dist_fig.add_trace(go.Histogram(x=states_data['home_value'], name='Home Value'), row=1, col=1)
    dist_fig.add_trace(go.Histogram(x=states_data['value_to_income_ratio'], name='Value/Income Ratio'), row=1, col=2)
    dist_fig.add_trace(go.Histogram(x=states_data['unemployment_rate'], name='Unemployment'), row=2, col=1)
    dist_fig.add_trace(go.Histogram(x=states_data['price_growth_yoy'], name='Price Growth'), row=2, col=2)
    
    dist_fig.update_layout(height=600, title_text="Market Metrics Distribution")
    
    return corr_fig, dist_fig

# Create Dash app
app = dash.Dash(__name__, title="Advanced US Real Estate Analytics Dashboard")

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏠 Advanced US Real Estate Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H3("Interactive State-Level Analysis with Regression Models", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 10}),
        html.H4("Developed by Maksim Kitikov - Upside Analytics", 
                style={'textAlign': 'center', 'color': '#000000', 'marginBottom': 10, 'fontWeight': 'bold'}),
        html.P("📊 Data Source: FRED API (Federal Reserve Economic Data)", 
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
        html.H3("🗺️ US States Interactive Map", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.Label("Select Metric for Map:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='map-metric-selector',
                    options=[
                        {'label': 'Home Value ($)', 'value': 'home_value'},
                        {'label': 'Price Growth YoY (%)', 'value': 'price_growth_yoy'},
                        {'label': 'Unemployment Rate (%)', 'value': 'unemployment_rate'},
                        {'label': 'Income Growth (%)', 'value': 'income_growth'},
                        {'label': 'GDP per Capita ($)', 'value': 'gdp_per_capita'},
                        {'label': 'Population', 'value': 'population'},
                        {'label': 'Days on Market', 'value': 'median_days_on_market'},
                        {'label': 'Months of Supply', 'value': 'months_of_supply'},
                        {'label': 'Homeownership Rate (%)', 'value': 'homeownership_rate'},
                        {'label': 'Education Rate (%)', 'value': 'education_rate'},
                        {'label': 'Divorce Rate (%)', 'value': 'divorce_rate'},
                        {'label': 'Risk Score', 'value': 'risk_score'},
                        {'label': 'Value/Income Ratio', 'value': 'value_to_income_ratio'},
                        {'label': 'Disaster Rate (3yr avg)', 'value': 'disaster_rate_3yr'},
                        {'label': 'Political Competitiveness', 'value': 'political_competitiveness'}
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
                        {'label': 'RdBu', 'value': 'RdBu'},
                        {'label': 'RdYlBu', 'value': 'RdYlBu'}
                    ],
                    value='Viridis',
                    style={'width': '200px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Time Period:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='map-time-period-selector',
                options=[
                    {'label': 'Latest Data', 'value': 'latest'},
                    {'label': '1 Month Ago', 'value': '1m'},
                    {'label': '3 Months Ago', 'value': '3m'},
                    {'label': '6 Months Ago', 'value': '6m'},
                    {'label': '1 Year Ago', 'value': '1y'},
                    {'label': '2 Years Ago', 'value': '2y'}
                ],
                value='latest',
                style={'width': '200px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center'}),
        
        # US States Map
        html.Div([
            dcc.Graph(id='us-states-map', style={'height': '600px'})
        ], style={'marginBottom': '30px'}),
        
        # Metrics Description
        html.Div([
            html.H4("📋 Metrics Description", style={'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Strong("🏠 Home Value:"), " Median home prices by state",
                    html.Br(),
                    html.Strong("📈 Price Growth YoY:"), " Year-over-year home price growth (%)",
                    html.Br(),
                    html.Strong("📊 Unemployment Rate:"), " State unemployment rate (%)",
                    html.Br(),
                    html.Strong("💰 Income Growth:"), " Annual household income growth (%)",
                    html.Br(),
                    html.Strong("🌍 GDP per Capita:"), " Economic output per person ($)",
                    html.Br(),
                    html.Strong("👥 Population:"), " Total state population"
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'fontSize': '12px'}),
                
                html.Div([
                    html.Strong("⏱️ Days on Market:"), " Average time homes stay listed (days)",
                    html.Br(),
                    html.Strong("📦 Months of Supply:"), " Housing inventory in months",
                    html.Br(),
                    html.Strong("🏘️ Homeownership Rate:"), " % of households that own homes",
                    html.Br(),
                    html.Strong("🎓 Education Rate:"), " % of population with college degree",
                    html.Br(),
                    html.Strong("💔 Divorce Rate:"), " Divorce rate per 1000 people",
                    html.Br(),
                    html.Strong("⚠️ Risk Score:"), " Composite real estate risk (0-100)"
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'fontSize': '12px'}),
                
                html.Div([
                    html.Strong("💵 Value/Income Ratio:"), " Home price to income ratio",
                    html.Br(),
                    html.Strong("🌪️ Disaster Rate (3yr avg):"), " Natural disasters per year",
                    html.Br(),
                    html.Strong("🗳️ Political Competitiveness:"), " Electoral competitiveness score",
                    html.Br(),
                    html.Strong("🌡️ Market Temperature:"), " Market activity indicator",
                    html.Br(),
                    html.Strong("⏰ Time Period:"), " Historical data comparison"
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'fontSize': '12px'})
            ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '10px'})
        ], style={'marginBottom': '30px'})
    ]),
    

    
    # Regression Analysis Section
    html.Div([
        html.H3("📊 Regression Analysis (R² Score)", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.H4("Model Performance", style={'textAlign': 'center'}),
                html.Div(id='regression-metrics', style={'textAlign': 'center', 'marginBottom': '20px'})
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                dcc.Graph(id='regression-chart', style={'height': '400px'})
            ], style={'width': '70%', 'display': 'inline-block'})
        ], style={'marginBottom': '30px'}),
        
        html.Div([
            dcc.Graph(id='feature-importance-chart', style={'height': '400px'})
        ], style={'marginBottom': '30px'})
    ]),
    
    # Risk Analysis Section
    html.Div([
        html.H3("⚠️ Real Estate Risk Analysis", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.H4("Risk Factors for Real Estate Companies", style={'textAlign': 'center'}),
                html.Div(id='risk-summary', style={'textAlign': 'center', 'marginBottom': '20px'})
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                dcc.Graph(id='risk-map', style={'height': '500px'})
            ], style={'width': '70%', 'display': 'inline-block'})
        ], style={'marginBottom': '30px'}),
        
        html.Div([
            dcc.Graph(id='risk-breakdown', style={'height': '400px'})
        ], style={'marginBottom': '30px'})
    ]),
    
    # Global Analysis Section
    html.Div([
        html.H3("🌍 Global Market Analysis", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            dcc.Graph(id='correlation-heatmap', style={'height': '500px'})
        ], style={'marginBottom': '30px'}),
        
        html.Div([
            dcc.Graph(id='distribution-charts', style={'height': '600px'})
        ], style={'marginBottom': '30px'})
    ]),
    
    # Time Series Analysis
    html.Div([
        html.H3("📈 Time Series Analysis", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                html.Label("Select Metric:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='timeseries-metric-selector',
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
                    id='timeseries-period-selector',
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
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            dcc.Graph(id='timeseries-chart', style={'height': '500px'})
        ], style={'marginBottom': '30px'})
    ]),
    
    # System Status
    html.Div([
        html.H3("📊 System Status", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        html.Div(id='system-status', style={'textAlign': 'center'})
    ]),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P("Advanced US Real Estate Analytics System - Production Ready", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
        html.P("Data Sources: FRED, BLS, BEA, Census, HUD, FHFA, Redfin | Features: State-Level Analysis, Regression Models", 
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
    [Input('map-metric-selector', 'value')]
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
    Output('us-states-map', 'figure'),
    [Input('map-metric-selector', 'value'),
     Input('color-scale-selector', 'value'),
     Input('map-time-period-selector', 'value')]
)
def update_us_map(selected_metric, color_scale, time_period):
    """Update US states map"""
    try:
        states_data = get_states_data(time_period)
        
        # Add time period info to title
        time_period_label = {
            'latest': 'Latest Data',
            '1m': '1 Month Ago',
            '3m': '3 Months Ago', 
            '6m': '6 Months Ago',
            '1y': '1 Year Ago',
            '2y': '2 Years Ago'
        }.get(time_period, 'Latest Data')
        
        title_suffix = f" - {time_period_label}"
        
        if selected_metric not in states_data.columns:
            return go.Figure()
        
        if states_data[selected_metric].isna().any():
            states_data[selected_metric] = states_data[selected_metric].fillna(0)
        
        fig = px.choropleth(
            states_data,
            locations='state',
            locationmode='USA-states',
            color=selected_metric,
            hover_name='state_name',
            hover_data=[selected_metric, 'state_name'],
            color_continuous_scale=color_scale,
            scope='usa',
            title=f"US States - {selected_metric.replace('_', ' ').title()}{title_suffix}",
            labels={selected_metric: selected_metric.replace('_', ' ').title()},
            range_color=[states_data[selected_metric].min(), states_data[selected_metric].max()]
        )
        
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
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=0.5,
            colorbar=dict(
                title=selected_metric.replace('_', ' ').title(),
                thickness=20,
                len=0.5
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating US map: {e}")
        fig = go.Figure()
        fig.update_layout(
            title="Error creating map",
            geo=dict(scope='usa', projection=dict(type='albers usa'))
        )
        return fig



@app.callback(
    [Output('regression-metrics', 'children'),
     Output('regression-chart', 'figure'),
     Output('feature-importance-chart', 'figure')],
    [Input('map-metric-selector', 'value')]
)
def update_regression_analysis(selected_metric):
    """Update comprehensive regression analysis"""
    try:
        # Perform regression analysis
        reg_results = perform_regression_analysis()
        
        # Metrics display with multiple models
        metrics_html = [
            html.Div([
                html.H5(f"Best Model: {reg_results['best_model_name']}", style={'color': '#27ae60', 'fontWeight': 'bold'}),
                html.H5(f"R² Score: {reg_results['best_model']['r2']:.4f}", style={'color': '#27ae60'}),
                html.H5(f"RMSE: ${reg_results['best_model']['rmse']:,.0f}", style={'color': '#e74c3c'}),
                html.H5(f"MAE: ${reg_results['best_model']['mae']:,.0f}", style={'color': '#f39c12'}),
                html.H5(f"CV R²: {reg_results['cv_mean']:.4f} ± {reg_results['cv_std']:.4f}", style={'color': '#9b59b6'}),
                html.P("Features: Unemployment, Population, GDP, Median Income", style={'fontSize': '12px'}),
                html.P("Models: Linear, Ridge, Random Forest", style={'fontSize': '12px'})
            ])
        ]
        
        # Regression chart
        states_data = get_states_data()
        reg_fig = go.Figure()
        
        reg_fig.add_trace(go.Scatter(
            x=reg_results['actual'],
            y=reg_results['predictions'],
            mode='markers',
            name='Predictions vs Actual',
            marker=dict(color='#3498db', size=8)
        ))
        
        # Add perfect prediction line
        min_val = min(reg_results['actual'].min(), reg_results['predictions'].min())
        max_val = max(reg_results['actual'].max(), reg_results['predictions'].max())
        reg_fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        reg_fig.update_layout(
            title=f"Regression Analysis: Predicted vs Actual Home Values (R² = {reg_results['best_model']['r2']:.4f})",
            xaxis_title="Actual Home Value ($)",
            yaxis_title="Predicted Home Value ($)",
            height=400
        )
        
        # Feature importance chart
        if reg_results['best_model_name'] == 'Random Forest':
            y_col = 'importance'
            title = "Feature Importance (Random Forest)"
        else:
            y_col = 'coefficient'
            title = "Feature Importance (Regression Coefficients)"
            
        importance_fig = px.bar(
            reg_results['feature_importance'],
            x='feature',
            y=y_col,
            title=title,
            color=y_col,
            color_continuous_scale='RdBu'
        )
        
        importance_fig.update_layout(height=400)
        
        return metrics_html, reg_fig, importance_fig
        
    except Exception as e:
        logger.error(f"Error updating regression analysis: {e}")
        return "Error", go.Figure(), go.Figure()

@app.callback(
    [Output('risk-summary', 'children'),
     Output('risk-map', 'figure'),
     Output('risk-breakdown', 'figure')],
    [Input('map-metric-selector', 'value')]
)
def update_risk_analysis(selected_metric):
    """Update real estate risk analysis"""
    try:
        states_data = get_states_data('latest')  # Use latest data for risk analysis
        risk_factors = calculate_real_estate_risk_score(states_data)
        
        # Add risk scores to states data
        risk_scores = []
        risk_categories = []
        risk_colors = []
        
        for idx in states_data.index:
            risk_info = risk_factors.get(idx, {})
            risk_scores.append(risk_info.get('risk_score', 0))
            risk_categories.append(risk_info.get('risk_category', 'Unknown'))
            risk_colors.append(risk_info.get('risk_color', '#95a5a6'))
        
        states_data['risk_score'] = risk_scores
        states_data['risk_category'] = risk_categories
        states_data['risk_color'] = risk_colors
        
        # Risk summary
        low_risk = len([r for r in risk_categories if r == 'Low Risk'])
        moderate_risk = len([r for r in risk_categories if r == 'Moderate Risk'])
        high_risk = len([r for r in risk_categories if r == 'High Risk'])
        very_high_risk = len([r for r in risk_categories if r == 'Very High Risk'])
        
        avg_risk = np.mean(risk_scores)
        
        risk_summary = [
            html.Div([
                html.H5(f"Average Risk Score: {avg_risk:.1f}/100", style={'color': '#e74c3c', 'fontWeight': 'bold'}),
                html.P(f"Low Risk States: {low_risk}", style={'color': '#27ae60'}),
                html.P(f"Moderate Risk States: {moderate_risk}", style={'color': '#f39c12'}),
                html.P(f"High Risk States: {high_risk}", style={'color': '#e67e22'}),
                html.P(f"Very High Risk States: {very_high_risk}", style={'color': '#e74c3c'}),
                html.Hr(),
                html.P("Risk Factors:", style={'fontWeight': 'bold'}),
                html.P("• Economic (Unemployment, GDP) - 45%", style={'fontSize': '12px'}),
                html.P("• Market (Income, Population) - 45%", style={'fontSize': '12px'}),
                html.P("• Housing (Education, Homeownership) - 10%", style={'fontSize': '12px'})
            ])
        ]
        
        # Risk map
        risk_fig = px.choropleth(
            states_data,
            locations='state',
            locationmode='USA-states',
            color='risk_score',
            hover_name='state_name',
            hover_data=['risk_score', 'risk_category', 'unemployment_rate', 'value_to_income_ratio'],
            color_continuous_scale='RdYlGn_r',  # Red (high risk) to Green (low risk)
            scope='usa',
            title="Real Estate Risk Score by State",
            labels={'risk_score': 'Risk Score (0-100)'},
            range_color=[0, 100]
        )
        
        risk_fig.update_layout(
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
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        risk_fig.update_traces(
            marker_line_color='white',
            marker_line_width=0.5,
            colorbar=dict(
                title='Risk Score',
                thickness=20,
                len=0.5
            )
        )
        
        # Risk breakdown chart
        risk_breakdown_data = []
        for idx, row in states_data.iterrows():
            risk_info = risk_factors.get(idx, {})
            risk_details = risk_info.get('risk_details', {})
            
            risk_breakdown_data.append({
                'state': row['state_name'],
                'unemployment_risk': risk_details.get('unemployment_risk', 0),
                'gdp_risk': risk_details.get('gdp_risk', 0),
                'income_risk': risk_details.get('income_risk', 0),
                'population_risk': risk_details.get('population_risk', 0),
                'education_risk': risk_details.get('education_risk', 0),
                'homeownership_risk': risk_details.get('homeownership_risk', 0)
            })
        
        breakdown_df = pd.DataFrame(risk_breakdown_data)
        
        # Top 10 highest risk states
        top_risk_states = states_data.nlargest(10, 'risk_score')
        
        breakdown_fig = go.Figure()
        
        breakdown_fig.add_trace(go.Bar(
            x=top_risk_states['state_name'],
            y=top_risk_states['risk_score'],
            marker_color=top_risk_states['risk_color'],
            text=top_risk_states['risk_score'].round(1),
            textposition='auto',
            name='Total Risk Score'
        ))
        
        breakdown_fig.update_layout(
            title="Top 10 Highest Risk States for Real Estate",
            xaxis_title="State",
            yaxis_title="Risk Score (0-100)",
            height=400,
            showlegend=False
        )
        
        return risk_summary, risk_fig, breakdown_fig
        
    except Exception as e:
        logger.error(f"Error updating risk analysis: {e}")
        error_div = html.Div("Error calculating risk analysis")
        error_fig = go.Figure()
        error_fig.update_layout(title="Error creating risk analysis")
        return error_div, error_fig, error_fig

@app.callback(
    [Output('correlation-heatmap', 'figure'),
     Output('distribution-charts', 'figure')],
    [Input('map-metric-selector', 'value')]
)
def update_global_analysis(selected_metric):
    """Update global analysis charts"""
    try:
        corr_fig, dist_fig = create_global_analysis_charts()
        return corr_fig, dist_fig
    except Exception as e:
        logger.error(f"Error updating global analysis: {e}")
        return go.Figure(), go.Figure()

@app.callback(
    Output('timeseries-chart', 'figure'),
    [Input('timeseries-metric-selector', 'value'),
     Input('timeseries-period-selector', 'value')]
)
def update_timeseries_chart(selected_metric, period_days):
    """Update time series chart"""
    try:
        df = get_metric_history(selected_metric, period_days)
        
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"No data available for {selected_metric}",
                xaxis_title="Date",
                yaxis_title="Value",
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
            'DGS10': '10-Year Treasury Rate',
            'T10Y2Y': 'Yield Curve Spread'
        }
        
        fig.update_layout(
            title=f"{metric_names.get(selected_metric, selected_metric)} - Historical Data",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            hovermode='x unified',
            showlegend=False,
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating timeseries chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title="Error loading data",
            template="plotly_white"
        )
        return fig

@app.callback(
    Output('system-status', 'children'),
    [Input('map-metric-selector', 'value')]
)
def update_system_status(selected_metric):
    """Update system status display"""
    try:
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
                html.Strong("States Coverage: "), "50 states + DC",
                html.Br(),
                html.Strong("Database: "), "✅ Connected",
                html.Br(),
                html.Strong("Interactive Maps: "), "✅ Active",
                html.Br(),
    
                html.Br(),
                html.Strong("Regression Models: "), "✅ R² Analysis Active",
                html.Br(),
                html.Strong("Last Update: "), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ], style={'textAlign': 'left', 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
        ]
        
        return status_html
        
    except Exception as e:
        logger.error(f"Error updating system status: {e}")
        return html.Div("Error loading system status", style={'color': 'red'})

# Export server for Gunicorn
server = app.server

if __name__ == '__main__':
    print("🚀 Starting Advanced US Real Estate Analytics Dashboard...")
    print("🌐 Open http://localhost:8050 in your browser")
    print("🗺️ Features: Interactive US States Map, Regression Analysis")
    print("📊 Production-ready analytics platform with R² scoring")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
