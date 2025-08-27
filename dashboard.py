#!/usr/bin/env python3
"""
Real Estate Analytics Dashboard
Простой дашборд на Python с Plotly для визуализации данных
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
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Подключение к БД
DB_URL = os.getenv("DATABASE_URL", "")
if not DB_URL:
    print("❌ DATABASE_URL не установлен в .env")
    exit(1)

engine = create_engine(DB_URL, pool_pre_ping=True)

def safe_query(query, default_df=None):
    """Безопасное выполнение запроса с обработкой ошибок"""
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Ошибка запроса: {e}")
        return default_df if default_df is not None else pd.DataFrame()

def get_mortgage_data():
    """Получение данных по ипотечным ставкам"""
    query = """
    SELECT date, mortgage_rate 
    FROM vw_pbi_mortgage_rates 
    WHERE date >= '2020-01-01' AND mortgage_rate IS NOT NULL
    ORDER BY date
    """
    return safe_query(query, pd.DataFrame(columns=['date', 'mortgage_rate']))

def get_cpi_data():
    """Получение данных CPI"""
    query = """
    SELECT date, cpi 
    FROM vw_pbi_cpi 
    WHERE date >= '2020-01-01' AND cpi IS NOT NULL
    ORDER BY date
    """
    return safe_query(query, pd.DataFrame(columns=['date', 'cpi']))

def get_unemployment_data():
    """Получение данных по безработице"""
    query = """
    SELECT date, unemployment_rate 
    FROM vw_pbi_unemployment 
    WHERE date >= '2020-01-01' AND unemployment_rate IS NOT NULL
    ORDER BY date
    """
    return safe_query(query, pd.DataFrame(columns=['date', 'unemployment_rate']))

def get_acs_data():
    """Получение данных ACS"""
    query = """
    SELECT year, median_gross_rent_usd, median_home_value_usd
    FROM vw_pbi_acs_rent_value 
    WHERE median_gross_rent_usd IS NOT NULL OR median_home_value_usd IS NOT NULL
    ORDER BY year
    """
    return safe_query(query, pd.DataFrame(columns=['year', 'median_gross_rent_usd', 'median_home_value_usd']))

def get_staging_data():
    """Получение данных из staging таблиц для дополнительной информации"""
    # FRED данные
    fred_query = """
    SELECT series_id, COUNT(*) as record_count, 
           MIN(obs_date) as min_date, MAX(obs_date) as max_date,
           AVG(value) as avg_value
    FROM stg_fred_series 
    GROUP BY series_id
    """
    fred_df = safe_query(fred_query, pd.DataFrame())
    
    # BLS данные
    bls_query = """
    SELECT COUNT(*) as record_count, 
           MIN(year) as min_year, MAX(year) as max_year,
           AVG(value) as avg_value
    FROM stg_bls_cpi
    """
    bls_df = safe_query(bls_query, pd.DataFrame())
    
    # Census данные
    census_query = """
    SELECT COUNT(*) as record_count, 
           MIN(year) as min_year, MAX(year) as max_year
    FROM stg_census_acs
    """
    census_df = safe_query(census_query, pd.DataFrame())
    
    return fred_df, bls_df, census_df

# Создание Dash приложения
app = dash.Dash(__name__, title="Real Estate Analytics Dashboard")

app.layout = html.Div([
    html.H1("🏠 Real Estate Analytics Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Статистика источников данных
    html.Div([
        html.H3("📊 Статистика источников данных", 
                style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': 20}),
        html.Div(id='data-stats')
    ]),
    
    # Фильтры
    html.Div([
        html.Label("Период анализа:", style={'fontWeight': 'bold'}),
        dcc.DatePickerRange(
            id='date-range',
            start_date='2020-01-01',
            end_date='2024-12-31',
            display_format='YYYY-MM-DD'
        )
    ], style={'marginBottom': 20}),
    
    # Графики
    html.Div([
        # Ипотечные ставки
        html.Div([
            html.H3("📈 Ипотечные ставки (30Y Fixed)", 
                    style={'textAlign': 'center', 'color': '#34495e'}),
            dcc.Graph(id='mortgage-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        # CPI
        html.Div([
            html.H3("💰 Индекс потребительских цен (CPI)", 
                    style={'textAlign': 'center', 'color': '#34495e'}),
            dcc.Graph(id='cpi-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        # Безработица
        html.Div([
            html.H3("📊 Уровень безработицы", 
                    style={'textAlign': 'center', 'color': '#34495e'}),
            dcc.Graph(id='unemployment-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        # ACS данные
        html.Div([
            html.H3("🏘️ Медианная рента и стоимость жилья", 
                    style={'textAlign': 'center', 'color': '#34495e'}),
            dcc.Graph(id='acs-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    # Ключевые метрики
    html.Div([
        html.H3("📊 Ключевые метрики", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': 30}),
        html.Div(id='metrics')
    ])
])

@app.callback(
    [Output('mortgage-chart', 'figure'),
     Output('cpi-chart', 'figure'),
     Output('unemployment-chart', 'figure'),
     Output('acs-chart', 'figure'),
     Output('metrics', 'children'),
     Output('data-stats', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_charts(start_date, end_date):
    """Обновление графиков при изменении фильтров"""
    
    try:
        # Получение данных
        mortgage_df = get_mortgage_data()
        cpi_df = get_cpi_data()
        unemployment_df = get_unemployment_data()
        acs_df = get_acs_data()
        fred_df, bls_df, census_df = get_staging_data()
        
        # Фильтрация по датам
        if start_date and end_date:
            try:
                # Конвертируем строки дат в datetime объекты
                start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
                end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                
                mortgage_df = mortgage_df[(mortgage_df['date'] >= start_dt) & 
                                         (mortgage_df['date'] <= end_dt)]
                cpi_df = cpi_df[(cpi_df['date'] >= start_dt) & 
                               (cpi_df['date'] <= end_dt)]
                unemployment_df = unemployment_df[(unemployment_df['date'] >= start_dt) & 
                                                (unemployment_df['date'] <= end_dt)]
            except (ValueError, TypeError) as e:
                logger.warning(f"Ошибка фильтрации дат: {e}, используем все данные")
                # В случае ошибки используем все данные без фильтрации
        
        # График ипотечных ставок
        mortgage_fig = go.Figure()
        if not mortgage_df.empty:
            mortgage_fig.add_trace(go.Scatter(
                x=mortgage_df['date'], 
                y=mortgage_df['mortgage_rate'],
                mode='lines',
                name='30Y Fixed Rate',
                line=dict(color='#3498db', width=2)
            ))
        mortgage_fig.update_layout(
            title="Ипотечные ставки",
            xaxis_title="Дата",
            yaxis_title="Ставка (%)",
            template="plotly_white",
            height=400
        )
        
        # График CPI
        cpi_fig = go.Figure()
        if not cpi_df.empty:
            cpi_fig.add_trace(go.Scatter(
                x=cpi_df['date'], 
                y=cpi_df['cpi'],
                mode='lines',
                name='CPI',
                line=dict(color='#e74c3c', width=2)
            ))
        cpi_fig.update_layout(
            title="Индекс потребительских цен",
            xaxis_title="Дата",
            yaxis_title="CPI",
            template="plotly_white",
            height=400
        )
        
        # График безработицы
        unemployment_fig = go.Figure()
        if not unemployment_df.empty:
            unemployment_fig.add_trace(go.Scatter(
                x=unemployment_df['date'], 
                y=unemployment_df['unemployment_rate'],
                mode='lines',
                name='Unemployment Rate',
                line=dict(color='#f39c12', width=2)
            ))
        unemployment_fig.update_layout(
            title="Уровень безработицы",
            xaxis_title="Дата",
            yaxis_title="Безработица (%)",
            template="plotly_white",
            height=400
        )
        
        # График ACS
        acs_fig = make_subplots(specs=[[{"secondary_y": True}]])
        if not acs_df.empty:
            if 'median_gross_rent_usd' in acs_df.columns and not acs_df['median_gross_rent_usd'].isna().all():
                acs_fig.add_trace(
                    go.Bar(x=acs_df['year'], y=acs_df['median_gross_rent_usd'], 
                           name="Медианная рента", marker_color='#9b59b6'),
                    secondary_y=False
                )
            if 'median_home_value_usd' in acs_df.columns and not acs_df['median_home_value_usd'].isna().all():
                acs_fig.add_trace(
                    go.Scatter(x=acs_df['year'], y=acs_df['median_home_value_usd'], 
                              name="Стоимость жилья", line=dict(color='#e67e22')),
                    secondary_y=True
                )
        acs_fig.update_layout(
            title="Медианная рента и стоимость жилья",
            xaxis_title="Год",
            template="plotly_white",
            height=400
        )
        acs_fig.update_yaxes(title_text="Рента ($)", secondary_y=False)
        acs_fig.update_yaxes(title_text="Стоимость жилья ($)", secondary_y=True)
        
        # Ключевые метрики
        latest_mortgage = mortgage_df['mortgage_rate'].iloc[-1] if not mortgage_df.empty else 0
        latest_cpi = cpi_df['cpi'].iloc[-1] if not cpi_df.empty else 0
        latest_unemployment = unemployment_df['unemployment_rate'].iloc[-1] if not unemployment_df.empty else 0
        
        metrics = html.Div([
            html.Div([
                html.H4(f"📈 Текущая ипотечная ставка: {latest_mortgage:.2f}%"),
                html.H4(f"💰 Текущий CPI: {latest_cpi:.2f}"),
                html.H4(f"📊 Текущая безработица: {latest_unemployment:.2f}%")
            ], style={'textAlign': 'center'})
        ])
        
        # Статистика источников данных
        stats_html = []
        
        # FRED статистика
        if not fred_df.empty:
            for _, row in fred_df.iterrows():
                stats_html.append(html.Div([
                    html.Strong(f"FRED {row['series_id']}: "),
                    f"{row['record_count']} записей, ",
                    f"период: {row['min_date']} - {row['max_date']}, ",
                    f"среднее: {row['avg_value']:.2f}"
                ]))
        
        # BLS статистика
        if not bls_df.empty:
            stats_html.append(html.Div([
                html.Strong("BLS CPI: "),
                f"{bls_df.iloc[0]['record_count']} записей, ",
                f"период: {bls_df.iloc[0]['min_year']} - {bls_df.iloc[0]['max_year']}, ",
                f"среднее: {bls_df.iloc[0]['avg_value']:.2f}"
            ]))
        
        # Census статистика
        if not census_df.empty:
            stats_html.append(html.Div([
                html.Strong("Census ACS: "),
                f"{census_df.iloc[0]['record_count']} записей, ",
                f"период: {census_df.iloc[0]['min_year']} - {census_df.iloc[0]['max_year']}"
            ]))
        
        data_stats = html.Div(stats_html, style={'textAlign': 'center', 'marginBottom': 20})
        
        return mortgage_fig, cpi_fig, unemployment_fig, acs_fig, metrics, data_stats
        
    except Exception as e:
        logger.error(f"Ошибка в callback: {e}")
        # Возвращаем пустые графики в случае ошибки
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Данные недоступны",
            template="plotly_white",
            height=400
        )
        
        error_msg = html.Div([
            html.H4("❌ Ошибка загрузки данных"),
            html.P(f"Ошибка: {str(e)}"),
            html.P("Проверьте подключение к базе данных и наличие данных")
        ], style={'textAlign': 'center', 'color': 'red'})
        
        return empty_fig, empty_fig, empty_fig, empty_fig, error_msg, error_msg

if __name__ == '__main__':
    print("🚀 Запуск Real Estate Analytics Dashboard...")
    print("🌐 Откройте http://localhost:8050 в браузере")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
