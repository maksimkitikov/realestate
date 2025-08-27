#!/usr/bin/env python3
"""
Упрощенная версия Real Estate Analytics Dashboard
"""

import os
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
from dotenv import load_dotenv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Загрузка переменных окружения
load_dotenv()

# Подключение к БД
DB_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DB_URL, pool_pre_ping=True)

def get_mortgage_data():
    """Получение данных по ипотечным ставкам"""
    try:
        query = """
        SELECT date, mortgage_rate 
        FROM vw_pbi_mortgage_rates 
        WHERE date >= '2020-01-01' AND mortgage_rate IS NOT NULL
        ORDER BY date
        LIMIT 100
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

# Создание Dash приложения
app = dash.Dash(__name__, title="Real Estate Analytics Dashboard")

app.layout = html.Div([
    html.H1("🏠 Real Estate Analytics Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # График ипотечных ставок
    html.Div([
        html.H3("📈 Ипотечные ставки (30Y Fixed)", 
                style={'textAlign': 'center', 'color': '#34495e'}),
        dcc.Graph(id='mortgage-chart')
    ]),
    
    # Кнопка обновления
    html.Button('Обновить данные', id='update-button', n_clicks=0),
    
    # Статус
    html.Div(id='status')
])

@app.callback(
    [Output('mortgage-chart', 'figure'),
     Output('status', 'children')],
    [Input('update-button', 'n_clicks')]
)
def update_chart(n_clicks):
    """Обновление графика"""
    try:
        # Получение данных
        df = get_mortgage_data()
        
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Данные недоступны",
                template="plotly_white",
                height=400
            )
            status = "❌ Данные не загружены"
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['mortgage_rate'],
                mode='lines',
                name='30Y Fixed Rate',
                line=dict(color='#3498db', width=2)
            ))
            fig.update_layout(
                title="Ипотечные ставки",
                xaxis_title="Дата",
                yaxis_title="Ставка (%)",
                template="plotly_white",
                height=400
            )
            status = f"✅ Загружено {len(df)} записей"
        
        return fig, status
        
    except Exception as e:
        print(f"Ошибка в callback: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Ошибка загрузки данных",
            template="plotly_white",
            height=400
        )
        return empty_fig, f"❌ Ошибка: {str(e)}"

if __name__ == '__main__':
    print("🚀 Запуск упрощенного Real Estate Analytics Dashboard...")
    print("🌐 Откройте http://localhost:8052 в браузере")
    app.run_server(debug=True, host='0.0.0.0', port=8052)
