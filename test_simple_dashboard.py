#!/usr/bin/env python3
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Тестовый дашборд"),
    dcc.Graph(id='test-chart'),
    html.Div(id='test-output')
])

@app.callback(
    Output('test-chart', 'figure'),
    Output('test-output', 'children'),
    Input('test-chart', 'clickData')
)
def update_test(click_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='lines'))
    fig.update_layout(title="Тестовый график")
    
    output_text = f"Клик: {click_data}" if click_data else "Нет клика"
    
    return fig, output_text

if __name__ == '__main__':
    print("🚀 Запуск тестового дашборда...")
    print("🌐 Откройте http://localhost:8051 в браузере")
    app.run_server(debug=True, host='0.0.0.0', port=8051)
