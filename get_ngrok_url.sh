#!/bin/bash

echo "🔍 Получение текущей ngrok ссылки..."

# Проверяем, запущен ли ngrok
if ! pgrep -f "ngrok" > /dev/null; then
    echo "❌ ngrok не запущен. Запустите ./start_dashboard_ngrok.sh сначала"
    exit 1
fi

# Получаем ссылку
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data['tunnels']:
        print(data['tunnels'][0]['public_url'])
    else:
        print('Ошибка: туннели не найдены')
except Exception as e:
    print(f'Ошибка получения ссылки: {e}')
")

if [[ $PUBLIC_URL == *"https://"* ]]; then
    echo "✅ Публичная ссылка:"
    echo "$PUBLIC_URL"
    echo ""
    echo "📋 Скопируйте эту ссылку для отправки"
else
    echo "❌ $PUBLIC_URL"
fi
