#!/bin/bash

echo "🚀 Запуск Real Estate Analytics Dashboard с ngrok..."

# Проверяем виртуальное окружение
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Виртуальное окружение активировано"
else
    echo "❌ Виртуальное окружение не найдено. Запустите setup_mac.sh сначала"
    exit 1
fi

# Устанавливаем зависимости
pip install -r requirements.txt > /dev/null 2>&1

# Останавливаем предыдущие процессы
echo "🛑 Остановка предыдущих процессов..."
pkill -f "python3 dashboard.py" || true
pkill -f "ngrok" || true

# Запускаем дашборд в фоне
echo "🌐 Запуск дашборда..."
python3 dashboard.py &
DASHBOARD_PID=$!

# Ждем запуска дашборда
sleep 5

# Запускаем ngrok туннель
echo "🔗 Создание ngrok туннеля..."
ngrok http 8050 > /dev/null 2>&1 &
NGROK_PID=$!

# Ждем запуска ngrok
sleep 3

# Получаем публичную ссылку
echo "📡 Получение публичной ссылки..."
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data['tunnels']:
    print(data['tunnels'][0]['public_url'])
else:
    print('Ошибка получения ссылки')
")

echo ""
echo "🎉 Дашборд запущен!"
echo "📊 Локальная ссылка: http://localhost:8050"
echo "🌍 Публичная ссылка: $PUBLIC_URL"
echo ""
echo "📋 Скопируйте эту ссылку для отправки:"
echo "$PUBLIC_URL"
echo ""
echo "🛑 Для остановки нажмите Ctrl+C"

# Функция очистки при выходе
cleanup() {
    echo ""
    echo "🛑 Остановка процессов..."
    kill $DASHBOARD_PID 2>/dev/null || true
    kill $NGROK_PID 2>/dev/null || true
    pkill -f "python3 dashboard.py" || true
    pkill -f "ngrok" || true
    echo "✅ Процессы остановлены"
    exit 0
}

# Перехватываем Ctrl+C
trap cleanup SIGINT

# Ждем завершения
wait
