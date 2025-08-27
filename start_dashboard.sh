#!/bin/bash

# Скрипт для запуска Real Estate Analytics Dashboard

echo "🚀 Запуск Real Estate Analytics Dashboard..."

# Активация виртуального окружения
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Виртуальное окружение активировано"
else
    echo "❌ Виртуальное окружение не найдено. Запустите setup_mac.sh сначала"
    exit 1
fi

# Проверка зависимостей
echo "📦 Проверка зависимостей..."
pip install -r requirements.txt > /dev/null 2>&1

# Запуск дашборда
echo "🌐 Запуск дашборда..."
echo "📊 Откройте http://localhost:8050 в браузере"
echo "🛑 Для остановки нажмите Ctrl+C"
echo ""

python3 dashboard.py
