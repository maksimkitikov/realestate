#!/bin/bash

# Real Estate Analytics Setup Script для macOS
# Автоматическая настройка и запуск ETL-пайплайна

set -e  # Остановка при ошибке

echo "🚀 Real Estate Analytics Setup для macOS"
echo "========================================"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.12+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $PYTHON_VERSION найден"

# Создание и активация виртуального окружения
echo ""
echo "📦 Создание виртуального окружения..."
if [ -d "venv" ]; then
    echo "⚠️ Виртуальное окружение уже существует, пересоздаю..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
echo "✓ Виртуальное окружение создано и активировано"

# Обновление pip
echo ""
echo "⬆️ Обновление pip..."
pip install --upgrade pip
echo "✓ pip обновлен"

# Установка зависимостей
echo ""
echo "📚 Установка зависимостей..."
pip install -r requirements.txt
echo "✓ Зависимости установлены"

# Создание .env файла с ключами
echo ""
echo "🔑 Настройка .env файла..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# API Keys
FRED_API_KEY=d7f56f7a50b44e780eb04b79cdcdd9b2
HUD_API_KEY=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI2IiwianRpIjoiYWRhODQyM2MwNzcwNmFhNTUwMzBjODQ5ZjdlZDdiNjIzYTQ5ZWVlYmJmMjI2NTkxN2Y1ZjM3NTA2MjU1OGE5ZDRkNzI0N2MzMmE5MGYxMTAiLCJpYXQiOjE3NTUxNzk2NDMuNTYxOTQzLCJuYmYiOjE3NTUxNzk2NDMuNTYxOTQ1LCJleHAiOjIwNzA3MTI0NDMuNTU3NTA5LCJzdWIiOiIxMDYwMTQiLCJzY29wZXMiOltdfQ.ECcjUPBU8-Qa_yyzFa41uGoKwFpkABAffnxckUektXmgkO7TNHWe20UbX_aKqDc03673OHM-Wc1lfW-LHnRWTQ
BEA_API_KEY=E6B8945F-F23B-4F40-B8ED-C02729F3B398
BLS_API_KEY=fc3c9ba2ac9546669cc41b719f4f1e51
CENSUS_API_KEY=cb539edde53a3ffe7f21b74441860717446bd3b9

# Database URL (Neon Postgres)
DATABASE_URL=postgresql+psycopg2://neondb_owner:npg_BTXFC4e2udvV@ep-icy-sunset-a1dfl5uk-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

# HTTP timeout settings
REQUESTS_TIMEOUT_SECONDS=60
EOF
    echo "✓ .env файл создан с API ключами"
else
    echo "✓ .env файл уже существует"
fi

# Применение схемы БД
echo ""
echo "🗄️ Применение схемы базы данных..."
python3 -c "
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv('DATABASE_URL')
if not db_url:
    print('❌ DATABASE_URL не найден в .env')
    exit(1)

engine = create_engine(db_url, pool_pre_ping=True)
with open('schema.sql', 'r') as f:
    schema = f.read()

with engine.begin() as conn:
    conn.execute(text(schema))
print('✓ Схема БД применена успешно')
"
echo "✓ Схема базы данных применена"

# Запуск тестов
echo ""
echo "🧪 Запуск тестов..."
python3 tests.py
echo "✓ Тесты пройдены"

# Запуск ETL-пайплайна
echo ""
echo "🔄 Запуск ETL-пайплайна..."
python3 run_pipeline.py
echo "✓ ETL-пайплайн завершен"

# Проверка витрин
echo ""
echo "🔍 Проверка витрин Power BI..."
python3 sample_queries.py
echo "✓ Витрины проверены"

# Создание исполняемого скрипта дашборда
echo ""
echo "📊 Настройка дашборда..."
chmod +x start_dashboard.sh
echo "✓ Скрипт дашборда создан"

echo ""
echo "🎉 Настройка завершена успешно!"
echo ""
echo "📊 Для запуска дашборда:"
echo "   ./start_dashboard.sh"
echo ""
echo "📊 Для подключения Power BI:"
echo "   1. Откройте powerbi-neon.pbids в Power BI"
echo "   2. Введите пароль: npg_BTXFC4e2udvV"
echo "   3. Выберите вьюхи и загрузите"
echo ""
echo "📁 Доступные витрины:"
echo "   - vw_pbi_mortgage_rates (ипотечные ставки)"
echo "   - vw_pbi_cpi (индекс потребительских цен)"
echo "   - vw_pbi_unemployment (безработица)"
echo "   - vw_pbi_rent_fmr (рыночная рента)"
echo "   - vw_pbi_affordability_hint (доступность жилья)"
echo "   - vw_pbi_acs_rent_value (медианная рента и стоимость)"
echo ""
echo "🔄 Для повторного запуска пайплайна:"
echo "   source venv/bin/activate && python3 run_pipeline.py"
