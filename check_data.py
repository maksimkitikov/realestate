#!/usr/bin/env python3
"""
Скрипт для проверки данных в базе
Диагностика проблем с данными
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Подключение к БД
DB_URL = os.getenv("DATABASE_URL", "")
if not DB_URL:
    print("❌ DATABASE_URL не установлен в .env")
    exit(1)

engine = create_engine(DB_URL, pool_pre_ping=True)

def check_table(table_name):
    """Проверка таблицы"""
    try:
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        with engine.connect() as conn:
            result = conn.execute(text(query))
            count = result.fetchone()[0]
        print(f"✓ {table_name}: {count} записей")
        return count
    except Exception as e:
        print(f"❌ {table_name}: ошибка - {e}")
        return 0

def check_view(view_name):
    """Проверка витрины"""
    try:
        query = f"SELECT COUNT(*) as count FROM {view_name}"
        with engine.connect() as conn:
            result = conn.execute(text(query))
            count = result.fetchone()[0]
        print(f"✓ {view_name}: {count} записей")
        
        if count > 0:
            # Показываем несколько примеров
            sample_query = f"SELECT * FROM {view_name} ORDER BY 1 DESC LIMIT 3"
            df = pd.read_sql(sample_query, engine)
            print(f"  Примеры данных:")
            print(df.to_string(index=False))
            print()
        
        return count
    except Exception as e:
        print(f"❌ {view_name}: ошибка - {e}")
        return 0

def check_data():
    engine = create_engine(os.getenv('DATABASE_URL'))
    with engine.connect() as conn:
        # Проверяем источники данных
        result = conn.execute(text("""
            SELECT source, metric, freq, COUNT(*) 
            FROM fact_metric 
            WHERE source IN ('FRED','BLS_LAUS','BEA','CENSUS_ACS') 
            GROUP BY source, metric, freq 
            ORDER BY source, metric
        """))
        print("Data sources summary:")
        for row in result:
            print(f"  {row}")
        
        # Проверяем последние значения FRED
        result = conn.execute(text("""
            SELECT metric, value, date 
            FROM fact_metric 
            WHERE source = 'FRED' 
            AND date = (SELECT MAX(date) FROM fact_metric WHERE source = 'FRED')
            ORDER BY metric
        """))
        print("\nLatest FRED values:")
        for row in result:
            print(f"  {row}")
        
        # Проверяем количество штатов с данными
        result = conn.execute(text("""
            SELECT COUNT(DISTINCT geo_key) as states_count
            FROM fact_metric 
            WHERE geo_level = 'STATE'
        """))
        print(f"\nStates with data: {result.fetchone()[0]}")

def main():
    """Основная функция проверки"""
    print("🔍 Проверка данных в базе...")
    print("=" * 60)
    
    # Проверка staging таблиц
    print("\n📊 Staging таблицы:")
    staging_tables = [
        "stg_fred_series",
        "stg_bls_cpi", 
        "stg_census_acs",
        "stg_hud_fmr",
        "stg_hud_chas",
        "stg_bea_gdp"
    ]
    
    for table in staging_tables:
        check_table(table)
    
    # Проверка DWH таблиц
    print("\n🏗️ DWH таблицы:")
    dwh_tables = [
        "dwh_dim_date",
        "dwh_dim_series",
        "dwh_fact_mortgage_rates",
        "dwh_fact_cpi",
        "dwh_fact_unemployment"
    ]
    
    for table in dwh_tables:
        check_table(table)
    
    # Проверка витрин
    print("\n📈 Витрины Power BI:")
    views = [
        "vw_pbi_mortgage_rates",
        "vw_pbi_cpi",
        "vw_pbi_unemployment", 
        "vw_pbi_rent_fmr",
        "vw_pbi_affordability_hint",
        "vw_pbi_acs_rent_value"
    ]
    
    for view in views:
        check_view(view)
    
    # Дополнительная диагностика
    print("\n🔧 Дополнительная диагностика:")
    
    # Проверка подключения
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
        print("✓ Подключение к БД работает")
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")
    
    # Проверка схем
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT schema_name FROM information_schema.schemata"))
            schemas = [row[0] for row in result]
        print(f"✓ Схемы в БД: {', '.join(schemas)}")
    except Exception as e:
        print(f"❌ Ошибка проверки схем: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Проверка завершена")

if __name__ == "__main__":
    main()
