#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DB_URL, pool_pre_ping=True)

def safe_query(query, default_df=None):
    """Безопасное выполнение запроса с обработкой ошибок"""
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Ошибка запроса: {e}")
        return default_df if default_df is not None else pd.DataFrame()

def test_data_loading():
    print("🧪 Тестирование загрузки данных...")
    
    # Тест 1: Ипотечные ставки
    print("\n1. Тест ипотечных ставок:")
    query = """
    SELECT date, mortgage_rate 
    FROM vw_pbi_mortgage_rates 
    WHERE date >= '2020-01-01' AND mortgage_rate IS NOT NULL
    ORDER BY date
    LIMIT 5
    """
    df = safe_query(query)
    print(f"   Результат: {len(df)} записей")
    if not df.empty:
        print(f"   Пример данных: {df.head().to_dict('records')}")
    
    # Тест 2: CPI
    print("\n2. Тест CPI:")
    query = """
    SELECT date, cpi 
    FROM vw_pbi_cpi 
    WHERE date >= '2020-01-01' AND cpi IS NOT NULL
    ORDER BY date
    LIMIT 5
    """
    df = safe_query(query)
    print(f"   Результат: {len(df)} записей")
    if not df.empty:
        print(f"   Пример данных: {df.head().to_dict('records')}")
    
    # Тест 3: Безработица
    print("\n3. Тест безработицы:")
    query = """
    SELECT date, unemployment_rate 
    FROM vw_pbi_unemployment 
    WHERE date >= '2020-01-01' AND unemployment_rate IS NOT NULL
    ORDER BY date
    LIMIT 5
    """
    df = safe_query(query)
    print(f"   Результат: {len(df)} записей")
    if not df.empty:
        print(f"   Пример данных: {df.head().to_dict('records')}")
    
    # Тест 4: ACS
    print("\n4. Тест ACS:")
    query = """
    SELECT year, median_gross_rent_usd, median_home_value_usd
    FROM vw_pbi_acs_rent_value 
    WHERE median_gross_rent_usd IS NOT NULL OR median_home_value_usd IS NOT NULL
    ORDER BY year
    LIMIT 5
    """
    df = safe_query(query)
    print(f"   Результат: {len(df)} записей")
    if not df.empty:
        print(f"   Пример данных: {df.head().to_dict('records')}")

if __name__ == "__main__":
    test_data_loading()
