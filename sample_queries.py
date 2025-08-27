#!/usr/bin/env python3
"""
Sample Queries для проверки витрин Power BI
Выводит по 5 строк из каждой витрины для проверки данных
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

def query_view(view_name: str, limit: int = 5) -> pd.DataFrame:
    """Выполнение запроса к витрине с лимитом"""
    try:
        query = f"SELECT * FROM {view_name} ORDER BY 1 DESC LIMIT {limit}"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"❌ Ошибка запроса к {view_name}: {e}")
        return pd.DataFrame()

def main():
    """Основная функция для проверки витрин"""
    print("🔍 Проверка витрин Power BI...")
    print("=" * 60)
    
    # Список витрин для проверки
    views = [
        "vw_pbi_mortgage_rates",
        "vw_pbi_cpi", 
        "vw_pbi_unemployment",
        "vw_pbi_rent_fmr",
        "vw_pbi_affordability_hint",
        "vw_pbi_acs_rent_value"
    ]
    
    for view in views:
        print(f"\n📊 {view}:")
        print("-" * 40)
        
        df = query_view(view)
        if not df.empty:
            print(f"Найдено записей: {len(df)}")
            print(df.to_string(index=False))
        else:
            print("Нет данных")
    
    print("\n" + "=" * 60)
    print("✅ Проверка витрин завершена")

if __name__ == "__main__":
    main()
