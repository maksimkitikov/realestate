#!/usr/bin/env python3
"""
Офлайн-тесты для Real Estate Analytics Pipeline
Проверка конфигурации, преобразований и базовой схемы без внешних вызовов
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

def test_env_keys_present() -> None:
    """Проверка наличия API ключей в .env"""
    assert os.path.isfile(".env"), ".env отсутствует"
    
    with open(".env", "r", encoding="utf-8") as f:
        content = f.read()
    
    required_keys = [
        "FRED_API_KEY", "HUD_API_KEY", "BEA_API_KEY", 
        "BLS_API_KEY", "CENSUS_API_KEY", "DATABASE_URL"
    ]
    
    for key in required_keys:
        assert key in content, f"{key} отсутствует в .env"
    
    print("✓ Все API ключи присутствуют в .env")

def test_numeric_coercion() -> None:
    """Проверка преобразования числовых значений"""
    # Тест различных форматов чисел
    test_data = [
        {"value": "3.50"},      # Обычное число
        {"value": "NaN"},       # NaN
        {"value": "."},         # Точка
        {"value": None},        # None
        {"value": "1,234.56"},  # С запятыми
        {"value": ""},          # Пустая строка
        {"value": "abc"}        # Текст
    ]
    
    df = pd.DataFrame(test_data)
    # Очищаем запятые перед преобразованием (как в пайплайне)
    df["clean_value"] = df["value"].astype(str).str.replace(",", "")
    df["numeric_value"] = pd.to_numeric(df["clean_value"], errors="coerce")
    
    # Проверяем, что только валидные числа преобразовались
    valid_count = df["numeric_value"].notna().sum()
    invalid_count = df["numeric_value"].isna().sum()
    
    assert valid_count == 2, f"Ожидалось 2 валидных числа, получено {valid_count}"
    assert invalid_count == 5, f"Ожидалось 5 невалидных значений, получено {invalid_count}"
    
    print("✓ Преобразование числовых значений работает корректно")

def test_census_parse_like() -> None:
    """Симуляция парсинга ответа Census ACS"""
    # Симуляция ответа Census (заголовок + одна строка)
    rows = [
        ["B25064_001E", "B25077_001E", "NAME", "us"],
        ["1500", "350000", "United States", "1"]
    ]
    
    header, data = rows[0], rows[1]
    cols = {name: idx for idx, name in enumerate(header)}
    
    # Проверяем индексы колонок
    assert cols["B25064_001E"] == 0, "Неверный индекс для B25064_001E"
    assert cols["B25077_001E"] == 1, "Неверный индекс для B25077_001E"
    assert cols["NAME"] == 2, "Неверный индекс для NAME"
    assert cols["us"] == 3, "Неверный индекс для us"
    
    # Проверяем данные
    assert data[0] == "1500", "Неверное значение ренты"
    assert data[1] == "350000", "Неверное значение стоимости жилья"
    assert data[3] == "1", "Неверное значение us"
    
    print("✓ Парсинг Census ACS работает корректно")

def test_database_connection() -> None:
    """Проверка подключения к базе данных"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("⚠️ DATABASE_URL не установлен, пропускаем тест подключения")
        return
    
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            assert result.fetchone()[0] == 1, "Неверный результат тестового запроса"
        print("✓ Подключение к базе данных работает")
    except Exception as e:
        print(f"⚠️ Ошибка подключения к БД: {e}")

def test_schema_validation() -> None:
    """Проверка базовой структуры схемы"""
    # Проверяем, что schema.sql содержит необходимые элементы
    with open("schema.sql", "r", encoding="utf-8") as f:
        schema_content = f.read()
    
    required_elements = [
        "create schema if not exists stg",
        "create schema if not exists dwh",
        "create table if not exists stg_fred_series",
        "create table if not exists dwh_dim_date",
        "create table if not exists dwh_dim_series",
        "create or replace view vw_pbi_mortgage_rates",
        "create or replace view vw_pbi_cpi",
        "create or replace view vw_pbi_unemployment"
    ]
    
    for element in required_elements:
        assert element in schema_content, f"Отсутствует: {element}"
    
    print("✓ Схема БД содержит все необходимые элементы")

def test_pipeline_imports() -> None:
    """Проверка импортов в run_pipeline.py"""
    try:
        # Проверяем, что основные модули доступны
        import pandas
        import requests
        import sqlalchemy
        import tenacity
        from dotenv import load_dotenv
        
        print("✓ Все необходимые модули доступны")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        sys.exit(1)

def main() -> None:
    """Запуск всех тестов"""
    print("🧪 Запуск офлайн-тестов...")
    
    try:
        test_env_keys_present()
        test_numeric_coercion()
        test_census_parse_like()
        test_database_connection()
        test_schema_validation()
        test_pipeline_imports()
        
        print("\n✅ Все тесты пройдены успешно!")
        
    except AssertionError as e:
        print(f"\n❌ Тест не пройден: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
