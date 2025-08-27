#!/usr/bin/env python3
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'vw_pbi_%'"))
    print('Доступные витрины:')
    for row in result:
        print(f'- {row[0]}')
        
    # Проверим данные в одной из витрин
    try:
        result = conn.execute(text("SELECT COUNT(*) FROM vw_pbi_mortgage_rates"))
        count = result.fetchone()[0]
        print(f'\nКоличество записей в vw_pbi_mortgage_rates: {count}')
    except Exception as e:
        print(f'Ошибка при проверке данных: {e}')
