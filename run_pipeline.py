#!/usr/bin/env python3
"""
Real Estate Analytics ETL Pipeline v2
Оркестратор для сбора данных из FRED, HUD, BEA, BLS, Census и загрузки в DWH
"""

import os
import logging
import sys
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Конфигурация из переменных окружения
DB_URL = os.getenv("DATABASE_URL", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
HUD_API_KEY = os.getenv("HUD_API_KEY", "")
BEA_API_KEY = os.getenv("BEA_API_KEY", "")
BLS_API_KEY = os.getenv("BLS_API_KEY", "")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
REQUESTS_TIMEOUT_SECONDS = int(os.getenv("REQUESTS_TIMEOUT_SECONDS", "60"))

# Проверка обязательных переменных
if not DB_URL or not FRED_API_KEY:
    raise SystemExit("DATABASE_URL и FRED_API_KEY обязательны в .env")

# Создание подключения к БД
engine = create_engine(DB_URL, pool_pre_ping=True)

def ensure_core() -> None:
    """Создание основных схем БД"""
    ddl = """
    create schema if not exists stg;
    create schema if not exists dwh;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("✓ Основные схемы созданы")

def ensure_staging_tables() -> None:
    """Создание staging-таблиц для сырых данных"""
    ddl = """
    create table if not exists stg_fred_series(
        series_id varchar(64),
        obs_date date,
        value numeric,
        is_sa boolean,
        frequency varchar(8),
        source varchar(32) default 'FRED',
        load_ts timestamp default now(),
        primary key(series_id, obs_date)
    );

    create table if not exists stg_hud_fmr(
        geo_code varchar(16),
        geo_type varchar(16),
        year int,
        fmr_0 numeric, fmr_1 numeric, fmr_2 numeric, fmr_3 numeric, fmr_4 numeric,
        load_ts timestamp default now()
    );

    create table if not exists stg_hud_chas(
        geo_code varchar(16),
        year int,
        households_total int,
        cost_burden_30 int,
        cost_burden_50 int,
        severe_shortage int,
        load_ts timestamp default now()
    );

    create table if not exists stg_bea_gdp(
        series varchar(64),
        time_period varchar(16),
        value numeric,
        unit varchar(16),
        load_ts timestamp default now()
    );

    create table if not exists stg_bls_cpi(
        series_id varchar(32),
        year int,
        period varchar(4),
        value numeric,
        load_ts timestamp default now()
    );

    create table if not exists stg_census_acs(
        year int,
        geo varchar(32),
        var_code varchar(16),
        value numeric,
        load_ts timestamp default now()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("✓ Staging-таблицы созданы")

# ---------- FRED API ----------
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type((requests.RequestException, ValueError))
)
def fred_obs(series_id: str, api_key: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Получение наблюдений из FRED API с retry-логикой"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    p = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if params:
        p.update(params)
    
    r = requests.get(url, params=p, timeout=REQUESTS_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json().get("observations", [])

def ingest_fred(series_id: str, frequency: str, sa: bool) -> int:
    """Загрузка данных из FRED с обработкой ошибок frequency"""
    try:
        obs = fred_obs(series_id, FRED_API_KEY, {"frequency": frequency.lower()})
    except requests.HTTPError as e:
        if e.response.status_code == 400 and frequency != "d":
            logger.warning(f"FRED вернул 400 для {series_id} с frequency={frequency}, пробуем без frequency")
            obs = fred_obs(series_id, FRED_API_KEY)  # Без frequency
        else:
            raise
    
    if not obs:
        logger.warning(f"FRED вернул 0 наблюдений для {series_id}")
        return 0
    
    df = pd.DataFrame(obs).rename(columns={"date": "obs_date"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["is_sa"] = sa
    df["frequency"] = frequency
    df["series_id"] = series_id
    
    sql = text("""
        insert into stg_fred_series(series_id, obs_date, value, is_sa, frequency)
        values(:series_id, :obs_date, :value, :is_sa, :frequency)
        on conflict(series_id, obs_date) do update set
            value=excluded.value, is_sa=excluded.is_sa, frequency=excluded.frequency
    """)
    
    with engine.begin() as conn:
        for rec in df[["series_id", "obs_date", "value", "is_sa", "frequency"]].to_dict("records"):
            conn.execute(sql, rec)
    
    return len(df)

# ---------- HUD API ----------
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type(requests.RequestException)
)
def hud_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Получение данных из HUD API с Bearer-токеном"""
    if not HUD_API_KEY:
        logger.info(f"HUD_API_KEY пустой; пропускаем endpoint {endpoint}")
        return {"data": []}
    
    headers = {"Authorization": f"Bearer {HUD_API_KEY}"}
    url = f"https://www.huduser.gov/hudapi/public/{endpoint}"
    
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=REQUESTS_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"HUD API 404 для {endpoint}: {e}")
            return {"data": []}
        else:
            raise

def ingest_hud_fmr(year: int = 2024) -> int:
    """Загрузка данных FMR из HUD с fallback на предыдущие годы"""
    try:
        js = hud_get("fmr", {"year": year})
        rows = js.get("data") or js.get("results") or []
        
        if not rows:
            # Пробуем предыдущие годы
            for fallback_year in [year-1, year-2, year-3]:
                logger.warning(f"HUD FMR {year} не найден, пробуем {fallback_year}")
                js = hud_get("fmr", {"year": fallback_year})
                rows = js.get("data") or js.get("results") or []
                if rows:
                    break
            
            if not rows:
                logger.warning(f"HUD FMR данные не найдены для {year}-{year-3}, пропускаем")
                return 0
        
        df = pd.json_normalize(rows)
        ins = text("""
            insert into stg_hud_fmr(geo_code, geo_type, year, fmr_0, fmr_1, fmr_2, fmr_3, fmr_4)
            values(:geo_code, :geo_type, :year, :fmr_0, :fmr_1, :fmr_2, :fmr_3, :fmr_4)
            on conflict do nothing
        """)
        
        with engine.begin() as conn:
            for r in df.to_dict("records"):
                payload = {
                    "geo_code": str(r.get("geoid") or r.get("code") or r.get("fips") or ""),
                    "geo_type": str(r.get("geotype") or r.get("geo_type") or ""),
                    "year": int(r.get("year") or year),
                    "fmr_0": r.get("fmr0") or r.get("efficiency"),
                    "fmr_1": r.get("fmr1"),
                    "fmr_2": r.get("fmr2"),
                    "fmr_3": r.get("fmr_3"),
                    "fmr_4": r.get("fmr4"),
                }
                conn.execute(ins, payload)
        
        return len(df)
        
    except Exception as e:
        logger.error(f"Ошибка загрузки HUD FMR: {e}")
        return 0

def ingest_hud_chas(year: int = 2021) -> int:
    """Загрузка данных CHAS из HUD с fallback на предыдущие годы"""
    try:
        js = hud_get("chas", {"year": year})
        rows = js.get("data") or js.get("results") or []
        
        if not rows:
            # Пробуем предыдущие годы
            for fallback_year in [year-1, year-2, year-3]:
                logger.warning(f"HUD CHAS {year} не найден, пробуем {fallback_year}")
                js = hud_get("chas", {"year": fallback_year})
                rows = js.get("data") or js.get("results") or []
                if rows:
                    break
            
            if not rows:
                logger.warning(f"HUD CHAS данные не найдены для {year}-{year-3}, пропускаем")
                return 0
        
        df = pd.json_normalize(rows)
        ins = text("""
            insert into stg_hud_chas(geo_code, year, households_total, cost_burden_30, cost_burden_50, severe_shortage)
            values(:geo_code, :year, :households_total, :cost_burden_30, :cost_burden_50, :severe_shortage)
            on conflict do nothing
        """)
        
        with engine.begin() as conn:
            for r in df.to_dict("records"):
                payload = {
                    "geo_code": str(r.get("geoid") or r.get("fips") or ""),
                    "year": int(r.get("year") or year),
                    "households_total": r.get("households_total") or r.get("hh_total"),
                    "cost_burden_30": r.get("cost_burden_30") or r.get("hh_cb_30"),
                    "cost_burden_50": r.get("cost_burden_50") or r.get("hh_cb_50"),
                    "severe_shortage": r.get("severe_shortage") or r.get("shortage_severe"),
                }
                conn.execute(ins, payload)
        
        return len(df)
        
    except Exception as e:
        logger.error(f"Ошибка загрузки HUD CHAS: {e}")
        return 0

# ---------- BEA API ----------
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type(requests.RequestException)
)
def bea_get(dataset: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Получение данных из BEA API"""
    if not BEA_API_KEY:
        logger.info("BEA_API_KEY пустой; пропускаем BEA")
        return {}
    
    url = "https://apps.bea.gov/api/data"
    p = {
        "UserID": BEA_API_KEY, 
        "method": "GetData", 
        "datasetname": dataset, 
        "ResultFormat": "JSON"
    }
    p.update(params)
    
    r = requests.get(url, params=p, timeout=REQUESTS_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()

def ingest_bea_gdp() -> int:
    """Загрузка данных GDP из BEA NIPA"""
    try:
        js = bea_get("NIPA", {"TableName": "T10101", "Frequency": "Q"})
        series = (js.get("BEAAPI", {}) or {}).get("Results", {}).get("Data", [])
        
        if not series:
            logger.warning("BEA GDP данные не найдены")
            return 0
        
        df = pd.DataFrame(series)
        ins = text("""
            insert into stg_bea_gdp(series, time_period, value, unit)
            values(:series, :time_period, :value, :unit)
            on conflict do nothing
        """)
        
        with engine.begin() as conn:
            for r in df.to_dict("records"):
                # Очистка DataValue от запятых
                data_value = str(r.get("DataValue", "0")).replace(",", "") if r.get("DataValue") else "0"
                payload = {
                    "series": str(r.get("SeriesCode") or "GDP"),
                    "time_period": str(r.get("TimePeriod")),
                    "value": float(data_value) if data_value != "0" else None,
                    "unit": str(r.get("UnitOfMeasure") or ""),
                }
                conn.execute(ins, payload)
        
        return len(df)
        
    except Exception as e:
        logger.error(f"Ошибка загрузки BEA GDP: {e}")
        return 0

# ---------- BLS API ----------
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type(requests.RequestException)
)
def bls_post(series_ids: List[str], start_year: int = 2019, end_year: int = 2025) -> Dict[str, Any]:
    """POST-запрос к BLS API"""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": series_ids, 
        "startyear": str(start_year), 
        "endyear": str(end_year)
    }
    
    if BLS_API_KEY:
        payload["registrationkey"] = BLS_API_KEY
    
    r = requests.post(url, json=payload, timeout=REQUESTS_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()

def ingest_bls_cpi(series_id: str = "CUSR0000SA0", start_year: int = 2019, end_year: int = 2025) -> int:
    """Загрузка данных CPI из BLS"""
    try:
        js = bls_post([series_id], start_year, end_year)
        data = js.get("Results", {}).get("series", [])
        
        if not data:
            logger.warning("BLS CPI данные не найдены")
            return 0
        
        obs = data[0].get("data", [])
        df = pd.DataFrame(obs)
        ins = text("""
            insert into stg_bls_cpi(series_id, year, period, value)
            values(:series_id, :year, :period, :value)
            on conflict do nothing
        """)
        
        with engine.begin() as conn:
            for r in df.to_dict("records"):
                payload = {
                    "series_id": series_id, 
                    "year": int(r["year"]), 
                    "period": r["period"], 
                    "value": float(r["value"])
                }
                conn.execute(ins, payload)
        
        return len(df)
        
    except Exception as e:
        logger.error(f"Ошибка загрузки BLS CPI: {e}")
        return 0

# ---------- Census API ----------
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type(requests.RequestException)
)
def census_get(year: int, dataset: str, variables: List[str], geo: str = "us:1") -> List[List[str]]:
    """Получение данных из Census API"""
    if not CENSUS_API_KEY:
        logger.info("CENSUS_API_KEY пустой; пропускаем Census")
        return []
    
    base = f"https://api.census.gov/data/{year}/{dataset}"
    get_vars = ",".join(variables)
    params = {"get": get_vars, "for": geo, "key": CENSUS_API_KEY}
    
    r = requests.get(base, params=params, timeout=REQUESTS_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()  # Первая строка - заголовок

def ingest_census_acs(year: int = 2023) -> int:
    """Загрузка данных ACS из Census (медианная рента и стоимость жилья)"""
    try:
        rows = census_get(year, "acs/acs1", ["B25064_001E", "B25077_001E", "NAME"], "us:1")
        
        if not rows:
            logger.warning(f"Census ACS {year} данные не найдены")
            return 0
        
        header, *data = rows
        cols = {name: idx for idx, name in enumerate(header)}
        idx_rent = cols.get("B25064_001E")
        idx_value = cols.get("B25077_001E")
        idx_name = cols.get("NAME")
        idx_state = cols.get("us") if "us" in cols else None

        ins = text("""
            insert into stg_census_acs(year, geo, var_code, value)
            values(:year, :geo, :var_code, :value)
            on conflict do nothing
        """)
        
        with engine.begin() as conn:
            for row in data:
                geo = f"us:{row[idx_state]}" if idx_state is not None else "us:1"
                
                if idx_rent is not None:
                    try:
                        val = float(row[idx_rent])
                    except (ValueError, TypeError):
                        val = None
                    conn.execute(ins, {"year": year, "geo": geo, "var_code": "B25064_001E", "value": val})
                
                if idx_value is not None:
                    try:
                        val2 = float(row[idx_value])
                    except (ValueError, TypeError):
                        val2 = None
                    conn.execute(ins, {"year": year, "geo": geo, "var_code": "B25077_001E", "value": val2})
        
        return len(data)
        
    except Exception as e:
        logger.error(f"Ошибка загрузки Census ACS: {e}")
        return 0

# ---------- DWH helpers ----------
def seed_date_dimension() -> None:
    """Заполнение измерения дат с 1940 года для покрытия всех исторических данных FRED"""
    sql = """
    insert into dwh_dim_date(date_key, date_value, year, quarter, month, week, day, is_month_end, is_quarter_end)
    select
        cast(to_char(d, 'YYYYMMDD') as int),
        d,
        extract(year from d)::smallint,
        extract(quarter from d)::smallint,
        extract(month from d)::smallint,
        extract(week from d)::smallint,
        extract(day from d)::smallint,
        (d = (date_trunc('month', d) + interval '1 month - 1 day')::date),
        (d = (date_trunc('quarter', d) + interval '3 month - 1 day')::date)
    from generate_series(date '1940-01-01', date '2035-12-31', interval '1 day') as gs(d)
    on conflict (date_key) do nothing;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("✓ Измерение дат заполнено (1940-2035)")

def upsert_series_meta() -> None:
    """Обновление метаданных серий"""
    sql = """
    insert into dwh_dim_series(series_id, series_name, unit, seasonal, frequency, source) values
      ('MORTGAGE30US','30Y Fixed Mortgage Rate','percent','NSA','W','FRED'),
      ('CPIAUCSL','CPI All Urban Consumers','index','SA','M','FRED'),
      ('UNRATE','Unemployment Rate','percent','SA','M','FRED')
    on conflict (series_id) do nothing;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("✓ Метаданные серий обновлены")

def conform_mortgage_rates() -> None:
    """Конформирование данных по ипотечным ставкам с фильтрацией по датам"""
    sql = """
    insert into dwh_fact_mortgage_rates(date_key, series_key, rate)
    select 
        cast(to_char(obs_date, 'YYYYMMDD') as int),
        (select series_key from dwh_dim_series where series_id='MORTGAGE30US'),
        value::numeric
    from stg_fred_series
    where series_id='MORTGAGE30US'
    and obs_date >= '1940-01-01'
    and obs_date <= '2035-12-31'
    on conflict do nothing;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("✓ Факты по ипотечным ставкам загружены")

def conform_cpi_from_fred() -> None:
    """Конформирование CPI из FRED с фильтрацией по датам"""
    sql = """
    insert into dwh_fact_cpi(date_key, cpi)
    select 
        cast(to_char(obs_date,'YYYYMMDD') as int), 
        value::numeric
    from stg_fred_series
    where series_id='CPIAUCSL'
    and obs_date >= '1940-01-01'
    and obs_date <= '2035-12-31'
    on conflict do nothing;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("✓ Факты CPI загружены")

def conform_unemployment_from_fred() -> None:
    """Конформирование данных по безработице из FRED с фильтрацией по датам"""
    sql = """
    insert into dwh_fact_unemployment(date_key, unemp_rate)
    select 
        cast(to_char(obs_date,'YYYYMMDD') as int), 
        value::numeric
    from stg_fred_series
    where series_id='UNRATE'
    and obs_date >= '1940-01-01'
    and obs_date <= '2035-12-31'
    on conflict do nothing;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("✓ Факты по безработице загружены")

def main() -> None:
    """Основная функция ETL-пайплайна"""
    logger.info("🚀 Запуск Real Estate Analytics Pipeline v2")
    
    # Создание схем и таблиц
    ensure_core()
    ensure_staging_tables()
    
    # Загрузка данных из источников
    total_fred = 0
    total_fred += ingest_fred("MORTGAGE30US", "w", False)
    total_fred += ingest_fred("CPIAUCSL", "m", True)
    total_fred += ingest_fred("UNRATE", "m", True)
    logger.info(f"📊 FRED загружено записей: {total_fred}")
    
    fmr_count = ingest_hud_fmr(2024)
    chas_count = ingest_hud_chas(2021)
    logger.info(f"🏠 HUD FMR записей: {fmr_count} | HUD CHAS записей: {chas_count}")
    
    bea_count = ingest_bea_gdp()
    bls_count = ingest_bls_cpi()
    logger.info(f"📈 BEA GDP записей: {bea_count} | BLS CPI записей: {bls_count}")
    
    acs_count = ingest_census_acs(2023)
    logger.info(f"🏘️ Census ACS записей: {acs_count}")
    
    # Заполнение DWH
    seed_date_dimension()
    upsert_series_meta()
    conform_mortgage_rates()
    conform_cpi_from_fred()
    conform_unemployment_from_fred()
    
    # Итоговая сводка
    total_records = total_fred + fmr_count + chas_count + bea_count + bls_count + acs_count
    logger.info(f"✅ Pipeline finished successfully. Всего загружено записей: {total_records}")

if __name__ == "__main__":
    main()
