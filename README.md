# 🏠 US Real Estate Analytics Dashboard

## 📊 Описание проекта

Интерактивная аналитическая панель для анализа рынка недвижимости США с использованием реальных данных из официальных источников.

## 🚀 Основные возможности

- **🗺️ Интерактивная карта США** с данными по штатам
- **📈 Регрессионный анализ** с множественными ML моделями
- **⚠️ Анализ рисков** для реэлтерских компаний
- **📊 Глобальный анализ рынка** с ключевыми метриками
- **⏰ Временные периоды** для исторического анализа

## 🛠️ Технический стек

- **Frontend:** Dash (Plotly), HTML/CSS
- **Backend:** Python 3.8+
- **База данных:** PostgreSQL (Neon)
- **ML:** scikit-learn (Linear Regression, Ridge, Random Forest)
- **API:** FRED, BLS, BEA, Census ACS

## 📁 Структура проекта

```
realestate_final_bundle_v2/
├── dashboard_advanced.py          # Основная панель
├── src/
│   ├── database.py               # Управление БД
│   └── ingest/                   # Ингестеры данных
│       ├── fred.py              # FRED API
│       ├── bls.py               # BLS LAUS API
│       ├── bea.py               # BEA API
│       ├── census.py            # Census ACS API
│       ├── fhfa.py              # FHFA HPI
│       ├── redfin.py            # Redfin Market Data
│       ├── fema.py              # FEMA Disasters
│       └── mit_election.py      # MIT Election Data
├── .env                          # API ключи
├── requirements.txt              # Зависимости
└── README.md                     # Документация
```

## 🔑 API Ключи

Проект использует следующие API:

- **FRED API:** `d7f56f7a50b44e780eb04b79cdcdd9b2`
- **BEA API:** `E6B8945F-F23B-4F40-B8ED-C02729F3B398`
- **BLS API:** `fc3c9ba2ac9546669cc41b719f4f1e51`
- **Census API:** `cb539edde53a3ffe7f21b74441860717446bd3b9`

## 🗄️ База данных

### Основные таблицы:
- `fact_metric` - фактовая таблица метрик
- `dim_geo` - географические измерения
- `dim_metric` - измерения метрик
- `dim_date` - временные измерения
- `vw_state_metrics` - представление для штатов

### Ключевые метрики:
- Цены на недвижимость (HPI, Median Sale Price)
- Доходы населения (Personal Income, Median Household Income)
- Безработица (Unemployment Rate)
- Население (Population)
- Образование (Education Levels)
- Владение недвижимостью (Homeownership Rate)

## 🚀 Установка и запуск

1. **Клонирование репозитория:**
```bash
git clone https://github.com/maksimkitikov/realestate.git
cd realestate
```

2. **Установка зависимостей:**
```bash
pip install -r requirements.txt
```

3. **Настройка переменных окружения:**
```bash
cp .env.example .env
# Отредактируйте .env файл с вашими API ключами
```

4. **Запуск панели:**
```bash
python dashboard_advanced.py
```

5. **Открытие в браузере:**
```
http://localhost:8050
```

## 📊 Аналитические возможности

### Регрессионный анализ
- **Модели:** Linear Regression, Ridge Regression, Random Forest
- **Метрики:** R², RMSE, MAE, Cross-validation
- **Признаки:** Population, Income, Unemployment, Education

### Анализ рисков
- **Экономические факторы:** Unemployment, Income Growth
- **Рыночные факторы:** Price Growth, Days on Market
- **Демографические факторы:** Population, Education
- **Политические факторы:** Election Competitiveness

### Временной анализ
- **Периоды:** 1 год, 2 года, 5 лет
- **Метрики:** YoY Growth, Historical Trends
- **Интерактивность:** Динамическое обновление карт

## 🔍 Источники данных

### Реальные API:
- **FRED (Federal Reserve):** Mortgage Rates, CPI, Unemployment, Treasury Yields
- **BLS (Bureau of Labor Statistics):** State Unemployment Data
- **BEA (Bureau of Economic Analysis):** Personal Income, GDP
- **Census ACS:** Population, Income, Education, Homeownership

### Планируемые источники:
- **FHFA:** House Price Index
- **Redfin:** Market Data (DOM, Months of Supply)
- **FEMA:** Disaster Declarations
- **MIT Election:** Political Competitiveness

## 📈 Метрики и измерения

### Географические измерения:
- **NATIONAL:** Общенациональные данные
- **STATE:** Данные по штатам США

### Временные измерения:
- **freq:** annual, monthly, quarterly
- **date:** Дата измерения
- **period:** Временной период для анализа

### Ключевые метрики:
- **price_growth_yoy:** Рост цен год к году
- **income_growth:** Рост доходов
- **unemployment_rate:** Уровень безработицы
- **population:** Население
- **median_household_income:** Медианный доход домохозяйств
- **homeownership_rate:** Уровень владения недвижимостью
- **education_bachelors_plus:** Доля с высшим образованием

## 🛡️ Безопасность

- API ключи хранятся в `.env` файле
- База данных защищена SSL соединением
- Все данные валидируются перед сохранением

## 📝 Лицензия

MIT License

## 👨‍💻 Автор

Максим Китиков - maksimkitikov@gmail.com

## 🔄 Версии

- **v2.0:** Полная интеграция с реальными API
- **v1.0:** Базовая функциональность с синтетическими данными

---

*Последнее обновление: Август 2025*
