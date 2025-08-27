-- US Real Estate Analytics Database Schema
-- Production-ready schema for comprehensive real estate market analysis

-- Dimension Tables
CREATE TABLE IF NOT EXISTS dim_date (
    date DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_start DATE NOT NULL,
    month_end DATE NOT NULL,
    yyyymm VARCHAR(6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_geo (
    geo_key VARCHAR(20) PRIMARY KEY,
    level VARCHAR(10) CHECK (level IN ('US', 'STATE', 'MSA', 'COUNTY', 'ZIP')) NOT NULL,
    state_fips VARCHAR(2),
    county_fips VARCHAR(5),
    msa VARCHAR(10),
    zip VARCHAR(10),
    state_abbr VARCHAR(2),
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_metric (
    metric_key VARCHAR(50) PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    unit VARCHAR(20),
    freq VARCHAR(10),
    source VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact Tables
CREATE TABLE IF NOT EXISTS fact_metric (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    geo_key VARCHAR(20) NOT NULL,
    geo_level VARCHAR(10),
    metric VARCHAR(50) NOT NULL,
    value DECIMAL(15,4),
    source VARCHAR(50),
    unit VARCHAR(20),
    freq VARCHAR(10),
    vintage_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quality_flag BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, geo_key, metric)
);

CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    geo_key VARCHAR(20) NOT NULL,
    home_value DECIMAL(15,2),
    home_value_yoy DECIMAL(8,4),
    home_value_5y DECIMAL(8,4),
    mortgage_rate DECIMAL(6,4),
    mortgage_payment DECIMAL(10,2),
    payment_to_income_pct DECIMAL(8,4),
    value_to_income_ratio DECIMAL(8,4),
    median_income DECIMAL(10,2),
    median_rent DECIMAL(8,2),
    buy_vs_rent_diff DECIMAL(10,2),
    overvalued_pct DECIMAL(8,4),
    permits INTEGER,
    inventory INTEGER,
    days_on_market INTEGER,
    new_listings INTEGER,
    disaster_rate DECIMAL(8,4),
    divorce_share DECIMAL(8,4),
    unemployment_rate DECIMAL(6,4),
    political_share_dem DECIMAL(8,4),
    yield_curve_spread DECIMAL(6,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, geo_key)
);

CREATE TABLE IF NOT EXISTS model_outputs (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    geo_key VARCHAR(20) NOT NULL,
    target_metric VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,
    horizon INTEGER NOT NULL,
    y_hat DECIMAL(15,4),
    y_lo DECIMAL(15,4),
    y_hi DECIMAL(15,4),
    r2 DECIMAL(8,4),
    mae DECIMAL(15,4),
    rmse DECIMAL(15,4),
    version VARCHAR(20) DEFAULT '1.0',
    params_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    geo_key VARCHAR(20) NOT NULL,
    score_name VARCHAR(50) NOT NULL,
    score_value DECIMAL(8,4),
    version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, geo_key, score_name)
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_fact_metric_date_geo_metric ON fact_metric(date, geo_key, metric);
CREATE INDEX IF NOT EXISTS idx_fact_metric_geo_metric_date ON fact_metric(geo_key, metric, date);
CREATE INDEX IF NOT EXISTS idx_feature_store_date_geo ON feature_store(date, geo_key);
CREATE INDEX IF NOT EXISTS idx_model_outputs_date_geo ON model_outputs(date, geo_key);
CREATE INDEX IF NOT EXISTS idx_scores_geo_name_date ON scores(geo_key, score_name, date);

-- Views for Analytics
CREATE OR REPLACE VIEW vw_affordability_state AS
SELECT 
    fs.date,
    dg.state_abbr as state,
    fs.mortgage_payment,
    fs.payment_to_income_pct,
    fs.value_to_income_ratio,
    fs.buy_vs_rent_diff,
    fs.median_rent as fmr,
    fs.home_value,
    fs.mortgage_rate
FROM feature_store fs
JOIN dim_geo dg ON fs.geo_key = dg.geo_key
WHERE dg.level = 'STATE'
ORDER BY fs.date DESC, dg.state_abbr;

CREATE OR REPLACE VIEW vw_growth_state AS
SELECT 
    fs.date,
    dg.state_abbr as state,
    fs.home_value as hpi_level,
    fs.home_value_yoy as yoy,
    fs.home_value_5y as five_y
FROM feature_store fs
JOIN dim_geo dg ON fs.geo_key = dg.geo_key
WHERE dg.level = 'STATE'
ORDER BY fs.date DESC, dg.state_abbr;

CREATE OR REPLACE VIEW vw_supply_state AS
SELECT 
    fs.date,
    dg.state_abbr as state,
    fs.permits,
    fs.inventory,
    fs.days_on_market as dom,
    fs.new_listings
FROM feature_store fs
JOIN dim_geo dg ON fs.geo_key = dg.geo_key
WHERE dg.level = 'STATE'
ORDER BY fs.date DESC, dg.state_abbr;

CREATE OR REPLACE VIEW vw_risk_state AS
SELECT 
    fs.date,
    dg.state_abbr as state,
    fs.disaster_rate,
    fs.divorce_share,
    fs.unemployment_rate,
    fs.political_share_dem
FROM feature_store fs
JOIN dim_geo dg ON fs.geo_key = dg.geo_key
WHERE dg.level = 'STATE'
ORDER BY fs.date DESC, dg.state_abbr;

-- Insert initial data
INSERT INTO dim_metric (metric_key, metric_name, unit, freq, source, description) VALUES
('MORTGAGE30US', '30-Year Fixed Rate Mortgage Average', 'percent', 'weekly', 'FRED', '30-year fixed rate mortgage average'),
('CPIAUCSL', 'Consumer Price Index for All Urban Consumers', 'index', 'monthly', 'FRED', 'Consumer price index'),
('DGS10', '10-Year Treasury Constant Maturity Rate', 'percent', 'daily', 'FRED', '10-year treasury rate'),
('T10Y2Y', '10-Year minus 2-Year Treasury Constant Maturity Rate', 'percent', 'daily', 'FRED', 'Yield curve spread'),
('UNRATE', 'Unemployment Rate', 'percent', 'monthly', 'FRED', 'National unemployment rate'),
('HPI', 'House Price Index', 'index', 'quarterly', 'FHFA', 'House price index'),
('INCOME', 'Personal Income', 'dollars', 'monthly', 'BEA', 'Personal income'),
('RENT', 'Fair Market Rent', 'dollars', 'annual', 'HUD', 'Fair market rent'),
('PERMITS', 'Building Permits', 'count', 'monthly', 'Census', 'Building permits issued'),
('INVENTORY', 'Active Inventory', 'count', 'monthly', 'Redfin', 'Active inventory'),
('DOM', 'Days on Market', 'days', 'monthly', 'Redfin', 'Average days on market'),
('NEW_LISTINGS', 'New Listings', 'count', 'monthly', 'Redfin', 'New listings'),
('DISASTER_RATE', 'Disaster Rate', 'rate', 'annual', 'FEMA', 'Disaster declaration rate'),
('DIVORCE_SHARE', 'Divorce Share', 'percent', 'annual', 'Census', 'Share of divorced population'),
('POLITICAL_SHARE_DEM', 'Democratic Vote Share', 'percent', 'biennial', 'MIT', 'Democratic vote share')
ON CONFLICT (metric_key) DO NOTHING;
