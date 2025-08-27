-- State-level metrics view for real estate analytics dashboard
-- Aggregates latest values from all state-level data sources

CREATE OR REPLACE VIEW vw_state_metrics AS
WITH latest_values AS (
  -- Get the most recent value for each metric per state
  SELECT 
    fm.geo_key,
    fm.metric,
    fm.value,
    fm.date,
    fm.source,
    fm.unit,
    ROW_NUMBER() OVER (PARTITION BY fm.geo_key, fm.metric ORDER BY fm.date DESC) as rn
  FROM fact_metric fm
  WHERE fm.geo_level = 'STATE'
    AND fm.quality_flag = TRUE
    AND fm.date >= CURRENT_DATE - INTERVAL '5 years'  -- Only consider data from last 5 years
),

state_pivot AS (
  -- Pivot key metrics into columns
  SELECT 
    lv.geo_key,
    -- Housing & Real Estate Metrics
    MAX(CASE WHEN lv.metric = 'MEDIAN_SALE_PRICE' THEN lv.value END) as median_sale_price,
    MAX(CASE WHEN lv.metric = 'MEDIAN_HOME_VALUE' THEN lv.value END) as median_home_value,
    MAX(CASE WHEN lv.metric = 'HPI_SA' THEN lv.value END) as house_price_index,
    MAX(CASE WHEN lv.metric = 'PRICE_GROWTH_YOY' THEN lv.value END) as price_growth_yoy,
    MAX(CASE WHEN lv.metric = 'HPI_GROWTH_YOY_SA' THEN lv.value END) as hpi_growth_yoy,
    MAX(CASE WHEN lv.metric = 'MEDIAN_DAYS_ON_MARKET' THEN lv.value END) as median_days_on_market,
    MAX(CASE WHEN lv.metric = 'INVENTORY' THEN lv.value END) as inventory,
    MAX(CASE WHEN lv.metric = 'MONTHS_OF_SUPPLY' THEN lv.value END) as months_of_supply,
    MAX(CASE WHEN lv.metric = 'HOMEOWNERSHIP_RATE' THEN lv.value END) as homeownership_rate,
    
    -- Economic Metrics
    MAX(CASE WHEN lv.metric = 'UNEMPLOYMENT_RATE' THEN lv.value END) as unemployment_rate,
    MAX(CASE WHEN lv.metric = 'PERSONAL_INCOME' THEN lv.value END) as personal_income,
    MAX(CASE WHEN lv.metric = 'INCOME_GROWTH_YOY' THEN lv.value END) as income_growth_yoy,
    MAX(CASE WHEN lv.metric = 'MEDIAN_HOUSEHOLD_INCOME' THEN lv.value END) as median_household_income,
    MAX(CASE WHEN lv.metric = 'GDP_TOTAL' THEN lv.value END) as gdp_total,
    
    -- Demographic Metrics
    MAX(CASE WHEN lv.metric = 'TOTAL_POPULATION' THEN lv.value END) as total_population,
    MAX(CASE WHEN lv.metric = 'EDUCATION_RATE' THEN lv.value END) as education_rate,
    MAX(CASE WHEN lv.metric = 'DIVORCE_RATE' THEN lv.value END) as divorce_rate,
    
    -- Risk Metrics
    MAX(CASE WHEN lv.metric = 'DISASTER_RATE_3YR_AVG' THEN lv.value END) as disaster_rate_3yr,
    MAX(CASE WHEN lv.metric = 'POLITICAL_COMPETITIVENESS' THEN lv.value END) as political_competitiveness,
    MAX(CASE WHEN lv.metric = 'PARTISAN_LEAN' THEN lv.value END) as partisan_lean,
    
    -- Get latest update timestamp
    MAX(lv.date) as last_updated
  FROM latest_values lv
  WHERE lv.rn = 1  -- Only latest values
  GROUP BY lv.geo_key
),

calculated_metrics AS (
  -- Calculate derived metrics for dashboard
  SELECT 
    sp.*,
    -- Geography info
    dg.state_abbr as state_code,
    dg.name as state_name,
    
    -- Calculated Housing Metrics
    COALESCE(sp.median_sale_price, sp.median_home_value) as home_value,
    
    -- GDP per capita
    CASE 
      WHEN sp.gdp_total IS NOT NULL AND sp.total_population IS NOT NULL AND sp.total_population > 0
      THEN sp.gdp_total / sp.total_population
      ELSE NULL
    END as gdp_per_capita,
    
    -- Value to Income Ratio
    CASE 
      WHEN sp.median_household_income IS NOT NULL AND sp.median_household_income > 0
        AND COALESCE(sp.median_sale_price, sp.median_home_value) IS NOT NULL
      THEN COALESCE(sp.median_sale_price, sp.median_home_value) / sp.median_household_income
      ELSE NULL
    END as value_to_income_ratio,
    
    -- Market temperature indicators
    CASE 
      WHEN sp.median_days_on_market IS NOT NULL
      THEN CASE 
        WHEN sp.median_days_on_market < 30 THEN 'Hot'
        WHEN sp.median_days_on_market < 60 THEN 'Warm'
        WHEN sp.median_days_on_market < 90 THEN 'Balanced'
        ELSE 'Cool'
      END
      ELSE 'Unknown'
    END as market_temperature,
    
    -- Risk score (normalized 0-100, higher = more risk)
    LEAST(100, GREATEST(0, 
      COALESCE(sp.disaster_rate_3yr * 10, 0) +  -- Disaster risk
      COALESCE(ABS(sp.partisan_lean) / 2, 0) +  -- Political risk
      COALESCE((sp.unemployment_rate - 3.5) * 5, 0) +  -- Economic risk
      COALESCE(GREATEST(0, sp.divorce_rate - 10), 0)  -- Social risk
    )) as risk_score
    
  FROM state_pivot sp
  LEFT JOIN dim_geo dg ON dg.geo_key = sp.geo_key AND dg.level = 'STATE'
  WHERE dg.geo_key IS NOT NULL  -- Only include states with geography data
)

-- Final selection with clean column names for dashboard
SELECT 
  cm.state_code as state,
  cm.state_name,
  cm.geo_key,
  
  -- Core Real Estate Metrics
  cm.home_value,
  cm.price_growth_yoy,
  cm.median_days_on_market,
  cm.inventory,
  cm.months_of_supply,
  cm.homeownership_rate,
  cm.market_temperature,
  
  -- Economic Metrics
  cm.unemployment_rate,
  cm.income_growth_yoy,
  cm.median_household_income,
  cm.gdp_per_capita,
  cm.value_to_income_ratio,
  
  -- Demographic Metrics
  cm.total_population,
  cm.education_rate,
  cm.divorce_rate,
  
  -- Risk Metrics
  cm.disaster_rate_3yr,
  cm.political_competitiveness,
  cm.risk_score,
  
  -- Meta
  cm.last_updated
  
FROM calculated_metrics cm
ORDER BY cm.state_code;

-- Create indexes on the underlying fact table for performance
CREATE INDEX IF NOT EXISTS idx_fact_metric_state_level 
ON fact_metric(geo_level, geo_key, metric, date DESC) 
WHERE geo_level = 'STATE';

CREATE INDEX IF NOT EXISTS idx_fact_metric_state_latest 
ON fact_metric(geo_key, metric, date DESC) 
WHERE geo_level = 'STATE' AND quality_flag = TRUE;

-- Add a comment describing the view
COMMENT ON VIEW vw_state_metrics IS 
'State-level real estate and economic metrics view. Provides latest values for all key metrics used in the dashboard, with calculated derived metrics like value-to-income ratio and risk scores. Updates automatically as new data is ingested into fact_metric.';
