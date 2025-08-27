-- Reset database schema
-- Drop all tables and views

DROP VIEW IF EXISTS vw_affordability_state CASCADE;
DROP VIEW IF EXISTS vw_growth_state CASCADE;
DROP VIEW IF EXISTS vw_supply_state CASCADE;
DROP VIEW IF EXISTS vw_risk_state CASCADE;

DROP TABLE IF EXISTS scores CASCADE;
DROP TABLE IF EXISTS model_outputs CASCADE;
DROP TABLE IF EXISTS feature_store CASCADE;
DROP TABLE IF EXISTS fact_metric CASCADE;
DROP TABLE IF EXISTS dim_metric CASCADE;
DROP TABLE IF EXISTS dim_geo CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
