#!/usr/bin/env python3
"""
Command Line Interface for US Real Estate Analytics System
Provides commands for data ingestion, feature engineering, model training, and exports
"""

import click
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database import DatabaseManager
from ingest.fred import FREDIngester
from ingest.bls import BLSIngester
from ingest.bea import BEAIngester
from ingest.census import CensusIngester
from ingest.fhfa import FHFAIngester
from ingest.redfin import RedfinIngester
from ingest.fema import FEMAIngester
from ingest.mit_election import MITElectionIngester
from geo import create_geo_dimension_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cli.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """US Real Estate Analytics System CLI"""
    pass

@cli.command()
@click.option('--schema-file', default='schema.sql', help='SQL schema file path')
def init_db(schema_file):
    """Initialize database schema and dimension tables"""
    try:
        logger.info("Initializing database...")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize database manager
        db = DatabaseManager()
        
        # Test connection
        if not db.test_connection():
            logger.error("Database connection failed")
            return
        
        # Create schema
        if not db.create_schema(schema_file):
            logger.error("Schema creation failed")
            return
        
        # Create and insert geographic dimension data
        geo_data = create_geo_dimension_data()
        if not db.insert_dimension_data('dim_geo', geo_data):
            logger.error("Failed to insert geographic dimension data")
            return
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

@cli.command()
@click.option('--source', type=click.Choice([
    'fred', 'bls', 'bea', 'census', 'fhfa', 'redfin', 'fema', 'mit', 'all'
]), default='all', help='Data source to ingest')
def ingest(source):
    """Ingest data from external sources"""
    try:
        logger.info(f"Starting data ingestion for source: {source}")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        sources_to_run = []
        
        if source == 'all':
            sources_to_run = ['fred', 'bls', 'bea', 'census', 'fhfa', 'redfin', 'fema', 'mit']
        else:
            sources_to_run = [source]
        
        results = {}
        
        for src in sources_to_run:
            try:
                logger.info(f"Ingesting {src.upper()} data...")
                
                if src == 'fred':
                    ingester = FREDIngester()
                elif src == 'bls':
                    ingester = BLSIngester()
                elif src == 'bea':
                    ingester = BEAIngester()
                elif src == 'census':
                    ingester = CensusIngester()
                elif src == 'fhfa':
                    ingester = FHFAIngester()
                elif src == 'redfin':
                    ingester = RedfinIngester()
                elif src == 'fema':
                    ingester = FEMAIngester()
                elif src == 'mit':
                    ingester = MITElectionIngester()
                else:
                    logger.warning(f"Unknown source: {src}")
                    continue
                
                success = ingester.run()
                results[src] = success
                
                if success:
                    logger.info(f"{src.upper()} data ingestion completed successfully")
                else:
                    logger.error(f"{src.upper()} data ingestion failed")
                    
            except Exception as e:
                logger.error(f"{src.upper()} ingestion error: {e}")
                results[src] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Ingestion Summary: {successful}/{total} successful")
        
        for src, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"   {status} {src.upper()}")
        
        if successful == total:
            logger.info("All data ingestion completed successfully!")
        else:
            logger.warning("Some ingestions failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

@cli.command()
def build_features():
    """Build feature store from raw data"""
    try:
        logger.info("Building feature store...")
        
        # TODO: Implement feature engineering
        # This will calculate derived metrics like:
        # - Affordability metrics
        # - Growth rates (YoY, 5Y)
        # - Risk indicators
        # - Supply metrics
        
        logger.info("Feature store building completed")
        
    except Exception as e:
        logger.error(f"Feature building failed: {str(e)}")
        raise

@cli.command()
@click.option('--model', default='all', help='Model to train (arima, ols, logistic, all)')
def train_models(model):
    """Train predictive models"""
    try:
        logger.info(f"Training models: {model}")
        
        # TODO: Implement model training
        # This will train:
        # - ARIMA/SARIMAX models for time series forecasting
        # - OLS regression for growth prediction
        # - Logistic regression for risk assessment
        
        logger.info("Model training completed")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

@cli.command()
@click.option('--metric', default='home_value_yoy', help='Metric to visualize')
def render_map(metric):
    """Generate interactive map visualization"""
    try:
        logger.info(f"Rendering map for metric: {metric}")
        
        # TODO: Implement map generation
        # This will create:
        # - Plotly choropleth maps
        # - Interactive visualizations
        # - Export to HTML files
        
        logger.info("Map rendering completed")
        
    except Exception as e:
        logger.error(f"Map rendering failed: {str(e)}")
        raise

@cli.command()
def export_excel():
    """Export data to Excel dashboard"""
    try:
        logger.info("Exporting to Excel...")
        
        # TODO: Implement Excel export
        # This will create:
        # - Pivot tables and charts
        # - Multiple worksheets
        # - Interactive slicers
        
        logger.info("Excel export completed")
        
    except Exception as e:
        logger.error(f"Excel export failed: {str(e)}")
        raise

@cli.command()
def make_demo():
    """Run complete demo pipeline"""
    try:
        logger.info("Starting complete demo pipeline...")
        
        # Run all steps in sequence
        ctx = click.get_current_context()
        
        # 1. Initialize database
        logger.info("Step 1: Initializing database...")
        ctx.invoke(init_db)
        
        # 2. Ingest data
        logger.info("Step 2: Ingesting data...")
        ctx.invoke(ingest, source='all')
        
        # 3. Build features
        logger.info("Step 3: Building features...")
        ctx.invoke(build_features)
        
        # 4. Train models
        logger.info("Step 4: Training models...")
        ctx.invoke(train_models, model='all')
        
        # 5. Generate visualizations
        logger.info("Step 5: Generating visualizations...")
        ctx.invoke(render_map, metric='home_value_yoy')
        
        # 6. Export to Excel
        logger.info("Step 6: Exporting to Excel...")
        ctx.invoke(export_excel)
        
        logger.info("Demo pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo pipeline failed: {str(e)}")
        raise

@cli.command()
def status():
    """Show system status and data summary"""
    try:
        logger.info("Checking system status...")
        
        # Initialize database manager
        db = DatabaseManager()
        
        if not db.test_connection():
            logger.error("Database connection failed")
            return
        
        # Get table information
        tables = ['dim_geo', 'dim_metric', 'fact_metric', 'feature_store', 'model_outputs', 'scores']
        
        for table in tables:
            info = db.get_table_info(table)
            if info:
                logger.info(f"{table}: {info['row_count']} rows")
            else:
                logger.warning(f"Could not get info for table: {table}")
        
        logger.info("Status check completed")
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise

if __name__ == '__main__':
    cli()
