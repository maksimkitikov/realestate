"""
Database connection and operations module
Handles database connections, schema creation, and data operations
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for real estate analytics system"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def create_schema(self, schema_file: str = "schema.sql") -> bool:
        """Create database schema from SQL file"""
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            with self.engine.connect() as conn:
                # Split by semicolon and execute each statement
                statements = schema_sql.split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        conn.execute(text(statement))
                conn.commit()
            
            logger.info("Database schema created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            return False
    
    def insert_dimension_data(self, table_name: str, data: pd.DataFrame) -> bool:
        """Insert dimension data with conflict handling"""
        try:
            data.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(data)} records into {table_name}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return False
    
    def upsert_fact_data(self, table_name: str, data: pd.DataFrame, unique_columns: List[str]) -> bool:
        """Upsert fact data with conflict resolution"""
        try:
            # Create temporary table
            temp_table_name = f"temp_{table_name}"
            data.to_sql(temp_table_name, self.engine, if_exists='replace', index=False)
            
            # Build upsert query
            columns = list(data.columns)
            update_columns = [col for col in columns if col not in unique_columns]
            
            update_set = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])
            unique_constraint = ", ".join(unique_columns)
            
            # Map metric_key to metric if needed
            if 'metric_key' in unique_columns:
                unique_constraint = unique_constraint.replace('metric_key', 'metric')
            
            upsert_sql = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            SELECT {', '.join(columns)} FROM {temp_table_name}
            ON CONFLICT ({unique_constraint})
            DO UPDATE SET {update_set}
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(upsert_sql))
                conn.commit()
            
            # Drop temporary table
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {temp_table_name}"))
                conn.commit()
            
            logger.info(f"Upserted {len(data)} records into {table_name}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error upserting into {table_name}: {str(e)}")
            return False
    
    def query_data(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        try:
            df = pd.read_sql(sql, self.engine, params=params)
            return df
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_date(self, table_name: str, date_column: str = 'date') -> Optional[str]:
        """Get latest date from table"""
        try:
            sql = f"SELECT MAX({date_column}) as latest_date FROM {table_name}"
            df = self.query_data(sql)
            if not df.empty and not df['latest_date'].isna().all():
                return df['latest_date'].iloc[0].strftime('%Y-%m-%d')
            return None
        except Exception as e:
            logger.error(f"Error getting latest date from {table_name}: {str(e)}")
            return None
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information"""
        try:
            sql = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            df = self.query_data(sql)
            return {
                'columns': df.to_dict('records'),
                'row_count': self.get_row_count(table_name)
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return {}
    
    def get_row_count(self, table_name: str) -> int:
        """Get row count for table"""
        try:
            sql = f"SELECT COUNT(*) as count FROM {table_name}"
            df = self.query_data(sql)
            return df['count'].iloc[0] if not df.empty else 0
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {str(e)}")
            return 0
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")
