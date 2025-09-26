#!/usr/bin/env python3
"""
Supabase Database Backup Script
Creates complete database backups using pg_dump for full recovery capability
Supports both PostgreSQL native backup and JSON data export
"""

import os
import json
import logging
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from supabase import create_client, Client

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is not installed, continue without it

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supabase_backup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Backup configuration
BACKUP_DIR = "database/backup"
DEFAULT_CHUNK_SIZE = 1000

# Known tables in the database
KNOWN_TABLES = [
    'properties', 
    'property_history', 
    'scraping_progress',
    'real_estate',
    'property_status'
]

def create_supabase_client() -> Client:
    """
    Create and return a Supabase client instance
    
    Returns:
        Client: Supabase client instance
        
    Raises:
        ValueError: If Supabase credentials are not set
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def ensure_backup_directory():
    """Ensure backup directory exists"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logger.info(f"Created backup directory: {BACKUP_DIR}")

def update_backup_status(status: str, message: str = ""):
    """Update backup status in scraping_progress table"""
    try:
        supabase = create_supabase_client()
        
        # Use ID 7 for database backup task
        data = {
            'status': status,
            'updated_at': 'now()'
        }
        
        if message:
            data['last_processed_id'] = message
        
        # Try to update existing record
        response = supabase.table('scraping_progress').update(data).eq('id', 7).execute()
        
        if not response.data:
            # If no record exists, create one
            data['id'] = 7
            data['batch_size'] = 1
            supabase.table('scraping_progress').insert(data).execute()
            
        logger.info(f"Updated backup status to: {status}")
        
    except Exception as e:
        logger.warning(f"Could not update backup status: {e}")

def create_pg_dump_backup() -> Optional[str]:
    """
    Create a complete database backup using pg_dump
    
    Returns:
        str: Path to the backup file if successful, None otherwise
    """
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is not set")
        return None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"supabase_complete_backup_{timestamp}.dump"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        logger.info("Starting complete database backup with pg_dump...")
        
        # Parse DATABASE_URL to get connection parameters
        import urllib.parse
        parsed = urllib.parse.urlparse(DATABASE_URL)
        
        # For Supabase, try different connection approaches
        connection_attempts = []
        
        # Attempt 1: Standard connection
        if parsed.port:
            connection_attempts.append({
                'host': parsed.hostname,
                'port': str(parsed.port),
                'user': parsed.username,
                'password': parsed.password,
                'dbname': parsed.path.lstrip('/')
            })
        
        # Attempt 2: Supabase pooler connection (port 6543)
        if parsed.hostname and 'supabase.co' in parsed.hostname:
            connection_attempts.append({
                'host': parsed.hostname,
                'port': '6543',
                'user': parsed.username,
                'password': parsed.password,
                'dbname': parsed.path.lstrip('/')
            })
        
        # Attempt 3: Direct connection with SSL
        connection_attempts.append({
            'host': parsed.hostname,
            'port': str(parsed.port) if parsed.port else '5432',
            'user': parsed.username,
            'password': parsed.password,
            'dbname': parsed.path.lstrip('/'),
            'sslmode': 'require'
        })
        
        for i, conn_params in enumerate(connection_attempts, 1):
            logger.info(f"Attempting pg_dump connection #{i} to {conn_params['host']}:{conn_params['port']}")
            
            # Build pg_dump command with connection parameters
            cmd = ['pg_dump']
            
            # Add connection parameters
            cmd.extend(['-h', conn_params['host']])
            cmd.extend(['-p', conn_params['port']])
            cmd.extend(['-U', conn_params['user']])
            cmd.extend(['-d', conn_params['dbname']])
            
            # Add SSL mode if specified
            if 'sslmode' in conn_params:
                cmd.extend(['--set', f"sslmode={conn_params['sslmode']}"])
            
            # Add backup options
            cmd.extend([
                '--format=custom',  # Custom format for compression and selective restore
                '--verbose',
                '--no-owner',       # Don't include ownership commands
                '--no-privileges',  # Don't include privilege commands
                '--file', backup_path
            ])
            
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = conn_params['password']
            
            logger.info(f"Running pg_dump with host={conn_params['host']}, port={conn_params['port']}")
            
            try:
                # Run pg_dump
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                    env=env
                )
                
                if result.returncode == 0:
                    # Success! Get file size
                    file_size = os.path.getsize(backup_path)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    logger.info(f"✅ Database backup completed successfully!")
                    logger.info(f"Backup file: {backup_path}")
                    logger.info(f"File size: {file_size_mb:.2f} MB")
                    
                    # Also create a SQL text backup for easier inspection
                    sql_filename = f"supabase_backup_{timestamp}.sql"
                    sql_path = os.path.join(BACKUP_DIR, sql_filename)
                    
                    sql_cmd = cmd.copy()
                    sql_cmd[sql_cmd.index('--format=custom')] = '--format=plain'
                    sql_cmd[sql_cmd.index(backup_path)] = sql_path
                    
                    logger.info("Creating additional SQL text backup...")
                    sql_result = subprocess.run(sql_cmd, capture_output=True, text=True, timeout=1800, env=env)
                    
                    if sql_result.returncode == 0:
                        sql_size = os.path.getsize(sql_path)
                        sql_size_mb = sql_size / (1024 * 1024)
                        logger.info(f"SQL backup created: {sql_path} ({sql_size_mb:.2f} MB)")
                    else:
                        logger.warning(f"SQL backup failed: {sql_result.stderr}")
                    
                    return backup_path
                    
                else:
                    logger.warning(f"pg_dump attempt #{i} failed with return code {result.returncode}")
                    logger.warning(f"Error output: {result.stderr}")
                    
                    # If this is not the last attempt, continue to next
                    if i < len(connection_attempts):
                        continue
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"pg_dump attempt #{i} timed out")
                if i < len(connection_attempts):
                    continue
        
        # All attempts failed
        logger.error("All pg_dump connection attempts failed")
        logger.info("This is common with Supabase free tier or network restrictions")
        logger.info("JSON backup will still be created using Supabase API")
        return None
            
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return None

def create_metadata_file(backup_path: str, json_backup_path: str = None):
    """Create a metadata file with backup information"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = f"backup_metadata_{timestamp}.json"
        metadata_path = os.path.join(BACKUP_DIR, metadata_filename)
        
        metadata = {
            "backup_timestamp": datetime.utcnow().isoformat(),
            "backup_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "backup_time": datetime.utcnow().strftime("%H:%M:%S"),
            "backup_type": "complete_database",
            "pg_dump_file": os.path.basename(backup_path) if backup_path else None,
            "json_backup_file": os.path.basename(json_backup_path) if json_backup_path else None,
            "database_url_provided": bool(DATABASE_URL),
            "supabase_url": SUPABASE_URL,
            "backup_method": "pg_dump + json_export",
            "restore_instructions": {
                "pg_dump_restore": "pg_restore -d TARGET_DATABASE_URL backup_file.dump",
                "sql_restore": "psql TARGET_DATABASE_URL < backup_file.sql",
                "json_restore": "Use custom script to import JSON data"
            }
        }
        
        # Add file sizes if files exist
        if backup_path and os.path.exists(backup_path):
            metadata["pg_dump_file_size_mb"] = os.path.getsize(backup_path) / (1024 * 1024)
        
        if json_backup_path and os.path.exists(json_backup_path):
            metadata["json_file_size_mb"] = os.path.getsize(json_backup_path) / (1024 * 1024)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Backup metadata saved to: {metadata_path}")
        return metadata_path
        
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")
        return None

def backup_table_data(table_name: str) -> Optional[str]:
    """
    Backup a single table's data to JSON format using Supabase API
    
    Args:
        table_name: Name of the table to backup
        
    Returns:
        str: Path to the JSON backup file if successful, None otherwise
    """
    try:
        supabase = create_supabase_client()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Backing up table: {table_name}")
        
        # Get all data from the table
        response = supabase.table(table_name).select("*").execute()
        
        if response.data:
            # Save to JSON file
            json_filename = f"{table_name}_backup_{timestamp}.json"
            json_path = os.path.join(BACKUP_DIR, json_filename)
            
            backup_data = {
                "table_name": table_name,
                "backup_timestamp": datetime.utcnow().isoformat(),
                "record_count": len(response.data),
                "data": response.data
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str, ensure_ascii=False)
            
            file_size = os.path.getsize(json_path)
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"✓ Table {table_name}: {len(response.data)} records, {file_size_mb:.2f} MB")
            return json_path
        else:
            logger.warning(f"Table {table_name} is empty or could not be accessed")
            return None
            
    except Exception as e:
        logger.error(f"Error backing up table {table_name}: {e}")
        return None

def create_json_backup() -> Optional[str]:
    """
    Create JSON backup of all known tables using Supabase API
    
    Returns:
        str: Path to the combined JSON backup file if successful, None otherwise
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"supabase_data_backup_{timestamp}.json"
        combined_path = os.path.join(BACKUP_DIR, combined_filename)
        
        logger.info("Creating JSON data backup using Supabase API...")
        
        supabase = create_supabase_client()
        all_backup_data = {
            "backup_timestamp": datetime.utcnow().isoformat(),
            "backup_type": "json_api_export",
            "tables": {}
        }
        
        total_records = 0
        successful_tables = 0
        
        for table_name in KNOWN_TABLES:
            try:
                logger.info(f"Backing up table: {table_name}")
                response = supabase.table(table_name).select("*").execute()
                
                if response.data:
                    all_backup_data["tables"][table_name] = {
                        "record_count": len(response.data),
                        "data": response.data
                    }
                    total_records += len(response.data)
                    successful_tables += 1
                    logger.info(f"✓ {table_name}: {len(response.data)} records")
                else:
                    logger.warning(f"Table {table_name} is empty or inaccessible")
                    all_backup_data["tables"][table_name] = {
                        "record_count": 0,
                        "data": [],
                        "note": "Empty or inaccessible"
                    }
                    
            except Exception as e:
                logger.error(f"Error backing up table {table_name}: {e}")
                all_backup_data["tables"][table_name] = {
                    "error": str(e),
                    "record_count": 0,
                    "data": []
                }
        
        # Add summary
        all_backup_data["summary"] = {
            "total_tables": len(KNOWN_TABLES),
            "successful_tables": successful_tables,
            "total_records": total_records
        }
        
        # Save combined backup
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_backup_data, f, indent=2, default=str, ensure_ascii=False)
        
        file_size = os.path.getsize(combined_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"✓ JSON backup completed: {total_records} total records, {file_size_mb:.2f} MB")
        return combined_path
        
    except Exception as e:
        logger.error(f"Error creating JSON backup: {e}")
        return None

def backup_database_complete() -> Dict[str, str]:
    """
    Main function to create complete Supabase database backup
    Creates both pg_dump backup (for full recovery) and JSON backup (for data inspection)
    
    Returns:
        Dict[str, str]: Dictionary with paths to created backup files
    """
    logger.info("Starting complete Supabase database backup")
    
    # Ensure backup directory exists
    ensure_backup_directory()
    
    # Update status to running
    update_backup_status('running', 'Starting database backup')
    
    backup_files = {}
    
    try:
        # 1. Create complete database backup using pg_dump (if DATABASE_URL is available)
        if DATABASE_URL:
            logger.info("Creating complete database backup with pg_dump...")
            pg_dump_path = create_pg_dump_backup()
            
            if pg_dump_path:
                backup_files['pg_dump'] = pg_dump_path
                logger.info("✓ Complete database backup created successfully")
            else:
                logger.warning("✗ pg_dump backup failed")
        else:
            logger.info("DATABASE_URL not available, skipping pg_dump backup")
        
        # 2. Create JSON data backup using Supabase API (always available)
        logger.info("Creating JSON data backup using Supabase API...")
        json_backup_path = create_json_backup()
        
        if json_backup_path:
            backup_files['json'] = json_backup_path
            logger.info("✓ JSON data backup created successfully")
        else:
            logger.warning("✗ JSON data backup failed")
        
        # 3. Create metadata file
        metadata_path = create_metadata_file(
            backup_files.get('pg_dump'), 
            backup_files.get('json')
        )
        
        if metadata_path:
            backup_files['metadata'] = metadata_path
        
        # Update status to complete
        if backup_files:
            update_backup_status('complete', f"Backup completed with {len(backup_files)} files")
            logger.info(f"✓ Complete database backup finished successfully!")
            logger.info(f"Created {len(backup_files)} backup files:")
            for backup_type, path in backup_files.items():
                logger.info(f"  - {backup_type}: {path}")
        else:
            update_backup_status('idle', 'Backup failed - no files created')
            logger.error("✗ No backup files were created")
        
        return backup_files
        
    except Exception as e:
        logger.error(f"Complete database backup failed: {str(e)}")
        update_backup_status('idle', f'Backup failed: {str(e)}')
        raise

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE SUPABASE DATABASE BACKUP")
        logger.info("=" * 60)
        
        # Check required environment variables
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        if not DATABASE_URL:
            logger.warning("DATABASE_URL not set - pg_dump backup will be skipped")
            logger.info("Only JSON data backup will be created")
        
        # Create complete backup (pg_dump + JSON)
        backup_files = backup_database_complete()
        
        if backup_files:
            logger.info("=" * 60)
            logger.info("BACKUP COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("Created backup files:")
            
            for backup_type, filepath in backup_files.items():
                logger.info(f"  {backup_type.upper()}: {filepath}")
            
            # Print restore instructions
            logger.info("\nRESTORE INSTRUCTIONS:")
            logger.info("-" * 40)
            
            if 'pg_dump' in backup_files:
                logger.info("To restore complete database:")
                logger.info(f"  pg_restore -d TARGET_DATABASE_URL {backup_files['pg_dump']}")
                logger.info("  OR")
                sql_file = backup_files['pg_dump'].replace('.dump', '.sql')
                if os.path.exists(sql_file):
                    logger.info(f"  psql TARGET_DATABASE_URL < {sql_file}")
            
            if 'json' in backup_files:
                logger.info("JSON backup contains data for inspection/analysis")
                logger.info("Use custom scripts to import specific tables if needed")
            
            print(f"\n✓ Backup completed successfully!")
            print(f"✓ Created {len(backup_files)} backup files in {BACKUP_DIR}/")
            
        else:
            logger.error("No backup files were created!")
            print("✗ Backup failed - no files were created")
            exit(1)
            
    except Exception as e:
        logger.error(f"Backup process failed: {str(e)}")
        print(f"✗ Error: {str(e)}")
        print("\nPlease check:")
        print("- SUPABASE_URL and SUPABASE_KEY environment variables are set correctly")
        print("- DATABASE_URL is set for complete pg_dump backup")
        print("- Network connectivity to Supabase")
        print("- PostgreSQL client tools are installed (for pg_dump)")
        exit(1)