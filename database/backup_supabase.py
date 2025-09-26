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
        
        # Use pg_dump to create a complete backup
        cmd = [
            'pg_dump',
            DATABASE_URL,
            '--format=custom',  # Custom format for compression and selective restore
            '--verbose',
            '--no-owner',       # Don't include ownership commands
            '--no-privileges',  # Don't include privilege commands
            '--file', backup_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd[:2])} [DATABASE_URL] {' '.join(cmd[2:])}")
        
        # Run pg_dump
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Get file size
            file_size = os.path.getsize(backup_path)
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"Database backup completed successfully!")
            logger.info(f"Backup file: {backup_path}")
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            # Also create a SQL text backup for easier inspection
            sql_filename = f"supabase_backup_{timestamp}.sql"
            sql_path = os.path.join(BACKUP_DIR, sql_filename)
            
            sql_cmd = [
                'pg_dump',
                DATABASE_URL,
                '--format=plain',   # Plain SQL format
                '--verbose',
                '--no-owner',
                '--no-privileges',
                '--file', sql_path
            ]
            
            logger.info("Creating additional SQL text backup...")
            sql_result = subprocess.run(sql_cmd, capture_output=True, text=True, timeout=1800)
            
            if sql_result.returncode == 0:
                sql_size = os.path.getsize(sql_path)
                sql_size_mb = sql_size / (1024 * 1024)
                logger.info(f"SQL backup created: {sql_path} ({sql_size_mb:.2f} MB)")
            else:
                logger.warning(f"SQL backup failed: {sql_result.stderr}")
            
            return backup_path
            
        else:
            logger.error(f"pg_dump failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Database backup timed out after 1 hour")
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

def get_all_tables_and_views(client: Client) -> List[str]:
    """
    Get list of all tables and views in the database
    
    Args:
        client (Client): Supabase client instance
        
    Returns:
        List[str]: List of table and view names
    """
    try:
        # Try to get all tables using a more direct approach
        # First, try to get tables using a simple query on a known table
        tables_found = []
        
        for table_name in KNOWN_TABLES:
            try:
                # Try to query the table to see if it exists
                client.table(table_name).select('id').limit(1).execute()
                tables_found.append(table_name)
                logger.info(f"Found table: {table_name}")
            except Exception as e:
                logger.warning(f"Table {table_name} not found or not accessible: {str(e)}")
        
        if not tables_found:
            # If we can't find any tables, raise an exception
            raise Exception("No tables found in the database")
        
        logger.info(f"Found {len(tables_found)} tables")
        return tables_found
    except Exception as e:
        logger.error(f"Error getting tables and views: {str(e)}")
        # If all else fails, return the known tables
        logger.info("Using default known tables")
        return KNOWN_TABLES

def get_table_count(client: Client, table_name: str) -> int:
    """
    Get the count of records in a table
    
    Args:
        client (Client): Supabase client instance
        table_name (str): Name of the table
        
    Returns:
        int: Number of records in the table
    """
    try:
        # For Supabase, we need to use count query differently
        response = client.table(table_name).select('*', count="exact").limit(1).execute()
        return response.count if hasattr(response, 'count') else 0
    except Exception as e:
        logger.warning(f"Could not get count for table {table_name}: {str(e)}")
        return 0

def backup_table_chunked(client: Client, table_name: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[Dict]:
    """
    Backup a large table in chunks to avoid memory issues
    
    Args:
        client (Client): Supabase client instance
        table_name (str): Name of the table to backup
        chunk_size (int): Size of each chunk
        
    Returns:
        List[Dict]: List of chunk data
    """
    try:
        logger.info(f"Starting chunked backup of table: {table_name}")
        
        # Get total count
        total_count = get_table_count(client, table_name)
        if total_count == 0:
            logger.info(f"Table {table_name} is empty")
            return []
        
        logger.info(f"Table {table_name} has {total_count} records, backing up in chunks of {chunk_size}")
        
        chunks = []
        offset = 0
        chunk_index = 0
        
        while offset < total_count:
            try:
                response = client.table(table_name).select("*").range(offset, offset + chunk_size - 1).execute()
                data = response.data
                
                if not data:
                    break
                
                chunk_info = {
                    "chunk_index": chunk_index,
                    "record_count": len(data),
                    "offset": offset,
                    "data": data
                }
                
                chunks.append(chunk_info)
                logger.info(f"Backed up chunk {chunk_index} of {table_name} ({len(data)} records)")
                
                offset += chunk_size
                chunk_index += 1
                
            except Exception as e:
                logger.error(f"Error backing up chunk {chunk_index} of {table_name}: {str(e)}")
                raise
        
        logger.info(f"Completed chunked backup of {table_name} with {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in chunked backup of table {table_name}: {str(e)}")
        raise

def backup_table(client: Client, table_name: str, use_chunked: bool = False, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, Any]:
    """
    Backup a specific table from Supabase
    
    Args:
        client (Client): Supabase client instance
        table_name (str): Name of the table to backup
        use_chunked (bool): Whether to use chunked backup for large tables
        chunk_size (int): Size of each chunk for chunked backup
        
    Returns:
        Dict[str, Any]: Backup result containing data and metadata
    """
    try:
        logger.info(f"Starting backup of table/view: {table_name}")
        
        # For small tables or when not using chunked backup
        if not use_chunked:
            response = client.table(table_name).select("*").execute()
            data = response.data
            
            backup_metadata = {
                "table_name": table_name,
                "record_count": len(data),
                "backup_method": "full",
                "backup_timestamp": datetime.utcnow().isoformat(),
                "backup_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "backup_time": datetime.utcnow().strftime("%H:%M:%S")
            }
            
            logger.info(f"Successfully backed up {len(data)} records from {table_name}")
            return {
                "data": data,
                "metadata": backup_metadata
            }
        else:
            # Use chunked backup
            chunks = backup_table_chunked(client, table_name, chunk_size)
            total_records = sum(chunk["record_count"] for chunk in chunks)
            
            backup_metadata = {
                "table_name": table_name,
                "record_count": total_records,
                "chunk_count": len(chunks),
                "backup_method": "chunked",
                "chunk_size": chunk_size,
                "backup_timestamp": datetime.utcnow().isoformat(),
                "backup_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "backup_time": datetime.utcnow().strftime("%H:%M:%S")
            }
            
            logger.info(f"Successfully backed up {total_records} records from {table_name} in {len(chunks)} chunks")
            return {
                "chunks": chunks,
                "metadata": backup_metadata
            }
        
    except Exception as e:
        logger.error(f"Error backing up table {table_name}: {str(e)}")
        raise

def get_table_structure(client: Client, table_name: str) -> Dict[str, Any]:
    """
    Get the structure/schema of a table
    
    Args:
        client (Client): Supabase client instance
        table_name (str): Name of the table to get structure for
        
    Returns:
        Dict[str, Any]: Table structure information
    """
    try:
        logger.info(f"Getting structure for table: {table_name}")
        
        # Try to get detailed column information
        try:
            # This is a simplified approach. In practice, you might need to use a different method
            # depending on your Supabase setup
            response = client.table(table_name).select("*").limit(1).execute()
            if response.data:
                sample_record = response.data[0] if response.data else {}
                columns = [{"name": key, "type": type(value).__name__} for key, value in sample_record.items()]
            else:
                columns = []
        except:
            columns = []
        
        structure = {
            "table_name": table_name,
            "columns": columns
        }
        
        return structure
    except Exception as e:
        logger.warning(f"Error getting structure for table {table_name}: {str(e)}")
        # Return basic info if we can't get detailed structure
        return {
            "table_name": table_name,
            "columns": []
        }

def save_backup_to_file(backup_data: Dict[str, Any], filename: str = None) -> str:
    """
    Save backup data to a JSON file
    
    Args:
        backup_data (Dict[str, Any]): Data to save
        filename (str, optional): Custom filename for backup
        
    Returns:
        str: Path to the created backup file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"supabase_json_backup_{timestamp}.json"
    
    # Ensure backup directory exists
    ensure_backup_directory()
    
    filepath = os.path.join(BACKUP_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"JSON backup saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving backup to file: {str(e)}")
        raise

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
        # 1. Create complete database backup using pg_dump
        logger.info("Creating complete database backup with pg_dump...")
        pg_dump_path = create_pg_dump_backup()
        
        if pg_dump_path:
            backup_files['pg_dump'] = pg_dump_path
            logger.info("✓ Complete database backup created successfully")
        else:
            logger.warning("✗ pg_dump backup failed, continuing with JSON backup only")
        
        # 2. Create JSON data backup for inspection
        logger.info("Creating JSON data backup...")
        json_backup_path = backup_database_json()
        
        if json_backup_path:
            backup_files['json'] = json_backup_path
            logger.info("✓ JSON data backup created successfully")
        
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

def backup_database_json(use_chunked: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE, large_table_threshold: int = 10000) -> str:
    """
    Create JSON backup of database data for inspection
    
    Args:
        use_chunked (bool): Whether to use chunked backup for large tables
        chunk_size (int): Size of chunks for chunked backup
        large_table_threshold (int): Threshold to determine if a table is large and needs chunked backup
        
    Returns:
        str: Path to the created JSON backup file
    """
    logger.info("Starting JSON data backup")
    logger.info(f"Chunked backup: {use_chunked}, Chunk size: {chunk_size}, Large table threshold: {large_table_threshold}")
    
    try:
        # Create Supabase client
        client = create_supabase_client()
        
        # Get all tables and views
        tables_and_views = get_all_tables_and_views(client)
        
        # Collect backup data for all tables and views
        all_backup_data = {
            "backup_metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "tables_and_views_backed_up": [],
                "total_record_count": 0,
                "backup_method": "chunked" if use_chunked else "full",
                "chunk_size": chunk_size if use_chunked else None,
                "backup_type": "json_data_only"
            },
            "tables": {},
            "structures": {}
        }
        
        # Backup each table/view
        for table_name in tables_and_views:
            try:
                # Determine if we should use chunked backup for this table
                should_use_chunked = use_chunked
                if use_chunked:
                    table_count = get_table_count(client, table_name)
                    if table_count > 0 and table_count < large_table_threshold:
                        should_use_chunked = False
                        logger.info(f"Table {table_name} has {table_count} records, using full backup")
                    elif table_count >= large_table_threshold:
                        logger.info(f"Table {table_name} has {table_count} records, using chunked backup")
                
                # Backup table data
                backup_result = backup_table(client, table_name, should_use_chunked, chunk_size)
                all_backup_data["tables"][table_name] = backup_result
                all_backup_data["backup_metadata"]["tables_and_views_backed_up"].append(table_name)
                all_backup_data["backup_metadata"]["total_record_count"] += backup_result["metadata"]["record_count"]
                
                # Get table structure
                structure = get_table_structure(client, table_name)
                all_backup_data["structures"][table_name] = structure
                
            except Exception as e:
                logger.warning(f"Failed to backup table/view {table_name}: {str(e)}")
                # Continue with other tables/views
        
        # Save to file
        backup_filepath = save_backup_to_file(all_backup_data)
        
        logger.info(f"JSON data backup completed successfully. Total records: {all_backup_data['backup_metadata']['total_record_count']}")
        return backup_filepath
        
    except Exception as e:
        logger.error(f"JSON database backup failed: {str(e)}")
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
            logger.info("
RESTORE INSTRUCTIONS:")
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
            
            print(f"
✓ Backup completed successfully!")
            print(f"✓ Created {len(backup_files)} backup files in {BACKUP_DIR}/")
            
        else:
            logger.error("No backup files were created!")
            print("✗ Backup failed - no files were created")
            exit(1)
            
    except Exception as e:
        logger.error(f"Backup process failed: {str(e)}")
        print(f"✗ Error: {str(e)}")
        print("
Please check:")
        print("- SUPABASE_URL and SUPABASE_KEY environment variables are set correctly")
        print("- DATABASE_URL is set for complete pg_dump backup")
        print("- Network connectivity to Supabase")
        print("- PostgreSQL client tools are installed (for pg_dump)")
        exit(1)