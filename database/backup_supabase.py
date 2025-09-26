#!/usr/bin/env python3
"""
Supabase Database Backup Script
Backs up data from Supabase tables to local storage
Supports both full backup and chunked backup for large databases
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
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

# Get Supabase configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Default chunk size for large table backup
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
        filename = f"supabase_backup_{timestamp}.json"
    
    # Ensure backup directory exists
    backup_dir = "backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    filepath = os.path.join(backup_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"Backup saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving backup to file: {str(e)}")
        raise

def backup_database(use_chunked: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE, large_table_threshold: int = 10000) -> str:
    """
    Main function to backup Supabase database
    Backs up all tables, views and their structures
    
    Args:
        use_chunked (bool): Whether to use chunked backup for large tables
        chunk_size (int): Size of chunks for chunked backup
        large_table_threshold (int): Threshold to determine if a table is large and needs chunked backup
        
    Returns:
        str: Path to the created backup file
    """
    logger.info("Starting Supabase database backup")
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
                "chunk_size": chunk_size if use_chunked else None
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
        
        logger.info(f"Database backup completed successfully. Total records: {all_backup_data['backup_metadata']['total_record_count']}")
        return backup_filepath
        
    except Exception as e:
        logger.error(f"Database backup failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Use chunked backup by default for large databases
        backup_file = backup_database(use_chunked=True, chunk_size=1000)
        print(f"Backup completed successfully. File saved to: {backup_file}")
    except Exception as e:
        logger.error(f"Backup process failed: {str(e)}")
        print(f"Error: {str(e)}")
        print("Please make sure SUPABASE_URL and SUPABASE_KEY environment variables are set correctly.")
        print("You can set them in a .env file or as system environment variables.")
        exit(1)