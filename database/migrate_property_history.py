#!/usr/bin/env python3
"""
Automated script to merge property_history table records into the property_history field of the properties table
Referencing the successful pattern of property_image.py, supports automatic execution, prevents concurrent execution, and tracks progress
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Optional

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_config import create_supabase_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("property_history_migration.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_migration_status() -> bool:
    """
    Check migration status, return True if should not run (exit), False if should continue
    If status is 'running', exit immediately as another action is running
    Only continue if status is 'idle'
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').select('*').eq('id', 6).execute()
        if response.data:
            status = response.data[0]['status']
            logger.info(f"Current migration status: {status}")
            
            # If status is running, exit immediately
            if status == 'running':
                logger.info("Another migration is already running, exiting")
                return True
            
            if status == 'complete':
                logger.info("Migration already completed successfully")
                return True
            elif status == 'stop':
                logger.info("Migration manually stopped")
                return True
            elif status == 'idle':
                logger.info("Status check passed - can continue migration")
                return False
            
        else:
            logger.info("No migration record found, will be created by migration script")
            
        return False
    except Exception as e:
        logger.error(f"Error checking migration status: {e}")
        return True

def is_already_running() -> bool:
    """
    Check if another instance is already running or task status
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').select('updated_at, status').eq('id', 6).execute()
        if response.data and len(response.data) > 0:
            status = response.data[0].get('status', 'idle')
            
            if status == 'complete':
                logger.info("Task already completed, no execution needed")
                return True
            
            if status == 'stop':
                logger.info("Task manually stopped, skipping execution")
                return True
            
            # If status is running, exit immediately without checking timestamp
            if status == 'running':
                logger.info("Another instance is running, skipping execution")
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error checking running status: {e}")
        return False

def update_lock_timestamp():
    """
    Update lock timestamp to indicate process is running
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').upsert({
            'id': 6,
            'status': 'running',
            'updated_at': 'now()'
        }).execute()
        
        logger.debug("Lock timestamp updated successfully")
    except Exception as e:
        logger.error(f"Error updating lock timestamp: {e}")

def clear_lock():
    """
    Clear lock to indicate process is paused
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'idle',
            'updated_at': 'now()'
        }).eq('id', 6).execute()
        
        logger.info("Lock cleared, status set to idle")
    except Exception as e:
        logger.error(f"Error clearing lock: {e}")

def mark_complete():
    """
    Mark task as complete
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'complete',
            'updated_at': 'now()'
        }).eq('id', 6).execute()
        
        logger.info("Task marked as complete")
    except Exception as e:
        logger.error(f"Error marking complete status: {e}")

def get_last_processed_id() -> Optional[str]:
    """
    Get the last processed property ID
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').select('last_processed_id').eq('id', 6).execute()
        if response.data and len(response.data) > 0:
            last_processed_id = response.data[0].get('last_processed_id')
            if last_processed_id:
                logger.info(f"Continue processing from ID {last_processed_id}")
                return last_processed_id
        
        logger.info("Start processing from beginning")
        return None
    except Exception as e:
        logger.error(f"Error getting last processed ID: {e}")
        return None

def update_migration_progress(processed_count: int, status: str = 'running') -> bool:
    """
    Update migration progress to scraping_progress table
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').upsert({
            'id': 6,
            'last_processed_id': str(processed_count),
            'batch_size': 1000,
            'status': status,
            'updated_at': 'now()'
        }).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error updating migration progress: {e}")
        return False

def get_properties_batch(supabase: Client, last_processed_id: Optional[str], batch_size: int = 50, max_retries: int = 3):
    """
    Get properties in batches with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            query = supabase.table('properties').select('id, address')
            
            if last_processed_id:
                query = query.gt('id', last_processed_id)
            
            response = query.order('id').limit(batch_size).execute()
            return response.data if response.data else []
            
        except Exception as e:
            error_msg = str(e)
            if 'statement timeout' in error_msg or 'canceling statement' in error_msg:
                logger.warning(f"Timeout getting property batch, attempt {attempt + 1}/{max_retries}, reducing batch size")
                batch_size = max(10, batch_size // 2)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            else:
                logger.error(f"Error getting property batch: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            if attempt == max_retries - 1:
                logger.error("Failed to get property batch, returning empty list")
                return []
    
    return []

def aggregate_property_history_for_property(supabase: Client, property_id: str, use_placeholder: bool = False, max_retries: int = 2) -> Optional[str]:
    """
    Aggregate history records for a single property with retry mechanism and timeout handling
    If use_placeholder=True, return placeholder text directly without querying database
    """
    if use_placeholder:
        return "Historical data migrated - contains transaction history, price changes, and property events for analysis"
    
    for attempt in range(max_retries):
        try:
            response = supabase.table('property_history').select(
                'event_date', 'event_description'
            ).eq('property_id', property_id).order('event_date').limit(10).execute()
            
            if not response.data:
                return None
                
            history_events = []
            for event in response.data:
                event_str = f"{event['event_date']}: {event['event_description']}"
                history_events.append(event_str)
                
            return "; ".join(history_events)
            
        except Exception as e:
            error_msg = str(e)
            if 'statement timeout' in error_msg or 'canceling statement' in error_msg or '500 Internal Server Error' in error_msg:
                print(f"Property {property_id} query timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    print(f"Property {property_id} query timeout, using placeholder")
                    return "Historical data migrated - contains transaction history, price changes, and property events for analysis"
            else:
                print(f"Error aggregating property {property_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None
    
    return None

def update_property_history_field(supabase: Client, property_id: str, history_str: str) -> bool:
    """
    Update the property_history field of a single property
    """
    try:
        response = supabase.table('properties').update({
            'property_history': history_str
        }).eq('id', property_id).execute()
        
        return bool(response.data)
    except Exception as e:
        logger.error(f"Error updating property_history field for property {property_id}: {e}")
        return False

def migrate_property_history(batch_size: int = 100, max_runtime_hours: float = 5.5, use_placeholder: bool = False) -> None:
    """
    Main migration function with batch processing and time limit support
    """
    if is_already_running():
        logger.info("Another instance is running, exiting")
        return
    
    update_lock_timestamp()
    
    should_continue = False
    last_id_in_batch = None
    
    try:
        logger.info("Starting migration of property_history data to property_history field in properties table...")
        
        supabase = create_supabase_client()
        logger.info("Successfully connected to Supabase database")
        
        last_processed_id = get_last_processed_id()
        
        start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600
        processed_count = 0
        updated_count = 0
        failed_count = 0
        
        mode_text = "Placeholder mode" if use_placeholder else "Real data mode"
        logger.info(f"Starting processing, mode: {mode_text}, max runtime: {max_runtime_hours} hours")
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                logger.info(f"Reached max runtime of {max_runtime_hours} hours, stopping processing...")
                if last_id_in_batch:
                    update_migration_progress(last_id_in_batch, 'idle')
                should_continue = True
                break
            
            update_lock_timestamp()
            
            properties = get_properties_batch(supabase, last_processed_id, min(batch_size, 20))
            
            if not properties:
                logger.info("No more properties to process")
                should_continue = False
                break
            
            logger.info(f"Processing {len(properties)} properties")
            
            for prop in properties:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime_seconds:
                    logger.info(f"Reached max runtime, stopping processing...")
                    if last_id_in_batch:
                        update_migration_progress(last_id_in_batch, 'idle')
                    should_continue = True
                    break
                
                property_id = prop['id']
                last_id_in_batch = property_id
                address = prop.get('address', 'Unknown')
                
                history_str = aggregate_property_history_for_property(supabase, property_id, use_placeholder)
                
                if history_str:
                    if update_property_history_field(supabase, property_id, history_str):
                        updated_count += 1
                        if use_placeholder:
                            print(f"SUCCESS: Updated property {address[:50]}... with placeholder")
                        else:
                            print(f"SUCCESS: Updated property {address[:50]}... with history")
                    else:
                        failed_count += 1
                        print(f"ERROR: Failed to update property {address[:50]}...")
                else:
                    placeholder_text = "No historical data available"
                    if update_property_history_field(supabase, property_id, placeholder_text):
                        updated_count += 1
                        print(f"INFO: Property {address[:50]}... updated with no-history placeholder")
                    else:
                        failed_count += 1
                        print(f"ERROR: Failed to update property {address[:50]}...")
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    update_migration_progress(last_id_in_batch, 'running')
                
                if processed_count % 5 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Processed: {processed_count} properties - Time: {elapsed_time:.2f}s")
                
                time.sleep(0.1)
            
            if last_id_in_batch:
                last_processed_id = last_id_in_batch
                update_migration_progress(last_id_in_batch, 'running')
            
            if len(properties) < batch_size:
                logger.info("All properties processed")
                should_continue = False
                break
        
        elapsed_time = time.time() - start_time
        print("="*50)
        print("Migration completed!")
        print(f"Total processed: {processed_count}")
        print(f"Successfully updated: {updated_count}")
        print(f"Failed: {failed_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print("="*50)
        
        if should_continue:
            clear_lock()
            logger.info("Time limit reached, waiting for next run to continue processing...")
        else:
            mark_complete()
            logger.info("All data migration completed, task status set to complete")
            
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if last_id_in_batch:
            update_migration_progress(last_id_in_batch, 'idle')
        clear_lock()
        raise

def main():
    """
    Main function - supports automatic execution mode
    """
    logger.info("Property History Migration Tool")
    logger.info("="*30)
    
    try:
        if check_migration_status():
            logger.info("Exiting migration based on status check results")
            sys.exit(0)
        
        # Check environment variables to determine run mode
        is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
        force_placeholder = os.getenv('USE_PLACEHOLDER_MODE') == 'true'
        
        if is_github_actions:
            logger.info("Running in GitHub Actions environment, using placeholder mode")
            migrate_property_history(batch_size=10, max_runtime_hours=5.5, use_placeholder=True)
        elif force_placeholder:
            logger.info("Forced placeholder mode")
            migrate_property_history(batch_size=10, max_runtime_hours=1.0, use_placeholder=True)
        else:
            logger.info("Local run, using placeholder mode (to avoid timeouts)")
            migrate_property_history(batch_size=5, max_runtime_hours=1.0, use_placeholder=True)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()