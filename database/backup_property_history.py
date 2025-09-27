#!/usr/bin/env python3
"""
Property History Table Backup Script
专门备份 property_history 表的脚本
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, Optional
from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("property_history_backup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BACKUP_DIR = "database/backup"
CHUNK_SIZE = 500

def create_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def ensure_backup_directory():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logger.info(f"Created backup directory: {BACKUP_DIR}")

def check_backup_status() -> bool:
    try:
        supabase = create_supabase_client()
        response = supabase.table('scraping_progress').select('status, updated_at').eq('id', 8).execute()
        
        if response.data:
            status = response.data[0]['status']
            logger.info(f"Current property_history backup status: {status}")
            
            if status == 'running':
                logger.info("Property history backup is already running, exiting")
                return True
            
            if status in ['complete', 'idle', 'stop']:
                logger.info(f"Previous backup status was '{status}', starting new backup")
                return False
        else:
            logger.info("No property_history backup record found, will create new one")
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking backup status: {e}")
        return False

def update_backup_status(status: str, message: str = ""):
    try:
        supabase = create_supabase_client()
        
        data = {
            'status': status,
            'updated_at': 'now()'
        }
        
        if message:
            data['last_processed_id'] = message
        
        response = supabase.table('scraping_progress').update(data).eq('id', 8).execute()
        
        if not response.data:
            data['id'] = 8
            data['batch_size'] = CHUNK_SIZE
            supabase.table('scraping_progress').insert(data).execute()
            
        logger.info(f"Updated property_history backup status to: {status}")
        
    except Exception as e:
        logger.warning(f"Could not update backup status: {e}")

def get_table_statistics():
    try:
        supabase = create_supabase_client()
        
        count_response = supabase.table('property_history').select('id', count='exact').execute()
        total_records = count_response.count if hasattr(count_response, 'count') else 0
        
        try:
            earliest_response = supabase.table('property_history').select('created_at').order('created_at', desc=False).limit(1).execute()
            earliest_date = earliest_response.data[0]['created_at'] if earliest_response.data else 'Unknown'
            
            latest_response = supabase.table('property_history').select('created_at').order('created_at', desc=True).limit(1).execute()
            latest_date = latest_response.data[0]['created_at'] if latest_response.data else 'Unknown'
            
            return {
                'total_records': total_records,
                'earliest_date': earliest_date,
                'latest_date': latest_date
            }
        except Exception as e:
            logger.warning(f"Could not get date range: {e}")
            return {
                'total_records': total_records,
                'earliest_date': 'Unknown',
                'latest_date': 'Unknown'
            }
            
    except Exception as e:
        logger.error(f"Error getting table statistics: {e}")
        return {
            'total_records': 0,
            'earliest_date': 'Unknown',
            'latest_date': 'Unknown'
        }

def backup_property_history_data(run_timestamp: str) -> Dict[str, str]:
    try:
        supabase = create_supabase_client()
        timestamp = run_timestamp
        
        logger.info("Starting property_history table backup...")
        
        all_data = []
        offset = 0
        total_records = 0
        
        while True:
            response = supabase.table('property_history').select("*").range(offset, offset + CHUNK_SIZE - 1).execute()
            
            if not response.data:
                break
                
            all_data.extend(response.data)
            total_records += len(response.data)
            offset += CHUNK_SIZE
            
            logger.info(f"  Fetched {len(response.data)} records (total: {total_records})")
            
            if len(response.data) < CHUNK_SIZE:
                break
        
        if not all_data:
            logger.warning("No data found in property_history table")
            return {}
        
        backup_files = {"backup_id": timestamp}
        
        json_filename = f"property_history_backup_{timestamp}.json"
        json_path = os.path.join(BACKUP_DIR, json_filename)
        
        backup_data = {
            "table_name": "property_history",
            "backup_timestamp": datetime.utcnow().isoformat(),
            "record_count": len(all_data),
            "chunk_size_used": CHUNK_SIZE,
            "backup_method": "chunked_supabase_api",
            "data": all_data
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str, ensure_ascii=False)
        
        json_size = os.path.getsize(json_path)
        json_size_mb = json_size / (1024 * 1024)
        backup_files['json'] = json_path
        
        logger.info(f"✓ JSON backup created: {json_filename} ({json_size_mb:.2f} MB)")
        
        csv_filename = f"property_history_backup_{timestamp}.csv"
        csv_path = os.path.join(BACKUP_DIR, csv_filename)
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                if all_data:
                    fieldnames = all_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_data)
            
            csv_size = os.path.getsize(csv_path)
            csv_size_mb = csv_size / (1024 * 1024)
            backup_files['csv'] = csv_path
            
            logger.info(f"✓ CSV backup created: {csv_filename} ({csv_size_mb:.2f} MB)")
            
        except Exception as csv_error:
            logger.warning(f"Could not create CSV backup: {csv_error}")
        
        metadata_filename = f"property_history_backup_metadata_{timestamp}.json"
        metadata_path = os.path.join(BACKUP_DIR, metadata_filename)
        
        stats = get_table_statistics()
        
        metadata = {
            "backup_timestamp": datetime.utcnow().isoformat(),
            "backup_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "backup_time": datetime.utcnow().strftime("%H:%M:%S"),
            "backup_type": "property_history_only",
            "table_name": "property_history",
            "actual_record_count": len(all_data),
            "estimated_total_records": stats['total_records'],
            "date_range": {
                "earliest": stats['earliest_date'],
                "latest": stats['latest_date']
            },
            "files": {
                "json_backup_file": os.path.basename(json_path),
                "csv_backup_file": os.path.basename(csv_path) if 'csv' in backup_files else None,
                "json_file_size_mb": json_size_mb,
                "csv_file_size_mb": csv_size_mb if 'csv' in backup_files else 0
            },
            "backup_method": "chunked_supabase_api",
            "chunk_size": CHUNK_SIZE,
            "restore_instructions": {
                "json_restore": "Use restore script to import JSON data back to property_history table",
                "csv_restore": "Import CSV file using database tools or custom scripts",
                "delete_command": "DELETE FROM property_history; -- Use with caution!",
                "truncate_command": "TRUNCATE TABLE property_history; -- Faster delete with auto-increment reset"
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        backup_files['metadata'] = metadata_path
        logger.info(f"✓ Metadata file created: {metadata_filename}")
        
        return backup_files
        
    except Exception as e:
        logger.error(f"Error backing up property_history data: {e}")
        raise

def main():
    try:
        logger.info("=" * 60)
        logger.info("PROPERTY HISTORY TABLE BACKUP")
        logger.info("=" * 60)
        
        # 本次备份ID，用于文件命名与数据库记录
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if check_backup_status():
            logger.info("Property history backup cannot proceed at this time")
            return
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        ensure_backup_directory()
        
        update_backup_status('running', f'Starting property_history table backup | backup_id:{run_timestamp}')
        
        stats = get_table_statistics()
        logger.info(f"Property History table statistics:")
        logger.info(f"  - Total records: {stats['total_records']}")
        logger.info(f"  - Date range: {stats['earliest_date']} to {stats['latest_date']}")
        
        backup_files = backup_property_history_data(run_timestamp)
        
        if backup_files:
            update_backup_status('complete', f"backup_id:{run_timestamp} | files:{len(backup_files)}")
            
            logger.info("=" * 60)
            logger.info("PROPERTY HISTORY BACKUP COMPLETED!")
            logger.info("=" * 60)
            logger.info("Created backup files:")
            
            for backup_type, filepath in backup_files.items():
                logger.info(f"  {backup_type.upper()}: {filepath}")
            
            logger.info("\nRESTORE INSTRUCTIONS:")
            logger.info("-" * 40)
            logger.info("1. To delete current property_history data:")
            logger.info("   DELETE FROM property_history;")
            logger.info("   OR")
            logger.info("   TRUNCATE TABLE property_history;")
            logger.info("2. To restore from JSON backup:")
            logger.info("   Use custom restore script with the JSON file")
            logger.info("3. To restore from CSV backup:")
            logger.info("   Import CSV file using database tools")
            
            print(f"\n✓ Property History backup completed successfully!")
            print(f"✓ Created {len(backup_files)} backup files in {BACKUP_DIR}/")
            print(f"✓ You can now safely delete the property_history table data")
            
        else:
            update_backup_status('idle', f'backup_id:{run_timestamp} | Property History backup failed - no files created')
            logger.error("No backup files were created!")
            print("✗ Property History backup failed - no files were created")
            exit(1)
            
    except Exception as e:
        logger.error(f"Property History backup process failed: {str(e)}")
        try:
            # 如果 run_timestamp 可用，记录本次备份ID
            update_backup_status('idle', f'backup_id:{run_timestamp} | error:{str(e)}')
        except Exception:
            update_backup_status('idle', f'error:{str(e)}')
        print(f"✗ Error: {str(e)}")
        print("\nPlease check:")
        print("- SUPABASE_URL and SUPABASE_KEY environment variables are set correctly")
        print("- Network connectivity to Supabase")
        print("- property_history table exists and is accessible")
        exit(1)

if __name__ == "__main__":
    main()