#!/usr/bin/env python3
"""
Supabase Database Restore Script
Helps restore database from backups created by backup_supabase.py
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

def restore_from_pg_dump(backup_file: str, target_database_url: str):
    """
    Restore database from pg_dump backup file
    
    Args:
        backup_file (str): Path to the .dump backup file
        target_database_url (str): Target database URL to restore to
    """
    print(f"Restoring database from: {backup_file}")
    print(f"Target database: {target_database_url[:50]}...")
    
    try:
        if backup_file.endswith('.dump'):
            # Use pg_restore for custom format
            cmd = [
                'pg_restore',
                '--clean',          # Clean (drop) database objects before recreating
                '--if-exists',      # Use IF EXISTS when dropping objects
                '--no-owner',       # Don't restore ownership
                '--no-privileges',  # Don't restore privileges
                '--verbose',
                '--dbname', target_database_url,
                backup_file
            ]
        elif backup_file.endswith('.sql'):
            # Use psql for SQL format
            cmd = ['psql', target_database_url, '-f', backup_file]
        else:
            raise ValueError("Backup file must be .dump or .sql format")
        
        print(f"Running restore command...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Database restore completed successfully!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"✗ Restore failed with return code {result.returncode}")
            print("Error:", result.stderr)
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error during restore: {e}")
        return False

def list_available_backups(backup_dir: str = "database/backup"):
    """List all available backup files"""
    if not os.path.exists(backup_dir):
        print(f"Backup directory not found: {backup_dir}")
        return []
    
    backup_files = []
    for filename in os.listdir(backup_dir):
        if filename.endswith(('.dump', '.sql', '.json')):
            filepath = os.path.join(backup_dir, filename)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            backup_files.append({
                'filename': filename,
                'filepath': filepath,
                'size_mb': file_size,
                'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
            })
    
    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: x['modified'], reverse=True)
    
    return backup_files

def show_backup_info(backup_dir: str = "database/backup"):
    """Show information about available backups"""
    backups = list_available_backups(backup_dir)
    
    if not backups:
        print("No backup files found.")
        return
    
    print("\nAvailable backup files:")
    print("-" * 80)
    print(f"{'Filename':<40} {'Size (MB)':<10} {'Modified':<20} {'Type':<10}")
    print("-" * 80)
    
    for backup in backups:
        file_type = "pg_dump" if backup['filename'].endswith('.dump') else \
                   "SQL" if backup['filename'].endswith('.sql') else "JSON"
        
        print(f"{backup['filename']:<40} {backup['size_mb']:<10.2f} "
              f"{backup['modified'].strftime('%Y-%m-%d %H:%M'):<20} {file_type:<10}")
    
    # Look for metadata files
    metadata_files = [b for b in backups if 'metadata' in b['filename']]
    if metadata_files:
        print(f"\nFound {len(metadata_files)} metadata file(s) with backup information.")

def main():
    parser = argparse.ArgumentParser(description='Restore Supabase database from backup')
    parser.add_argument('--list', action='store_true', help='List available backup files')
    parser.add_argument('--restore', type=str, help='Path to backup file to restore')
    parser.add_argument('--target-db', type=str, help='Target database URL')
    parser.add_argument('--backup-dir', type=str, default='database/backup', 
                       help='Backup directory (default: database/backup)')
    
    args = parser.parse_args()
    
    if args.list:
        show_backup_info(args.backup_dir)
        return
    
    if args.restore:
        if not args.target_db:
            print("Error: --target-db is required when restoring")
            print("Example: --target-db 'postgresql://user:pass@host:5432/dbname'")
            sys.exit(1)
        
        if not os.path.exists(args.restore):
            print(f"Error: Backup file not found: {args.restore}")
            sys.exit(1)
        
        # Confirm before restore
        print(f"WARNING: This will restore data to the target database.")
        print(f"Backup file: {args.restore}")
        print(f"Target database: {args.target_db[:50]}...")
        
        confirm = input("\nAre you sure you want to continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Restore cancelled.")
            return
        
        success = restore_from_pg_dump(args.restore, args.target_db)
        if not success:
            sys.exit(1)
    else:
        # Show help and available backups
        parser.print_help()
        print("\n")
        show_backup_info(args.backup_dir)
        print("\nExamples:")
        print("  python restore_supabase.py --list")
        print("  python restore_supabase.py --restore database/backup/backup.dump --target-db 'postgresql://...'")

if __name__ == "__main__":
    main()