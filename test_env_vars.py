#!/usr/bin/env python3
"""
Test script to verify that environment variables are correctly set
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_env_vars():
    """Test that all required environment variables are set"""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY',
        'REALESTATE_URL',
        'REALESTATE_RENT_URL',
        'PROPERTY_VALUE_BASE_URL'
    ]
    
    print("Checking environment variables...")
    all_good = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:50]}{'...' if len(value) > 50 else ''}")
        else:
            print(f"  ❌ {var}: NOT SET")
            all_good = False
    
    # Also check Redis variables
    redis_vars = [
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    
    print("\nChecking Redis environment variables...")
    for var in redis_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:50]}{'...' if len(value) > 50 else ''}")
        else:
            print(f"  ⚠️  {var}: NOT SET (may not be needed)")
    
    return all_good

def main():
    print("Running environment variable tests...\n")
    
    if test_env_vars():
        print("\n🎉 All required environment variables are set!")
        return 0
    else:
        print("\n💥 Some required environment variables are missing!")
        return 1

if __name__ == "__main__":
    exit(main())