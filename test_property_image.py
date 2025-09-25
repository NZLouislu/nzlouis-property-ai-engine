#!/usr/bin/env python3
"""
Test script to verify that property_image.py components work correctly
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all required modules can be imported"""
    print("  Testing individual imports...")
    
    imports = [
        ("time", lambda: __import__('time')),
        ("requests", lambda: __import__('requests')),
        ("bs4", lambda: __import__('bs4')),
        ("fetch_property_links", lambda: __import__('fetch_property_links')),
        ("properties", lambda: __import__('properties')),
        ("config.supabase_config", lambda: __import__('config.supabase_config')),
        ("traceback", lambda: __import__('traceback')),
        ("sys", lambda: __import__('sys')),
        ("logging", lambda: __import__('logging')),
        ("os", lambda: __import__('os')),
        ("dotenv", lambda: __import__('dotenv'))
    ]
    
    failed_imports = []
    for name, import_func in imports:
        try:
            import_func()
            print(f"    ✅ {name}")
        except Exception as e:
            print(f"    ❌ {name}: {e}")
            failed_imports.append((name, str(e)))
    
    if failed_imports:
        print(f"  ❌ Failed imports: {failed_imports}")
        return False
    else:
        print("  ✅ All imports successful")
        return True

def test_supabase_connection():
    """Test Supabase connection"""
    try:
        from config.supabase_config import create_supabase_client
        supabase = create_supabase_client()
        print("✅ Supabase client created successfully")
        
        # Test a simple query
        response = supabase.table('scraping_progress').select('*').limit(1).execute()
        print("✅ Supabase connection test successful")
        return True
    except Exception as e:
        print(f"❌ Supabase connection test failed: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set"""
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def main():
    print("Running property_image.py component tests...\n")
    
    tests = [
        ("Import tests", test_imports),
        ("Environment variable tests", test_environment_variables),
        ("Supabase connection tests", test_supabase_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
        print()
    
    if all(results):
        print("🎉 All tests passed! property_image.py should work correctly.")
        return 0
    else:
        print("💥 Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())