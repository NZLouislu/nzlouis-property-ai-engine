#!/usr/bin/env python3
"""
Test script to verify that real_estate_auckland.py components work correctly
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
        ("playwright.sync_api", lambda: __import__('playwright.sync_api')),
        ("time", lambda: __import__('time')),
        ("random", lambda: __import__('random')),
        ("os", lambda: __import__('os')),
        ("dotenv", lambda: __import__('dotenv')),
        ("traceback", lambda: __import__('traceback')),
        ("logging", lambda: __import__('logging')),
        ("config.supabase_config", lambda: __import__('config.supabase_config')),
    ]
    
    failed_imports = []
    for name, import_func in imports:
        try:
            import_func()
            print(f"✅ {name}")
        except Exception as e:
            print(f" ❌ {name}: {e}")
            failed_imports.append((name, str(e)))
    
    # Test specific imports from config.supabase_config
    try:
        from config.supabase_config import insert_real_estate, create_supabase_client
        print(f" ✅ config.supabase_config.insert_real_estate")
        print(f"✅ config.supabase_config.create_supabase_client")
    except Exception as e:
        print(f" ❌ config.supabase_config functions: {e}")
        failed_imports.append(("config.supabase_config functions", str(e)))
    
    if failed_imports:
        print(f" ❌ Failed imports: {failed_imports}")
        return False
    else:
        print("  ✅ All imports successful")
        return True

def test_environment_variables():
    """Test that required environment variables are set"""
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'REALESTATE_URL']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"  ❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("  ✅ All required environment variables are set")
        return True

def main():
    print("Running real_estate_auckland.py component tests...\n")
    
    tests = [
        ("Import tests", test_imports),
        ("Environment variable tests", test_environment_variables),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append(False)
        print()
    
    if all(results):
        print("🎉 All tests passed! real_estate_auckland.py should work correctly.")
        return 0
    else:
        print("💥 Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())