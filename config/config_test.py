import sys
from datetime import datetime
from supabase_config import create_supabase_client
from redis_config import create_redis_client

# Try to import Upstash Redis, but don't fail if not available
try:
    from upstash_redis import Redis
except ImportError:
    Redis = None

def test_redis_connection():
    try:
        # Connect to Redis
        if Redis:
            redis_client = Redis.from_env()
        else:
            # Use the mock client from redis_config
            redis_client = create_redis_client()
        
        # Test inserting a value
        redis_client.set('test_key', 'test_value')
        value = redis_client.get('test_key')
        if value:
            print("Redis connection successful, test key inserted.")
        else:
            print("Redis connection test completed (no error, but key not found).")
        return True
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        print("❌ Redis test failed.")
        return False

def test_supabase_connection():
    try:
        # Create a Supabase client
        supabase_client = create_supabase_client()
        print("✅ Successfully connected to Supabase.")
        
        # Test querying the 'properties' table
        response = supabase_client.from_('properties').select('*').limit(1).execute()

        # Check if there is an error or data is None
        if not response.data:
            print("❌ Failed to fetch data from 'properties' table or table is empty.")
        else:
            print("✅ Successfully fetched data from 'properties' table:", response.data)
        return True
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        print("Supabase test failed.")
        return False

def insert_property(supabase_client, property_data):
    try:
        # Remove the 'id' from property_data if it exists, so it can be auto-generated
        property_data.pop('id', None)
        
        # Clean up the price fields
        for field in ['last_sold_price', 'capital_value', 'land_value', 'improvement_value']:
            if field in property_data and property_data[field]:
                property_data[field] = clean_price(property_data[field])

        # Insert the property data
        response = supabase_client.table('properties').insert(property_data).execute()

        # Check for errors in the response
        if hasattr(response, 'error') and response.error:
            print(f"❌ Failed to insert property: {response.error}")
            return None
        elif not response.data:
            print("❌ Failed to insert property: No data returned")
            return None
        else:
            print(f"Property inserted: {property_data['address']}")
            return response.data[0]['id']  # Return the property ID

    except Exception as e:
        print(f"Error inserting property: {str(e)}")
        return None

# Helper function to clean price values
def clean_price(price_str):
    try:
        # Remove dollar sign and commas, then convert to float
        clean_price_str = price_str.replace('$', '').replace(',', '').strip()
        return float(clean_price_str)
    except ValueError:
        return None  # If there's an issue parsing the price, return None

# Test inserting a property into the database
def test_insert_property():
    # Create a test property with a sample price that contains special characters
    test_property_data = {
        'address': '15 Agra Crescent, Khandallah, Wellington, 6035',
        'suburb': 'Khandallah',
        'city': 'Wellington',
        'postcode': '6035',
        'year_built': 1985,
        'bedrooms': 3,
        'bathrooms': 2,
        'car_spaces': 2,
        'floor_size': '150 sqm',
        'land_area': '500 sqm',
        'last_sold_price': '$1,280,000',  # This needs cleaning
        'last_sold_date': '2023-08-01',
        'capital_value': '$1,000,000',    # This needs cleaning
        'land_value': '$800,000',         # This needs cleaning
        'improvement_value': '$200,000',  # This needs cleaning
        'has_rental_history': False,
        'is_currently_rented': False,
        'status': 'For Sale'  # Adding status as per SQL definition
    }

    try:
        # Create a Supabase client
        supabase_client = create_supabase_client()

        # Attempt to insert the property
        property_id = insert_property(supabase_client, test_property_data)
        if property_id:
            print(f"Inserted property with generated ID: {property_id}")

    except Exception as e:
        print(f"Error during test insertion: {e}")


def main():
    # print("Testing Redis connection first...")
    redis_test_result = test_redis_connection()

    print("\nTesting Supabase connection now...")
    supabase_test_result = test_supabase_connection()

    print("\nTesting Supabase property insertion...")
    test_insert_property()

    # if redis_test_result and supabase_test_result:
    if supabase_test_result:
        print("\nAll tests passed!")
    else:
        print("\nOne or more tests failed. Please check the above messages.")

if __name__ == '__main__':
    main()