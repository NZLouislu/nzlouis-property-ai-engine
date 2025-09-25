from datetime import datetime
import os
from supabase import create_client, Client
import logging

# Set up logging
logger = logging.getLogger(__name__)

def create_supabase_client() -> Client:
    """
    Create and return a Supabase client instance.
    """
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
            
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Error creating Supabase client: {e}")
        raise

def clean_price(price_str):
    if price_str is None:
        return None
    if isinstance(price_str, (int, float)):
        return price_str
    try:
        return float(price_str.replace('$', '').replace(',', '').strip())
    except ValueError:
        return None

def clean_property_data(property_data):
    price_fields = ['last_sold_price', 'capital_value', 'land_value', 'improvement_value']
    for field in price_fields:
        if field in property_data:
            property_data[field] = clean_price(property_data[field])
    return property_data

def parse_date(date_str):
    if not date_str:
        return None
    date_formats = ['%d %b %Y', '%Y', '%b %Y', '%d/%m/%Y', '%Y-%m-%d']
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    print(f"Warning: Unable to parse date '{date_str}'")
    return None

def format_date_for_json(date_obj):
    if date_obj is None:
        return None
    return date_obj.isoformat()  # Converts date to ISO 8601 string format

def insert_property_and_history(property_data: dict, history_data: list) -> bool:
    """
    Insert a property and its history into the Supabase database.
    Returns True if successful, False if duplicate.
    """
    supabase = create_supabase_client()
    
    try:
        # Clean property data
        cleaned_property_data = clean_property_data(property_data)
        
        # Try to insert the property
        property_response = supabase.table('properties').insert(cleaned_property_data).execute()
        
        if not property_response.data:
            logger.warning("Failed to insert property")
            return False
            
        property_id = property_response.data[0]['id']
        logger.info(f"Successfully inserted property with ID: {property_id}")
        
        # Insert history data if provided
        if history_data and isinstance(history_data, list):
            for event in history_data:
                history_entry = {
                    'property_id': property_id,
                    'event_description': event.get('event_description', ''),
                    'event_date': format_date_for_json(parse_date(event.get('event_date'))),
                    'interval_since_last_event': event.get('event_interval', '')
                }
                
                # Only insert if we have a valid date
                if history_entry['event_date'] is not None:
                    try:
                        history_response = supabase.table('property_history').insert(history_entry).execute()
                        if not history_response.data:
                            logger.warning(f"Failed to insert history: {event}")
                    except Exception as e:
                        logger.error(f"Error inserting history: {str(e)}")
                        logger.warning(f"Skipped history entry: {event}")
                else:
                    logger.info(f"Skipped invalid date entry: {event}")
            
            logger.info(f"Property history insertion completed for property ID {property_id}")
        else:
            logger.info("No history data to insert.")
        
        return True
        
    except Exception as e:
        # Check if it's a duplicate key error
        error_str = str(e).lower()
        if "duplicate key" in error_str or "unique constraint" in error_str:
            logger.info(f"Property already exists in database: {property_data.get('address', 'Unknown')}")
            return False
        else:
            logger.error(f"Error inserting property and history: {e}")
            raise

def check_property_exists(address: str) -> bool:
    """
    Check if a property already exists in the Supabase database by address.
    """
    try:
        supabase = create_supabase_client()
        response = supabase.table('properties').select('id').eq('address', address).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Property already exists in database: {address}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking property in database: {e}")
        # If there's an error checking, we assume it doesn't exist to avoid missing data
        return False

def insert_real_estate(address: str, status: str) -> bool:
    """
    Insert a real estate property into the Supabase database.
    Returns True if successful, False if duplicate (based on address).
    """
    try:
        supabase = create_supabase_client()
        
        # Try to insert the property
        response = supabase.table('real_estate').insert({
            'address': address,
            'status': status
        }).execute()
        
        if response.data:
            logger.info(f"Successfully inserted property: {address}")
            return True
        else:
            logger.warning(f"Failed to insert property: {address}")
            return False
            
    except Exception as e:
        # Check if it's a duplicate key error
        error_str = str(e).lower()
        if "duplicate key" in error_str or "unique constraint" in error_str:
            logger.info(f"Property already exists in database: {address}")
            return False
        else:
            logger.error(f"Error inserting property {address}: {e}")
            raise

def insert_real_estate_rent(address: str, status: str) -> bool:
    """
    Insert a rental property into the Supabase database.
    Returns True if successful, False if duplicate (based on address).
    """
    try:
        supabase = create_supabase_client()
        
        # Try to insert the property
        response = supabase.table('real_estate_rent').insert({
            'address': address,
            'status': status
        }).execute()
        
        if response.data:
            logger.info(f"Successfully inserted rental property: {address}")
            return True
        else:
            logger.warning(f"Failed to insert rental property: {address}")
            return False
            
    except Exception as e:
        # Check if it's a duplicate key error
        error_str = str(e).lower()
        if "duplicate key" in error_str or "unique constraint" in error_str:
            logger.info(f"Rental property already exists in database: {address}")
            return False
        else:
            logger.error(f"Error inserting rental property {address}: {e}")
            raise