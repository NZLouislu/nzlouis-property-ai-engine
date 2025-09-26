import time
import requests
from bs4 import BeautifulSoup
from fetch_property_links import fetch_property_links
from properties import fetch_property_details
import os
from dotenv import load_dotenv
import traceback
import logging
from config.supabase_config import create_supabase_client, insert_property_and_history

# Load environment variables
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("propertyvalue_wellington.log"),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger(__name__)

def create_scraping_progress_table():
    """
    Create a table to track scraping progress.
    """
    supabase = create_supabase_client()
    try:
        # Check if the scraping_progress table exists by trying to select from it
        response = supabase.table('scraping_progress').select('*').limit(1).execute()
        logger.info("Scraping progress table already exists.")
    except Exception as e:
        logger.error(f"Scraping progress table may not exist: {e}")
        logger.info("Please ensure the scraping_progress table exists in your Supabase database with the following structure:")
        logger.info("- id (int, primary key)")
        logger.info("- last_processed_id (text)")
        logger.info("- batch_size (int)")
        logger.info("- updated_at (timestamp)")

def get_last_processed_region():
    """
    Get the last processed region ID from the progress table for PropertyValue Wellington.
    """
    supabase = create_supabase_client()
    try:
        # Try to get the record with id=5 for PropertyValue Wellington
        response = supabase.table('scraping_progress').select('last_processed_id').eq('id', 5).execute()
        if response.data and len(response.data) > 0:
            last_processed_id = response.data[0].get('last_processed_id')
            # Return the region ID if it's not None or empty
            if last_processed_id:
                logger.info(f"Resuming from region ID: {last_processed_id}")
                return last_processed_id
            else:
                # If last_processed_id is None or empty, it means we're starting from the beginning
                logger.info("Starting from the beginning (empty last_processed_id)")
                return None
        
        logger.info("Starting from the beginning (no progress records found for PropertyValue Wellington)")
        return None
    except Exception as e:
        logger.error(f"Error getting last processed region: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

def update_last_processed_region(region_id):
    """
    Update the last processed region ID in the progress table for PropertyValue Wellington.
    """
    supabase = create_supabase_client()
    try:
        # First, try to update the existing record with id=5 for PropertyValue Wellington
        response = supabase.table('scraping_progress').update({
            'last_processed_id': region_id,
            'batch_size': 1000,  # Default batch size
            'updated_at': 'now()'
        }).eq('id', 5).execute()
        
        # Check if the update was successful
        if response.data:
            logger.info(f"Updated last processed region ID to: {region_id}")
        else:
            # If no record was updated, insert a new one with id=5
            data = {
                'id': 5,  # Use ID 5 for PropertyValue Wellington progress
                'last_processed_id': region_id,
                'batch_size': 1000,  # Default batch size
                'updated_at': 'now()'
            }
            response = supabase.table('scraping_progress').insert(data).execute()
            
            if response.data:
                logger.info(f"Inserted new record with last processed region ID: {region_id}")
            else:
                logger.error(f"Failed to insert/update last processed region ID: {region_id}")
    except Exception as e:
        logger.error(f"Error updating last processed region ID: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def get_max_pages(soup):
    """
    Get the maximum number of pages from the pagination element.
    """
    try:
        # Find the pagination element using role='group' and class_='btn-group'
        pagination = soup.find('div', {'role': 'group', 'class': 'btn-group'})
        if pagination:
            # Find the label with "of" and the next label for the max page number
            of_label = pagination.find('label', string='of')
            if of_label and of_label.find_next_sibling('label'):
                max_page = int(of_label.find_next_sibling('label').get_text(strip=True))
                return max_page
            else:
                logger.warning("No page numbers found")
                return 1
        else:
            logger.warning("No pagination element found")
            return 1
    except Exception as e:
        logger.error(f"Error getting max pages: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return 1

def extract_cover_image(property_url):
    """
    Extract the cover image URL from the property page.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(property_url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find the image element with testid="property-photo-0"
            image_element = soup.find('img', {'testid': 'property-photo-0'})
            if image_element and image_element.get('src'):
                return image_element['src']
        return None
    except Exception as e:
        logger.error(f"Error extracting cover image from {property_url}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

def update_property_cover_image(property_id, image_url):
    """
    Update the cover image URL for a property in the database.
    """
    try:
        supabase = create_supabase_client()
        response = supabase.table('properties').update({'cover_image_url': image_url}).eq('id', property_id).execute()
        if response.data:
            logger.info(f"Successfully updated cover image for property ID {property_id}")
        else:
            logger.error(f"Failed to update cover image for property ID {property_id}")
    except Exception as e:
        logger.error(f"Error updating cover image for property ID {property_id}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def fetch_regions(base_url):
    """
    Fetch all regions from the main Wellington page with retry mechanism.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            timeout = 60 + (attempt * 30)
            logger.info(f"Fetching regions (attempt {attempt + 1}/{max_retries}) with {timeout}s timeout")
            
            response = requests.get(base_url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find the region links container
                region_links_container = soup.find('div', {'testid': 'taLinksContainer'})
                if region_links_container:
                    regions = []
                    region_links = region_links_container.find_all('a')
                    for link in region_links:
                        region_name = link.get_text(strip=True)
                        region_href = link.get('href')
                        if region_href:
                            region_url = base_url.rstrip('/') + region_href
                            region_id = link.get('testid', '').replace('ta-link-', '')
                            regions.append({
                                'id': region_id,
                                'name': region_name,
                                'url': region_url
                            })
                    return regions
            return []
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Network error fetching regions on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed to fetch regions after {max_retries} attempts: {e}")
                return []
        except Exception as e:
            logger.error(f"Error fetching regions: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []
    
    return []

def fetch_suburbs(region_url):
    """
    Fetch all suburbs from a region page with enhanced retry mechanism.
    """
    max_retries = 5
    base_timeout = 60
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            timeout = base_timeout + (attempt * 30)
            logger.info(f"Attempting to fetch suburbs (attempt {attempt + 1}/{max_retries}) with {timeout}s timeout")
            
            response = session.get(region_url, timeout=timeout, stream=False)
            response.raise_for_status()
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find the suburb links container
                suburb_links_container = soup.find('div', {'testid': 'suburbLinksContainer'})
                if suburb_links_container:
                    suburbs = []
                    suburb_links = suburb_links_container.find_all('a')
                    for link in suburb_links:
                        suburb_name = link.get_text(strip=True)
                        suburb_href = link.get('href')
                        if suburb_href:
                            base_url = os.getenv("PROPERTY_VALUE_BASE_URL")
                            if not base_url:
                                raise ValueError("PROPERTY_VALUE_BASE_URL environment variable is not set")
                            suburb_url = base_url.rstrip('/') + suburb_href
                            suburb_id = link.get('testid', '').replace('suburb-link-', '')
                            suburbs.append({
                                'id': suburb_id,
                                'name': suburb_name,
                                'url': suburb_url
                            })
                    return suburbs
            return []
            
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            wait_time = 10 * (attempt + 1)
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retries} for {region_url}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed to fetch suburbs after {max_retries} attempts from {region_url}: {e}")
                logger.info("This may be due to external website connectivity issues. The script will continue with other regions.")
                return []
        except Exception as e:
            logger.error(f"Error fetching suburbs from {region_url}: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []
    
    return []

def scrape_properties(suburb_url, city, suburb_name):
    """
    Scrape properties from a suburb page with pagination and retry mechanism.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            timeout = 60 + (attempt * 30)
            logger.info(f"Fetching suburb page (attempt {attempt + 1}/{max_retries}) with {timeout}s timeout: {suburb_name}")
            
            # Get the first page to determine the number of pages
            response = requests.get(suburb_url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to fetch suburb page (attempt {attempt + 1}): {suburb_url}")
                    time.sleep(10 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Failed to fetch suburb page after {max_retries} attempts: {suburb_url}")
                    return
            
            break
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Network error fetching suburb page on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed to fetch suburb page after {max_retries} attempts: {e}")
                return
        except Exception as e:
            logger.error(f"Error scraping properties for suburb {suburb_name}: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return
    
    try:
            
        soup = BeautifulSoup(response.content, 'html.parser')
        max_pages = get_max_pages(soup)
        logger.info(f"Found {max_pages} pages for suburb: {suburb_name}")
        
        # Process each page
        for page in range(1, max_pages + 1):
            logger.info(f"Processing page {page} of {max_pages} for suburb: {suburb_name}")
            property_links, titles = fetch_property_links(suburb_url, page)
            
            # Process each property on the page
            for property_url, title in zip(property_links, titles):
                try:
                    logger.info(f"Fetching details for: {title}")
                    
                    # Fetch property details and history
                    property_data, history_data = fetch_property_details(property_url, title, city, suburb_name)
                    
                    if property_data:
                        # Insert into Supabase
                        insert_property_and_history(property_data, history_data)
                        
                        # Extract and update cover image URL
                        cover_image_url = extract_cover_image(property_url)
                        if cover_image_url:
                            # Note: In a real implementation, you would need to get the property ID
                            # from the database after insertion
                            logger.info(f"Cover image URL for {title}: {cover_image_url}")
                        
                    # Add a small delay to be respectful to the server
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing property {title}: {e}")
                    logger.error(f"Error details: {traceback.format_exc()}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error scraping properties for suburb {suburb_name}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def is_already_running():
    """
    Check if another instance of the PropertyValue Wellington scraper is already running or task status.
    Uses the scraping_progress table to store a lock timestamp.
    """
    supabase = create_supabase_client()
    try:
        # Get the lock timestamp and status for PropertyValue Wellington scraper (id=5)
        response = supabase.table('scraping_progress').select('updated_at, status').eq('id', 5).execute()
        if response.data and len(response.data) > 0:
            status = response.data[0].get('status', 'idle')
            
            # If status is running, exit immediately
            if status == 'running':
                logger.info("Another PropertyValue Wellington scraper instance is already running. Exiting.")
                return True
            
            # Check if task is completed
            if status == 'complete':
                logger.info("PropertyValue Wellington scraper task is completed. No execution needed.")
                return True
            
            # Check if task is manually stopped
            if status == 'stop':
                logger.info("PropertyValue Wellington scraper task is manually stopped. Skipping execution.")
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error checking if PropertyValue Wellington scraper is already running: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # In case of error, assume not running to avoid blocking legitimate runs
        return False

def update_lock_timestamp():
    """
    Update the lock timestamp for PropertyValue Wellington scraper (id=5) to indicate it's still running.
    """
    supabase = create_supabase_client()
    try:
        # Update the updated_at timestamp for PropertyValue Wellington scraper (id=5)
        response = supabase.table('scraping_progress').update({
            'updated_at': 'now()',
            'status': 'running'
        }).eq('id', 5).execute()
        
        if response.data:
            logger.info("PropertyValue Wellington scraper lock timestamp updated successfully.")
        else:
            logger.warning("Failed to update PropertyValue Wellington scraper lock timestamp.")
    except Exception as e:
        logger.error(f"Error updating PropertyValue Wellington scraper lock timestamp: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def mark_complete():
    """
    Mark the PropertyValue Wellington scraper task as complete.
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'complete',
            'updated_at': 'now()'
        }).eq('id', 5).execute()
        
        if response.data:
            logger.info("PropertyValue Wellington scraper task marked as complete.")
        else:
            logger.warning("Failed to mark PropertyValue Wellington scraper task as complete.")
    except Exception as e:
        logger.error(f"Error marking PropertyValue Wellington scraper task as complete: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def main():
    """
    Main function to start the scraping process.
    """
    try:
        # Check if another instance is already running
        if is_already_running():
            logger.info("Another PropertyValue Wellington scraper instance is already running. Exiting.")
            return
        
        # Create progress table if it doesn't exist
        create_scraping_progress_table()
        
        # Update lock timestamp to indicate we're starting
        update_lock_timestamp()
        
        # Get base URL from environment variables and construct Wellington URL
        base_url = os.getenv("PROPERTY_VALUE_BASE_URL")
        if not base_url:
            raise ValueError("PROPERTY_VALUE_BASE_URL environment variable is not set")
        
        wellington_url = base_url + "/wellington"
        
        # Get the last processed region to resume from where we left off
        last_processed_region = get_last_processed_region()
        skip_regions = bool(last_processed_region)
        
        # Fetch all regions
        regions = fetch_regions(wellington_url)
        logger.info(f"Found {len(regions)} regions")
        
        # Process each region
        for region in regions:
            # If we're resuming, skip regions until we reach the last processed one
            if skip_regions and region['id'] != last_processed_region:
                continue
            elif skip_regions and region['id'] == last_processed_region:
                # We've reached the last processed region, stop skipping
                skip_regions = False
                # Don't skip this region, process it again to ensure completion
            
            logger.info(f"Processing region: {region['name']}")
            
            # Fetch all suburbs in this region
            suburbs = fetch_suburbs(region['url'])
            logger.info(f"Found {len(suburbs)} suburbs in region: {region['name']}")
            
            # Process each suburb
            for suburb in suburbs:
                logger.info(f"Processing suburb: {suburb['name']}")
                scrape_properties(suburb['url'], region['name'], suburb['name'])
                
                # Add a delay between suburbs
                time.sleep(2)
            
            # Update progress after processing a region
            update_last_processed_region(region['id'])
            
            # Add a delay between regions
            time.sleep(3)
            
        logger.info("Scraping process completed successfully")
        mark_complete()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error in script execution: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        exit(1)