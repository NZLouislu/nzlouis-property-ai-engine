import time
import requests
from bs4 import BeautifulSoup
from fetch_property_links import fetch_property_links
from properties import fetch_property_details
from config.supabase_config import create_supabase_client, insert_property_and_history
import traceback
import sys
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("property_image_updater.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# 创建日志记录器
logger = logging.getLogger(__name__)

# Main function to scrape properties

def fetch_suburbs(url, city):
    """
    Fetches the list of suburbs and their links from a given URL.
    """
    
    # Get base URL from environment variables
    base_url = os.getenv("PROPERTY_VALUE_BASE_URL")
    if not base_url:
        raise ValueError("PROPERTY_VALUE_BASE_URL environment variable is not set")
    
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        suburb_links_container = soup.find('div', {'testid': 'suburbLinksContainer'})
        if suburb_links_container:
            suburb_links = suburb_links_container.find_all('a')
            # Reverse the order of links
            for link in suburb_links:
                suburb_name = link.get_text(strip=True)
                # Use base URL from environment variables
                suburb_link = base_url + link.get('href')                
                print(f"Suburb: {suburb_name}, Link: {suburb_link}")
                
                # Fetch the page content for the suburb link
                suburb_response = requests.get(suburb_link)
                print(f"  Status code for {suburb_name}: {suburb_response.status_code}")
                if suburb_response.status_code == 200:
                    suburb_soup = BeautifulSoup(suburb_response.content, 'html.parser')
                    # Find the pagination element using role='group' and class_='btn-group'
                    pagination = suburb_soup.find('div', {'role': 'group', 'class': 'btn-group'})
                    if pagination:
                        # Find the label with "of" and the next label for the max page number
                        of_label = pagination.find('label', string='of')
                        if of_label and of_label.find_next_sibling('label'):
                            max_page = int(of_label.find_next_sibling('label').get_text(strip=True))
                            print(f"Suburb: {suburb_name}, Max Pages: {max_page}")
                        else:
                            print(f"  No page numbers found for {suburb_name}")
                    else:
                        print(f"  No pagination element found for {suburb_name}")
                        max_page = 1 # Default to 1 page if no pagination

                    scrape_properties(suburb_link, max_page, city, suburb_name)

def scrape_properties(main_url, max_pages, city, suburb):
    for page in range(1, max_pages + 1):
        # Fetch property links and titles for the current page
        property_links, titles = fetch_property_links(main_url, page)
        
        # Print and fetch details for each property on the current page
        for property_url, title in zip(property_links, titles):
            print(f"Fetching details for: {title}")
            
            # Fetch property details and history
            property_data, history_data = fetch_property_details(property_url, title, city, suburb)
            
            # Insert into Supabase
            insert_property_and_history(property_data, history_data)
            
            # time.sleep(0.5)  # Adding a delay to avoid overloading the server

# Function to create scraping progress table if it doesn't exist
def create_scraping_progress_table():
    """
    Create a table to track scraping progress.
    """
    supabase = create_supabase_client()
    try:
        # Check if the scraping_progress table exists by trying to select from it
        response = supabase.table('scraping_progress').select('*').limit(1).execute()
        print("Scraping progress table already exists.")
    except Exception as e:
        print(f"Scraping progress table may not exist: {e}")
        print("Please ensure the scraping_progress table exists in your Supabase database with the following structure:")
        print("- id (int, primary key)")
        print("- last_processed_id (text)")
        print("- batch_size (int)")
        print("- updated_at (timestamp)")

# Function to get the last processed ID from the progress table
def get_last_processed_id():
    """
    Get the last processed property ID from the progress table.
    """
    supabase = create_supabase_client()
    try:
        # First, try to get the record with id=1 for image updater
        response = supabase.table('scraping_progress').select('last_processed_id').eq('id', 1).execute()
        if response.data and len(response.data) > 0:
            last_processed_id = response.data[0].get('last_processed_id')
            # Return the ID if it's not None or empty
            if last_processed_id:
                logger.info(f"Resuming from ID: {last_processed_id}")
                return last_processed_id
            else:
                # If last_processed_id is None or empty, it means we're starting from the beginning
                logger.info("Starting from the beginning (empty last_processed_id)")
                return None
        
        # If no record with id=1, try to get the latest record
        response = supabase.table('scraping_progress').select('last_processed_id').order('id', desc=True).limit(1).execute()
        if response.data and len(response.data) > 0:
            last_processed_id = response.data[0].get('last_processed_id')
            # Return the ID if it's not None or empty
            if last_processed_id:
                logger.info(f"Resuming from ID: {last_processed_id}")
                return last_processed_id
            else:
                # If last_processed_id is None or empty, it means we're starting from the beginning
                logger.info("Starting from the beginning (no valid last_processed_id)")
                return None
            
        logger.info("Starting from the beginning (no progress records found)")
        return None
    except Exception as e:
        logger.error(f"Error getting last processed ID: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

# Function to update the last processed ID in the progress table
def update_last_processed_id(last_id):
    """
    Update the last processed property ID in the progress table.
    """
    supabase = create_supabase_client()
    try:
        # First, try to update the existing record with id=1 for image updater
        response = supabase.table('scraping_progress').update({
            'last_processed_id': last_id,
            'batch_size': 1000,  # Default batch size
            'updated_at': 'now()'
        }).eq('id', 1).execute()
        
        # Check if the update was successful
        if response.data:
            logger.info(f"Updated last processed ID to: {last_id}")
        else:
            # If no record was updated, insert a new one with id=1
            data = {
                'id': 1,  # Use ID 1 for the image updater progress
                'last_processed_id': last_id,
                'batch_size': 1000,  # Default batch size
                'updated_at': 'now()'
            }
            response = supabase.table('scraping_progress').insert(data).execute()
            
            if response.data:
                logger.info(f"Inserted new record with last processed ID: {last_id}")
            else:
                logger.error(f"Failed to insert/update last processed ID: {last_id}")
    except Exception as e:
        logger.error(f"Error updating last processed ID: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

# Function to check if another instance is already running
def is_already_running():
    """
    Check if another instance of the image updater is already running or task status.
    Uses the scraping_progress table to store a lock timestamp.
    """
    supabase = create_supabase_client()
    try:
        # Get the lock timestamp for image updater (id=1)
        response = supabase.table('scraping_progress').select('updated_at, status').eq('id', 1).execute()
        if response.data and len(response.data) > 0:
            updated_at_str = response.data[0].get('updated_at')
            status = response.data[0].get('status', 'idle')
            
            # Check if task is completed
            if status == 'complete':
                logger.info("Task is completed. No execution needed.")
                return True
            
            # Check if task is manually stopped
            if status == 'stop':
                logger.info("Task is manually stopped. Skipping execution.")
                return True
            
            # Check if another instance is running
            if updated_at_str and status == 'running':
                # Parse the timestamp
                from datetime import datetime, timezone
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                # Check if the lock is still valid (less than 30 minutes old for active running status)
                current_time = datetime.now(timezone.utc)
                time_diff = current_time - updated_at
                if time_diff.total_seconds() < 30 * 60:  # 30 minutes in seconds
                    logger.info("Another instance is actively running. Skipping execution.")
                    return True
                else:
                    # Lock is stale, clear it
                    logger.info("Found stale lock, clearing it.")
                    clear_lock()
        return False
    except Exception as e:
        logger.error(f"Error checking if already running: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False

def update_lock_timestamp():
    """
    Update the lock timestamp to indicate the process is running.
    """
    supabase = create_supabase_client()
    try:
        # Update the updated_at timestamp and status for image updater (id=1)
        response = supabase.table('scraping_progress').update({
            'updated_at': 'now()',
            'status': 'running'
        }).eq('id', 1).execute()
        
        if not response.data:
            # If no record exists, create one
            data = {
                'id': 1,
                'last_processed_id': None,
                'batch_size': 1000,
                'updated_at': 'now()',
                'status': 'running'
            }
            response = supabase.table('scraping_progress').insert(data).execute()
            
        logger.info("Lock timestamp updated successfully.")
    except Exception as e:
        logger.error(f"Error updating lock timestamp: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def clear_lock():
    """
    Clear the lock to indicate the process has paused.
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'idle',
            'updated_at': 'now()'
        }).eq('id', 1).execute()
        
        logger.info("Lock cleared successfully, status set to idle.")
    except Exception as e:
        logger.error(f"Error clearing lock: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def mark_complete():
    """
    Mark the task as completed.
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'complete',
            'updated_at': 'now()'
        }).eq('id', 1).execute()
        
        logger.info("Task marked as complete.")
    except Exception as e:
        logger.error(f"Error marking task as complete: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def trigger_next_workflow():
    """
    Trigger the next workflow run using GitHub API.
    """
    import os
    github_token = os.getenv('GITHUB_TOKEN')
    github_repo = os.getenv('GITHUB_REPOSITORY')
    
    if not github_token or not github_repo:
        logger.warning("GitHub token or repository not available. Cannot trigger next workflow.")
        return
    
    try:
        import requests
        
        url = f"https://api.github.com/repos/{github_repo}/actions/workflows/update_property_image.yml/dispatches"
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        data = {
            'ref': 'main'
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 204:
            logger.info("Successfully triggered next workflow run.")
        else:
            logger.warning(f"Failed to trigger next workflow. Status: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error triggering next workflow: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

# Function to fetch properties from Supabase and update cover images with pagination
def update_property_images(batch_size=1000, max_runtime_hours=5.5):
    """
    Fetch properties from Supabase that don't have cover_image_url,
    scrape the property page to get the image URL, and update the record.
    Implements pagination and progress tracking to work within GitHub Actions time limits.
    
    Args:
        batch_size (int): Number of properties to process in each batch
        max_runtime_hours (float): Maximum runtime in hours before stopping (default 5.5 hours to stay within 6-hour limit)
    """
    # Check if another instance is already running
    if is_already_running():
        logger.info("Another instance is already running. Exiting.")
        return
    
    # Update lock timestamp to indicate we're running
    update_lock_timestamp()
    
    should_continue = True
    
    supabase = create_supabase_client()
    start_time = time.time()
    max_runtime_seconds = max_runtime_hours * 3600  # Convert hours to seconds
    processed_count = 0
    last_id_in_batch = None
    
    try:
        # Get the last processed ID to resume from where we left off
        last_processed_id = get_last_processed_id()
        
        # Counter for processed properties
        processed_count = 0
        
        # Log start of process
        logger.info(f"Starting property image update process. Max runtime: {max_runtime_hours} hours")
        
        # Flag to track if we have data to process
        has_data_to_process = False
        
        while True:
            # Check if we've exceeded the maximum runtime
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                logger.info(f"Maximum runtime of {max_runtime_hours} hours reached. Stopping...")
                # Save progress before exiting
                if last_id_in_batch:
                    update_last_processed_id(last_id_in_batch)
                should_continue = True
                break
            
            # Update lock timestamp periodically to indicate we're still running
            update_lock_timestamp()
            
            # Fetch properties that don't have cover_image_url or have null cover_image_url
            # Use pagination and ordering to process in batches
            try:
                query = supabase.table('properties').select('id, property_url, cover_image_url')
                query = query.is_('cover_image_url', 'null')
                
                # If we have a last processed ID, start from the next record
                if last_processed_id:
                    query = query.gt('id', last_processed_id)
                
                # Order by ID and limit to batch size
                response = query.order('id').limit(batch_size).execute()
            except Exception as e:
                logger.error(f"Error querying properties from Supabase: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")
                # Save progress before exiting
                if last_id_in_batch:
                    update_last_processed_id(last_id_in_batch)
                break
            
            if not response.data or len(response.data) == 0:
                logger.info("No more properties found without cover images.")
                should_continue = False
                break
                
            # If we have data, set the flag
            has_data_to_process = True
            logger.info(f"Found {len(response.data)} properties without cover images in this batch.")
            
            last_id_in_batch = None
            for property_record in response.data:
                # Check if we've exceeded the maximum runtime
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime_seconds:
                    logger.info(f"Maximum runtime of {max_runtime_hours} hours reached. Stopping...")
                    # Save progress before exiting
                    if last_id_in_batch:
                        update_last_processed_id(last_id_in_batch)
                    should_continue = True
                    break
                
                # Update lock timestamp periodically to indicate we're still running
                update_lock_timestamp()
                
                property_id = property_record['id']
                last_id_in_batch = property_id
                property_url = property_record['property_url']
                
                if not property_url:
                    logger.warning(f"Property ID {property_id} has no property_url. Skipping...")
                    continue
                    
                logger.info(f"Processing property ID {property_id}: {property_url}")
                
                try:
                    # Fetch the property page
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    page_response = requests.get(property_url, headers=headers, timeout=30)
                    
                    if page_response.status_code != 200:
                        logger.warning(f"Failed to fetch property page for ID {property_id}. Status code: {page_response.status_code}")
                        # Log error details for debugging
                        logger.debug(f"Response headers: {page_response.headers}")
                        continue
                        
                    # Parse the HTML content
                    soup = BeautifulSoup(page_response.content, 'html.parser')
                    
                    # Find the image element with testid="property-photo-0"
                    image_element = soup.find('img', {'testid': 'property-photo-0'})
                    
                    if image_element and image_element.get('src'):
                        image_url = image_element['src']
                        logger.info(f"Found image for property ID {property_id}: {image_url}")
                        
                        # Update the property record with the image URL
                        try:
                            update_response = supabase.table('properties').update({'cover_image_url': image_url}).eq('id', property_id).execute()
                            
                            if update_response.data:
                                logger.info(f"Successfully updated cover image for property ID {property_id}")
                            else:
                                logger.error(f"Failed to update cover image for property ID {property_id}")
                                logger.debug(f"Update response: {update_response}")
                        except Exception as update_e:
                            logger.error(f"Error updating cover image for property ID {property_id}: {str(update_e)}")
                            logger.error(f"Update error details: {traceback.format_exc()}")
                    else:
                        logger.warning(f"⚠️ No image found for property ID {property_id}")
                        
                except Exception as e:
                    logger.error(f"Error processing property ID {property_id}: {str(e)}")
                    logger.error(f"Error details: {traceback.format_exc()}")
                    # Continue with the next property instead of stopping
                    continue
                    
                # Update progress
                processed_count += 1
                
                # Add a small delay to be respectful to the server
                time.sleep(1)
            
            # Update the last processed ID after completing a batch
            if last_id_in_batch:
                update_last_processed_id(last_id_in_batch)
            
            # If we processed fewer properties than the batch size, we've reached the end
            if len(response.data) < batch_size:
                logger.info("Processed all available properties in this batch.")
                should_continue = False
                break
                
        logger.info(f"Finished processing. Total properties processed: {processed_count}")
        
        # Set appropriate status based on completion
        if should_continue:
            # Due to time limit, need to continue - set to idle for next run
            clear_lock()
            logger.info("Triggering next workflow run to continue processing...")
            trigger_next_workflow()
        else:
            # Task completely finished - mark as complete
            mark_complete()
            logger.info("All image updates completed. Task status set to complete.")
            
    except Exception as e:
        logger.error(f"Error fetching properties from Supabase: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # Save progress before exiting
        if last_id_in_batch:
            update_last_processed_id(last_id_in_batch)
        # Clear the lock on error
        clear_lock()
        # Re-raise the exception so it's visible in GitHub Actions logs
        raise

# Run the scraper
if __name__ == "__main__":
    try:
        # Check if user wants to update images or scrape new properties
        if len(sys.argv) > 1 and sys.argv[1] == 'update_images':
            logger.info("Starting property image update process")
            # Create progress table if it doesn't exist
            create_scraping_progress_table()
            # Update property images with default batch size and time limit
            update_property_images()
            logger.info("Property image update process completed")
        else:
            logger.info("Starting property scraping process")
            # Get base URL from environment variables and construct Wellington URL
            base_url = os.getenv("PROPERTY_VALUE_BASE_URL")
            if not base_url:
                raise ValueError("PROPERTY_VALUE_BASE_URL environment variable is not set")
            
            wellington_url = base_url + "/wellington/wellington-city/47"
            city = "Wellington City"
            fetch_suburbs(wellington_url, city)
            logger.info("Property scraping process completed")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        sys.exit(1)
