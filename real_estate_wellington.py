from playwright.sync_api import sync_playwright, TimeoutError
import time
import random
import os
from dotenv import load_dotenv
import traceback
import logging

# Load environment variables
load_dotenv()

from config.supabase_config import insert_real_estate, create_supabase_client

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_estate_wellington.log"),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger(__name__)

# Function to create scraping progress table if it doesn't exist
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

# Function to get the last processed page from the progress table
def get_last_processed_page():
    """
    Get the last processed page number from the progress table for Wellington real estate.
    """
    supabase = create_supabase_client()
    try:
        # Try to get the record with id=3 for Wellington real estate
        response = supabase.table('scraping_progress').select('last_processed_id').eq('id', 3).execute()
        if response.data and len(response.data) > 0:
            last_processed_id = response.data[0].get('last_processed_id')
            # Return the page number if it's not None or empty
            if last_processed_id:
                page_num = int(last_processed_id)
                logger.info(f"Resuming from page: {page_num}")
                return page_num
            else:
                # If last_processed_id is None or empty, it means we're starting from the beginning
                logger.info("Starting from the beginning (empty last_processed_id)")
                return 0
        
        # If no record with id=3, it means we're starting from the beginning
        logger.info("Starting from the beginning (no progress records found for Wellington)")
        return 0
    except Exception as e:
        logger.error(f"Error getting last processed page: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return 0

# Function to update the last processed page in the progress table
def update_last_processed_page(last_page):
    """
    Update the last processed page number in the progress table for Wellington real estate.
    """
    supabase = create_supabase_client()
    try:
        # First, try to update the existing record with id=3 for Wellington real estate
        response = supabase.table('scraping_progress').update({
            'last_processed_id': str(last_page),
            'batch_size': 1000,  # Default batch size
            'updated_at': 'now()'
        }).eq('id', 3).execute()
        
        # Check if the update was successful
        if response.data:
            logger.info(f"Updated last processed page to: {last_page}")
        else:
            # If no record was updated, insert a new one with id=3
            data = {
                'id': 3,  # Use ID 3 for Wellington real estate progress
                'last_processed_id': str(last_page),
                'batch_size': 1000,  # Default batch size
                'updated_at': 'now()'
            }
            response = supabase.table('scraping_progress').insert(data).execute()
            
            if response.data:
                logger.info(f"Inserted new record with last processed page: {last_page}")
            else:
                logger.error(f"Failed to insert/update last processed page: {last_page}")
    except Exception as e:
        logger.error(f"Error updating last processed page: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

# Function to check if another instance is already running
def is_already_running():
    """
    Check if another instance of the Wellington scraper is already running or task status.
    Uses the scraping_progress table to store a lock timestamp.
    """
    supabase = create_supabase_client()
    try:
        # Get the lock timestamp and status for Wellington scraper (id=3)
        response = supabase.table('scraping_progress').select('updated_at, status').eq('id', 3).execute()
        if response.data and len(response.data) > 0:
            updated_at_str = response.data[0].get('updated_at')
            status = response.data[0].get('status', 'idle')
            
            if status == 'complete':
                logger.info("Wellington scraper task is completed. No execution needed.")
                return True
            
            if status == 'stop':
                logger.info("Wellington scraper task is manually stopped. Skipping execution.")
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
                    logger.info("Another Wellington scraper instance is already running. Exiting.")
                    return True
                else:
                    # Lock is stale, clear it
                    logger.info("Found stale lock, clearing it")
                    clear_lock()
                
        return False
    except Exception as e:
        logger.error(f"Error checking if Wellington scraper is already running: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # In case of error, assume not running to avoid blocking legitimate runs
        return False

def update_lock_timestamp():
    """
    Update the lock timestamp to indicate the process is running.
    """
    supabase = create_supabase_client()
    try:
        # Update the updated_at timestamp for Wellington scraper (id=3)
        response = supabase.table('scraping_progress').update({
            'updated_at': 'now()',
            'status': 'running'
        }).eq('id', 3).execute()
        
        if response.data:
            logger.info("Wellington scraper lock timestamp updated successfully.")
        else:
            logger.warning("Failed to update Wellington scraper lock timestamp.")
    except Exception as e:
        logger.error(f"Error updating Wellington scraper lock timestamp: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def mark_complete():
    """
    Mark the Wellington scraper task as complete.
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'complete',
            'updated_at': 'now()'
        }).eq('id', 3).execute()
        
        if response.data:
            logger.info("Wellington scraper task marked as complete.")
        else:
            logger.warning("Failed to mark Wellington scraper task as complete.")
    except Exception as e:
        logger.error(f"Error marking Wellington scraper task as complete: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def clear_lock():
    """
    Clear the lock to indicate the process has paused or stopped due to an error.
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'idle',
            'updated_at': 'now()'
        }).eq('id', 3).execute()
        
        logger.info("Lock cleared successfully, status set to idle.")
    except Exception as e:
        logger.error(f"Error clearing lock: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def handle_dialog(dialog):
    """
    Handle dialog boxes that may appear during scraping.
    """
    try:
        print(f"Dialog message: {dialog.message}")
        dialog.accept()
    except Exception as e:
        logger.error(f"Error handling dialog: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def scroll_to_bottom(page):
    """
    Scroll to the bottom of the page to load all content.
    """
    try:
        print("开始模拟鼠标下滑操作...")
        last_height = page.evaluate("document.body.scrollHeight")
        while True:
            print(f"  - 当前页面高度: {last_height}，继续下滑...")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(1, 2))  # Wait for page to load
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                print("  - 已到达页面底部")
                break
            last_height = new_height

            # Check if pagination navigation appeared
            if page.query_selector('nav[aria-label="Pagination"]') or page.query_selector('div[class*="pagination"]'):
                print("  - 检测到页码导航，停止滚动")
                break
    except Exception as e:
        logger.error(f"Error scrolling to bottom: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def simulate_user_behavior(page):
    """
    Simulate user behavior to avoid anti-scraping mechanisms.
    """
    try:
        scroll_to_bottom(page)
        
        # Randomly click on property cards
        print("模拟查看房产卡片...")
        card_selectors = [
            'div[class*="listing-tile"]',
            'div[class*="property-card"]',
            'div[class*="search-result"]'
        ]
        for selector in card_selectors:
            cards = page.query_selector_all(selector)
            if cards:
                for _ in range(min(3, len(cards))):
                    card = random.choice(cards)
                    try:
                        card.scroll_into_view_if_needed()
                        card.hover()
                        print(f"  - 悬停在一个房产卡片上")
                        time.sleep(random.uniform(0.5, 1.5))
                    except Exception as e:
                        logger.warning(f"Error hovering over card: {e}")
                        pass
                break

        # Additional scrolling operations
        print("模拟额外的滚动操作")
        for i in range(10):
            scroll_distance = random.randint(500, 1500)
            page.evaluate(f"window.scrollBy(0, {scroll_distance})")
            print(f"  - 向下滚动 {scroll_distance} 像素")
            time.sleep(random.uniform(1, 2))

        # Scroll to bottom again
        print("再次滚动到页面底部")
        scroll_to_bottom(page)
    except Exception as e:
        logger.error(f"Error simulating user behavior: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

def fetch_addresses(page, url):
    """
    Fetch addresses from the given URL.
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
    except TimeoutError as e:
        logger.warning(f"Timeout while loading {url}. Continuing with partial page load. Error: {e}")
    except Exception as e:
        logger.error(f"Error navigating to {url}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return []

    try:
        page.wait_for_selector('button:has-text("Accept")', timeout=5000)
        page.click('button:has-text("Accept")')
        print("Clicked cookie consent button.")
    except Exception as e:
        logger.info("No cookie consent button found or unable to click it.")
        pass

    # Simulate user behavior
    simulate_user_behavior(page)

    addresses = []
    try:
        selectors = [
            'h3[data-test="standard-tile__search-result__address"]',
            '.standard-tile__search-result__address',
            'h3[class*="address"]',
            'div[class*="address"]',
            'div[class*="listing-tile"] h3',
            'div[class*="listing-tile"] div[class*="address"]'
        ]
        
        for selector in selectors:
            try:
                address_elements = page.query_selector_all(selector)
                if address_elements:
                    addresses = [element.inner_text().strip() for element in address_elements if element.inner_text().strip()]
                    if addresses:
                        print(f"Found {len(addresses)} addresses using selector: {selector}")
                        break
            except Exception as e:
                logger.warning(f"Error using selector {selector}: {e}")
                continue
        
        if not addresses:
            logger.warning(f"No address elements found on {url} using any of the selectors.")
            print("Page Title:", page.title())
            print("Current URL:", page.url)
            # Only log first 1000 characters of content to avoid huge logs
            print("HTML content:", page.content()[:1000])
    except Exception as e:
        logger.error(f"An error occurred while scraping {url}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")

    return addresses

def check_real_estate_in_supabase(address: str) -> bool:
    """
    Check if a real estate property already exists in the Supabase database.
    """
    try:
        supabase = create_supabase_client()
        response = supabase.table('real_estate').select('id').eq('address', address).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Property already exists in database: {address}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking property in database: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # If there's an error checking, we assume it doesn't exist to avoid missing data
        return False

def scrape_properties(main_url, max_pages, max_runtime_hours=5.5):
    """
    Scrape properties with progress tracking and time limit.
    
    Args:
        main_url (str): The base URL to scrape
        max_pages (int): Maximum number of pages to scrape
        max_runtime_hours (float): Maximum runtime in hours before stopping (default 5.5 hours)
    """
    # Check if another instance is already running
    if is_already_running():
        logger.info("Another Wellington scraper instance is already running. Exiting.")
        return
    
    # Update lock timestamp to indicate we're running
    update_lock_timestamp()
    
    all_addresses = []
    
    # Create progress table if it doesn't exist
    create_scraping_progress_table()
    
    # Get the last processed page to resume from where we left off
    start_page = get_last_processed_page()
    
    # If we've reached the end, reset to start from beginning
    if start_page >= max_pages:
        logger.info(f"Reached end of pages ({start_page}), resetting to start from beginning")
        start_page = 0
        update_last_processed_page(0)
    
    # Calculate maximum runtime
    start_time = time.time()
    max_runtime_seconds = max_runtime_hours * 3600  # Convert hours to seconds
    
    browser = None
    context = None
    page = None
    
    # Log start of process
    logger.info(f"Starting Wellington property scraping process. Max runtime: {max_runtime_hours} hours")
    
    # Flag to track if we have data to process
    has_data_to_process = False
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = context.new_page()
            page.on("dialog", handle_dialog)

            for page_num in range(start_page + 1, max_pages + 1):
                # Check if we've exceeded the maximum runtime
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime_seconds:
                    logger.info(f"Maximum runtime of {max_runtime_hours} hours reached. Stopping...")
                    # Save progress before exiting
                    update_last_processed_page(page_num - 1)
                    # Update status to 'stop' to indicate timeout
                    supabase = create_supabase_client()
                    try:
                        supabase.table('scraping_progress').update({
                            'status': 'stop',
                            'updated_at': 'now()'
                        }).eq('id', 3).execute()
                        logger.info("Wellington scraper status updated to 'stop' due to timeout.")
                    except Exception as e:
                        logger.error(f"Error updating status to 'stop': {e}")
                    break
                
                # Update lock timestamp periodically to indicate we're still running
                update_lock_timestamp()
                
                try:
                    url = f"{main_url}?page={page_num}"
                    print(f"\nScraping page {page_num}: {url}")
                    
                    addresses = fetch_addresses(page, url)
                    if addresses:
                        # If we have data, set the flag
                        has_data_to_process = True
                        all_addresses.extend(addresses)
                        print(f"Found {len(addresses)} addresses on page {page_num}")
                        print("Addresses found on this page:")
                        for addr in addresses:
                            print(f"  - {addr}")
                            try:
                                # Use Supabase to check for duplicates instead of Redis
                                if not check_real_estate_in_supabase(addr):
                                    # Insert into Supabase
                                    insert_real_estate(addr, "for Sale")  # Assume status is "For Sale"
                                    # No need to add to Redis anymore
                                    print(f"Added new property to database: {addr}")
                                else:
                                    print(f"Property {addr} already exists in database. Skipping...")
                            except Exception as e:
                                logger.error(f"Error processing address {addr}: {e}")
                                logger.error(f"Error details: {traceback.format_exc()}")
                                # Continue with next address instead of stopping
                                continue
                    else:
                        print(f"No addresses found on page {page_num}. Continuing to next page.")
                    
                    # Update progress after successfully processing a page
                    update_last_processed_page(page_num)
                    
                    if page_num < max_pages:
                        delay = random.uniform(5, 10)
                        print(f"Waiting for {delay:.2f} seconds before next request...")
                        time.sleep(delay)
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    logger.error(f"Error details: {traceback.format_exc()}")
                    # Save progress and continue with next page instead of stopping
                    update_last_processed_page(page_num)
                    continue

            # If we've processed all pages but still have time, continue running
            # But only if we have processed data
            while has_data_to_process and (time.time() - start_time) < max_runtime_seconds:
                logger.info("Processed all available pages. Waiting before next cycle.")
                time.sleep(60)  # Wait for 1 minute before checking again
                update_lock_timestamp()  # Update lock timestamp to indicate we're still running

        # If we finished normally (not due to timeout), mark as complete
        if elapsed_time <= max_runtime_seconds:
            mark_complete()
            
    except Exception as e:
        logger.error(f"Error in scraping process: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # Save progress before exiting
        if 'page_num' in locals():
            update_last_processed_page(page_num)
        # Clear the lock when there's an error
        clear_lock()
        raise
    finally:
        # If we haven't processed any data, we can exit early
        if not has_data_to_process:
            logger.info("No data to process. Stopping early.")
        # Browser will be automatically closed by the context manager
        logger.info("Browser context closed automatically")

def force_clear_lock():
    """
    Force clear lock to allow script to run.
    """
    supabase = create_supabase_client()
    try:
        from datetime import datetime, timezone
        # Set timestamp to 2 hours ago to ensure it's considered expired
        old_time = datetime.now(timezone.utc).replace(hour=datetime.now(timezone.utc).hour-2)
        supabase.table('scraping_progress').update({
            'updated_at': old_time.isoformat(),
            'status': 'idle'
        }).eq('id', 3).execute()
        logger.info("Force cleared Wellington scraper lock")
        return True
    except Exception as e:
        logger.error(f"Error force clearing lock: {e}")
        return False

def main():
    """
    Main function to start the scraping process.
    """
    try:
        # Read base URL from environment variables and append /wellington
        base_url = os.getenv("REALESTATE_URL")
        if not base_url:
            raise ValueError("REALESTATE_URL environment variable is not set")
        main_url = f"{base_url}/wellington"
        
        max_pages = 99
        scrape_properties(main_url, max_pages)
        logger.info("Scraping process completed successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # Clear the lock when there's an error
        clear_lock()
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error in script execution: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        # Clear the lock on error in main execution
        clear_lock()
        exit(1)