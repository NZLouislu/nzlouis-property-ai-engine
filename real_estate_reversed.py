from playwright.sync_api import sync_playwright, TimeoutError
import time
import random

from config.supabase_config import insert_real_estate, create_supabase_client

def handle_dialog(dialog):
    print(f"Dialog message: {dialog.message}")
    dialog.accept()

def scroll_to_bottom(page):
    print("开始模拟鼠标下滑操作...")
    last_height = page.evaluate("document.body.scrollHeight")
    while True:
        print(f"  - 当前页面高度: {last_height}，继续下滑...")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(random.uniform(1, 2))  # 等待页面加载
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            print("  - 已到达页面底部")
            break
        last_height = new_height

        # 检查是否出现了页码导航
        if page.query_selector('nav[aria-label="Pagination"]') or page.query_selector('div[class*="pagination"]'):
            print("  - 检测到页码导航，停止滚动")
            break

def simulate_user_behavior(page):
    scroll_to_bottom(page)
    
    # 随机点击几个房产卡片
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
                except:
                    pass
            break

    # 额外的滚动操作
    print("模拟额外的滚动操作")
    for i in range(10):
        scroll_distance = random.randint(500, 1500)
        page.evaluate(f"window.scrollBy(0, {scroll_distance})")
        print(f"  - 向下滚动 {scroll_distance} 像素")
        time.sleep(random.uniform(1, 2))

    # 再次滚动到底部
    print("再次滚动到页面底部")
    scroll_to_bottom(page)

def fetch_addresses(page, url):
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
    except TimeoutError:
        print(f"Timeout while loading {url}. Continuing with partial page load.")

    try:
        page.wait_for_selector('button:has-text("Accept")', timeout=5000)
        page.click('button:has-text("Accept")')
        print("Clicked cookie consent button.")
    except:
        print("No cookie consent button found or unable to click it.")

    # 模拟用户行为
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
            address_elements = page.query_selector_all(selector)
            if address_elements:
                addresses = [element.inner_text().strip() for element in address_elements if element.inner_text().strip()]
                print(f"Found {len(addresses)} addresses using selector: {selector}")
                break
        
        if not addresses:
            print(f"No address elements found on {url} using any of the selectors.")
            print("Page Title:", page.title())
            print("Current URL:", page.url)
            print("HTML content:", page.content()[:1000])
    except Exception as e:
        print(f"An error occurred while scraping {url}: {str(e)}")

    return addresses

def check_real_estate_in_supabase(address: str) -> bool:
    """
    Check if a real estate property already exists in the Supabase database.
    """
    try:
        supabase = create_supabase_client()
        response = supabase.table('real_estate').select('id').eq('address', address).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            print(f"Property already exists in database: {address}")
            return True
        return False
    except Exception as e:
        print(f"Error checking property in database: {e}")
        # If there's an error checking, we assume it doesn't exist to avoid missing data
        return False

def scrape_properties(main_url, max_pages):
    all_addresses = []

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

        for page_num in range(max_pages, 0, -1):
            url = f"{main_url}?page={page_num}"
            print(f"\nScraping page {page_num}: {url}")
            
            addresses = fetch_addresses(page, url)
            if addresses:
                all_addresses.extend(addresses)
                print(f"Found {len(addresses)} addresses on page {page_num}")
                print("Addresses found on this page:")
                for addr in addresses:
                    print(f"  - {addr}")
                    if not check_real_estate_in_supabase(addr):
                        # 插入到Supabase中
                        insert_real_estate(addr, "for Sale")  # 假设状态为 "For Sale"
                        print(f"Added new property to database: {addr}")
                    else:
                        print(f"Property {addr} already exists in database. Skipping...")
            else:
                print(f"No addresses found on page {page_num}. Continuing to next page.")
            
            if page_num < max_pages:
                delay = random.uniform(5, 10)
                print(f"Waiting for {delay:.2f} seconds before next request...")
                time.sleep(delay)

        browser.close()

def main():
    main_url = "https://www.realestate.co.nz/residential/sale/auckland"
    max_pages = 500
    scrape_properties(main_url, max_pages)

if __name__ == "__main__":
    main()