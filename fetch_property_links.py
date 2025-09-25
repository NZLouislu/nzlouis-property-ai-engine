import requests
from bs4 import BeautifulSoup
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_property_links(main_url, page=1, max_retries=3):
    property_links = []
    titles = []
    
    url = f"{main_url}?page={page}" if page > 1 else main_url
    print(f"Fetching page {page}...")
    
    # 创建一个带有重试机制的会话
    session = requests.Session()
    retries = Retry(total=max_retries, 
                    backoff_factor=0.1, 
                    status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()  # 这将抛出一个异常，如果状态码不是200

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找所有属性链接
            base_url = os.getenv("PROPERTY_VALUE_BASE_URL", "https://propertyvalue.co.nz")
            for link in soup.find_all('a', class_='PropertyCard_PropertyCardLink__icVIl'):
                full_link = base_url + link['href']
                property_links.append(full_link)
                titles.append(link['title'])  # 获取标题属性
            
            print(f"\nFound {len(property_links)} properties on page {page}:")
            # 如果需要打印标题，取消下面的注释
            # for title in titles:
            #     print(title)

        else:
            print(f"Unexpected status code {response.status_code} for URL: {url}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page}: {e}")
    
    finally:
        time.sleep(2)  # 增加延迟到2秒，以避免过度加载服务器
    
    return property_links, titles

# 使用示例
if __name__ == "__main__":
    main_url = "https://propertyvalue.co.nz/wellington/wellington-city/khandallah-6035/200020"
    links, titles = fetch_property_links(main_url)
    print(f"Total properties found: {len(links)}")