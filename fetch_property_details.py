import requests
from bs4 import BeautifulSoup

# 固定的城市和 suburb
CITY = "Porirua City"
SUBURB = "Aotea"

# Step 2: Fetch details for each property
def fetch_property_details(property_url, title):
    print(f"\nFetching details for {property_url}...")
    response = requests.get(property_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extracting property details
        address_line1 = title.split(',')[0].strip()  # 从 title 中提取 address_line1
        address_line2 = soup.find('span', {'testid': 'addressLine2'}).get_text(strip=True) if soup.find('span', {'testid': 'addressLine2'}) else 'N/A'
        postcode = title.split(',')[-1].strip()  # 获取最后一个逗号后的邮政编码

        # Combine address_line1 and address_line2 into a single address field
        address = f"{address_line1}, {address_line2}"

        suburb = SUBURB
        city = CITY

        try:
            year_built = int(soup.find('div', {'testid': 'yearBuiltValue'}).get_text(strip=True))
        except (AttributeError, ValueError):
            year_built = None

        try:
            bedrooms = int(soup.find('span', {'testid': 'bed'}).get_text(strip=True))
        except (AttributeError, ValueError):
            bedrooms = None
        
        try:
            bathrooms = int(soup.find('span', {'testid': 'bath'}).get_text(strip=True))
        except (AttributeError, ValueError):
            bathrooms = None

        try:
            car_spaces = int(soup.find('span', {'testid': 'car'}).get_text(strip=True))
        except (AttributeError, ValueError):
            car_spaces = None

        try:
            floor_size = soup.find('span', class_='floor PropertyAttributes_attribute__3bkWm').get_text(strip=True)
        except AttributeError:
            floor_size = 'N/A'
        
        try:
            land_area = soup.find('span', class_='land PropertyAttributes_attribute__3bkWm').get_text(strip=True)
        except AttributeError:
            land_area = 'N/A'

        last_sold_price, last_sold_date = parse_sold_details(soup)

        capital_value = extract_value(soup, 'Capital Value')
        land_value = extract_value(soup, 'Land Value')
        improvement_value = extract_value(soup, 'Improvement Value')

        # Fetch rental history (this is where rental status is determined)
        rental_history = fetch_rental_history(soup)

        # Storing and printing the property data
        property_data = {
            'property_url': property_url,
            'address': address,
            'suburb': suburb,
            'city': city,
            'postcode': postcode,
            'year_built': year_built,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'car_spaces': car_spaces,
            'floor_size': floor_size,
            'land_area': land_area,
            'last_sold_price': last_sold_price,
            'last_sold_date': last_sold_date,
            'capital_value': capital_value,
            'land_value': land_value,
            'improvement_value': improvement_value,
            'rental_history': rental_history['history'],
            'has_rental_history': rental_history['has_rental_history'],
            'is_currently_rented': rental_history['is_currently_rented']
        }

        # 打印房产详情
        for key, value in property_data.items():
            print(f"{key}: {value}")

    else:
        print(f"Failed to fetch details for: {property_url}")

# Step 3: Parse sold details (remove 'Last Sold on' and keep the correct date and price)
def parse_sold_details(soup):
    last_sold = soup.find('strong', {'testid': 'lastSoldAttribute'})
    if last_sold:
        last_sold_text = last_sold.get_text(strip=True)

        # 提取日期并去掉 'Last Sold on'
        if 'Last Sold on' in last_sold_text and 'for' in last_sold_text:
            last_sold_date = last_sold_text.replace('Last Sold on', '').split('for')[0].strip()

            # 提取价格
            last_sold_price = last_sold_text.split('for')[-1].strip()

            return last_sold_price, last_sold_date

    # 如果没有找到相关信息，返回 None
    return None, None


# Helper to extract values like Capital Value, Land Value, Improvement Value
def extract_value(soup, value_type):
    try:
        value = soup.find('div', string=value_type).find_next_sibling('div').get_text(strip=True)
        return value
    except AttributeError:
        return 'N/A'

# Fetch rental history function
def fetch_rental_history(soup):
    rental_history = []
    has_rental_history = False
    is_currently_rented = False

    # 解析租赁历史记录
    events = soup.find_all('div', class_='d-flex flex-row w-100 align-items-center pr-3 mb-2')
    for event in events:
        description = event.find('strong', {'testid': lambda x: x and x.startswith('pt-description')})
        if description:
            description_text = description.get_text(strip=True)
            
            # 检查是否有 "Listed for Rent at" 的字样
            if "Listed for Rent at" in description_text:
                has_rental_history = True
                is_currently_rented = True  # 根据逻辑判断当前是否出租
                rental_price = description_text.split('Listed for Rent at')[-1].strip()  # 提取租金
                rental_history.append(f"Rented for {rental_price}")

    return {
        'history': rental_history if rental_history else "No rental history available",
        'has_rental_history': has_rental_history,
        'is_currently_rented': is_currently_rented
    }
