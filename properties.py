import requests
from bs4 import BeautifulSoup
from property_history import fetch_property_history
from config.supabase_config import insert_property_and_history  # Assuming this function exists

# 固定的城市和 suburb
# CITY = "Porirua City"
# SUBURB = "Aotea"

# Fetch property details
def fetch_property_details(property_url, title, city, suburb):
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

        # suburb = SUBURB
        # city = CITY

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

        # Fetch rental history from property_history.py
        rental_history = fetch_property_history(soup)

        # Prepare property data for insertion into Supabase
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
            'has_rental_history': rental_history['has_rental_history'],
            'is_currently_rented': rental_history['is_currently_rented']
        }

        # Prepare history data for insertion into Supabase
        history_data = rental_history['history']

        return property_data, history_data  # Return the data for insertion

    else:
        print(f"❌ Failed to fetch details for: {property_url}")
        return None, None

# Step 3: Parse sold details
def parse_sold_details(soup):
    last_sold = soup.find('strong', {'testid': 'lastSoldAttribute'})
    if last_sold:
        last_sold_text = last_sold.get_text(strip=True)

        if 'for' in last_sold_text and 'on' in last_sold_text:
            last_sold_price = last_sold_text.split('for')[-1].strip()

            # Clean up the last_sold_price: remove '$' and ',' and convert to a float
            last_sold_price = last_sold_price.replace('$', '').replace(',', '')
            try:
                last_sold_price = float(last_sold_price)
            except ValueError:
                last_sold_price = None  # If price cannot be parsed, set it to None

            if 'Last Sold on' in last_sold_text:
                last_sold_date = last_sold_text.replace('Last Sold on', '').split('for')[0].strip()
            else:
                last_sold_date = last_sold_text.split('on')[-1].strip()

            return last_sold_price, last_sold_date

    return None, None


# Helper to extract values like Capital Value, Land Value, Improvement Value
def extract_value(soup, value_type):
    try:
        value = soup.find('div', string=value_type).find_next_sibling('div').get_text(strip=True)
        return value
    except AttributeError:
        return 'N/A'
