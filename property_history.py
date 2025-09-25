from bs4 import BeautifulSoup

# Fetch property history (like sale history, rental history)
def fetch_property_history(soup):
    property_history = []
    has_rental_history = False
    is_currently_rented = False

    # 解析历史记录
    events = soup.find_all('div', class_='d-flex flex-row w-100 align-items-center pr-3 mb-2')
    for event in events:
        # 提取日期部分
        date_day = event.find('div', {'testid': lambda x: x and x.startswith('pt-monthDay')})
        date_year = event.find('div', {'testid': lambda x: x and x.startswith('pt-year')})

        if date_day and date_year:
            event_date = f"{date_day.get_text(strip=True)} {date_year.get_text(strip=True)}"
        elif date_year:
            event_date = date_year.get_text(strip=True)
        else:
            event_date = "Unknown date"

        # 提取描述
        description = event.find('strong', {'testid': lambda x: x and x.startswith('pt-description')})
        event_description = description.get_text(strip=True) if description else "No description"

        # 提取间隔信息
        interval = event.find('div', {'testid': lambda x: x and x.startswith('pt-interval')})
        event_interval = interval.get_text(strip=True) if interval else "No interval info"

        # 创建事件字典
        event_dict = {
            'event_date': event_date,
            'event_description': event_description,
            'event_interval': event_interval
        }

        property_history.append(event_dict)

        # 检查是否有租赁历史
        if "Listed for Rent at" in event_description or "Rented for" in event_description:
            has_rental_history = True
            is_currently_rented = True  # 根据逻辑判断当前是否出租

    # 打印 property_history 详情
    print("----------------")
    print("Property History")
    print("----------------")
    
    for event in property_history:
        print(f"Date: {event['event_date']}")
        print(f"Description: {event['event_description']}")
        print(f"Interval: {event['event_interval']}")
        print("----------------")

    return {
        'history': property_history if property_history else "No rental history available",
        'has_rental_history': has_rental_history,
        'is_currently_rented': is_currently_rented
    }

# Example usage (for testing purposes):
# if __name__ == "__main__":
#     sample_html = """<your HTML content here>"""
#     soup = BeautifulSoup(sample_html, 'html.parser')
#     fetch_property_history(soup)
