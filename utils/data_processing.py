import datetime

def parse_event_date(date_str):
    # 将字符串日期解析为标准日期格式
    try:
        return datetime.datetime.strptime(date_str, "%d %B %Y")
    except ValueError:
        return None  # 返回 None 表示日期格式不正确
