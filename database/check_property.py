def check_property_exists(redis_client, address):
    # 从 Redis 中检查地址是否存在
    if redis_client.exists(address):
        print(f"Property at {address} exists in Redis cache.")
        return True
    return False

def cache_property_address(redis_client, address):
    # 将地址缓存到 Redis 中，过期时间为 24 小时
    redis_client.set(address, "exists", ex=86400)  # 设置缓存 24 小时
    print(f"Cached property address {address} in Redis.")
