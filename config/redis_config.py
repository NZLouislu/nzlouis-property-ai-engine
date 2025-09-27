# redis_config.py
import os

# Try to import Upstash Redis, fallback if not available
try:
    from upstash_redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    Redis = None
    REDIS_AVAILABLE = False

# Redis connection configuration
def create_redis_client():
    if REDIS_AVAILABLE:
        # Use Redis.from_env() which is the correct way to initialize from environment variables
        return Redis.from_env()
    else:
        # Return a mock client when Redis is not available
        return MockRedisClient()

# Mock Redis client for when the real client is not available
class MockRedisClient:
    def get(self, key):
        return None
    
    def set(self, key, value):
        pass

# Check if a property address exists in Redis
def check_property_in_redis(redis_client, address):
    # Upstash Redis client uses get() and checks for None
    return redis_client.get(address) is not None

# Add a property address to Redis after insertion
def add_property_to_redis(redis_client, address):
    redis_client.set(address, 1)

# Check if a real estate property address exists in Redis
def check_real_estate_in_redis(redis_client, address):
    return redis_client.get("real" + address) is not None

# Add a real estate property address to Redis after insertion
def add_real_estate_to_redis(redis_client, address):
    redis_client.set("real"+address, 1)

# Check if a real estate rent property address exists in Redis
def check_real_estate_rent_in_redis(redis_client, address):
    return redis_client.get("real" + address) is not None

# Add a real estate property rent address to Redis after insertion
def add_real_estate_rent_to_redis(redis_client, address):
    redis_client.set("real"+address, 1)