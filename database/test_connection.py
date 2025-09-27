#!/usr/bin/env python3
"""
测试数据库连接和基本查询
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_config import create_supabase_client

# 加载环境变量
load_dotenv()

def test_basic_queries():
    """
    测试基本的数据库查询
    """
    print("Testing database connection...")
    
    try:
        supabase = create_supabase_client()
        print("✓ Supabase client created successfully")
        
        # 测试 properties 表查询
        print("\nTesting properties table...")
        response = supabase.table('properties').select('id, address').limit(3).execute()
        print(f"✓ Properties query successful, got {len(response.data)} records")
        
        if response.data:
            first_property = response.data[0]
            print(f"First property ID: {first_property['id']}")
        
        print("\n" + "="*50)
        print("Database connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_queries()