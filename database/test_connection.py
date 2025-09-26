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
            property_id = first_property['id']
            print(f"First property ID: {property_id}")
            
            # 测试 property_history 表查询
            print(f"\nTesting property_history for property {property_id}...")
            
            # 先测试 count 查询
            try:
                count_response = supabase.table('property_history').select(
                    'id', count='exact'
                ).eq('property_id', property_id).execute()
                print(f"✓ History count query successful: {count_response.count} records")
                
                # 如果有记录，测试获取数据
                if count_response.count and count_response.count > 0:
                    print("Testing history data retrieval...")
                    data_response = supabase.table('property_history').select(
                        'event_date, event_description'
                    ).eq('property_id', property_id).limit(5).execute()
                    print(f"✓ History data query successful: {len(data_response.data)} records retrieved")
                    
                    if data_response.data:
                        print(f"Sample record: {data_response.data[0]}")
                else:
                    print("No history records found for this property")
                    
            except Exception as e:
                print(f"✗ History query failed: {e}")
        
        print("\n" + "="*50)
        print("Database connection test completed successfully!")
        
    except Exception as e:
        print(f"✗ Database connection test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_basic_queries()