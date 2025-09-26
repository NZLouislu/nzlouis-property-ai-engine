#!/usr/bin/env python3
"""
测试所有脚本的状态检查功能
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_config import create_supabase_client

# 加载环境变量
load_dotenv()

def test_status_check():
    """
    测试所有脚本的状态检查
    """
    supabase = create_supabase_client()
    
    # 脚本和对应的 ID 映射
    scripts = {
        1: "property_image.py",
        2: "real_estate_auckland.py", 
        3: "real_estate_wellington.py",
        4: "real_estate_rent.py",
        5: "property_wellington.py",
        6: "migrate_property_history.py"
    }
    
    print("检查所有脚本的状态...")
    print("=" * 50)
    
    try:
        response = supabase.table('scraping_progress').select('*').execute()
        
        if response.data:
            for record in response.data:
                script_id = record['id']
                status = record['status']
                updated_at = record.get('updated_at', 'N/A')
                script_name = scripts.get(script_id, f"Unknown script (ID: {script_id})")
                
                print(f"ID {script_id}: {script_name}")
                print(f"  状态: {status}")
                print(f"  更新时间: {updated_at}")
                print()
        else:
            print("没有找到任何记录")
            
    except Exception as e:
        print(f"错误: {e}")

def reset_all_status():
    """
    重置所有状态为 idle
    """
    supabase = create_supabase_client()
    
    try:
        # 重置所有状态为 idle
        for script_id in range(1, 7):
            response = supabase.table('scraping_progress').update({
                'status': 'idle',
                'updated_at': 'now()'
            }).eq('id', script_id).execute()
            
        print("✓ 所有状态已重置为 idle")
        
    except Exception as e:
        print(f"✗ 重置状态失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_all_status()
    else:
        test_status_check()