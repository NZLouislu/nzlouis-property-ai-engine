#!/usr/bin/env python3
"""
检查运行状态
每隔30分钟检查一次，如果发现running状态就退出，只有idle状态才允许执行
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_config import create_supabase_client

# 加载环境变量
load_dotenv()

def check_and_allow_execution():
    """
    检查所有任务状态，如果有running状态就退出，只有idle状态才允许执行
    """
    supabase = create_supabase_client()
    
    try:
        # 获取所有记录
        response = supabase.table('scraping_progress').select('*').execute()
        
        if not response.data:
            print("✓ 没有找到任何记录，允许执行")
            return
        
        running_tasks = []
        
        for record in response.data:
            script_id = record['id']
            status = record['status']
            
            if status == 'running':
                running_tasks.append(script_id)
        
        if running_tasks:
            print(f"✗ 发现正在运行的任务: {running_tasks}")
            print("✗ 有其他任务正在执行，当前任务退出")
            sys.exit(1)
        else:
            print("✓ 所有任务状态为idle或complete，允许执行")
            
    except Exception as e:
        print(f"✗ 状态检查失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_and_allow_execution()