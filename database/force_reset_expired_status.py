#!/usr/bin/env python3
"""
强制重置过期的运行状态
在每个 GitHub Action 开始时运行，确保没有过期的 running 状态
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

def force_reset_expired_status():
    """
    强制重置所有过期的 running 状态（超过30分钟）
    """
    supabase = create_supabase_client()
    
    try:
        # 获取所有记录
        response = supabase.table('scraping_progress').select('*').execute()
        
        if not response.data:
            print("没有找到任何记录")
            return
        
        current_time = datetime.now(timezone.utc)
        reset_count = 0
        
        for record in response.data:
            script_id = record['id']
            status = record['status']
            updated_at_str = record.get('updated_at')
            
            if status == 'running' and updated_at_str:
                try:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                    time_diff = current_time - updated_at
                    
                    # 如果超过30分钟，强制重置为 idle
                    if time_diff.total_seconds() > 1800:  # 30 minutes
                        supabase.table('scraping_progress').update({
                            'status': 'idle',
                            'updated_at': 'now()'
                        }).eq('id', script_id).execute()
                        
                        print(f"✓ 重置过期状态: ID {script_id} (过期 {time_diff.total_seconds():.0f} 秒)")
                        reset_count += 1
                        
                except Exception as e:
                    print(f"✗ 处理记录 ID {script_id} 时出错: {e}")
        
        if reset_count == 0:
            print("✓ 没有发现过期的运行状态")
        else:
            print(f"✓ 总共重置了 {reset_count} 个过期状态")
            
    except Exception as e:
        print(f"✗ 强制重置失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    force_reset_expired_status()