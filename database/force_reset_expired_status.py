#!/usr/bin/env python3
"""
强制重置过期状态
根据任务ID重置对应任务的running状态为idle，让系统自动重试
每隔30分钟运行一次，确保异常退出的任务能被重新启动
"""

import os
import sys
from datetime import datetime, timezone, timedelta

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_config import create_supabase_client

# 尝试加载环境变量，如果 dotenv 不可用则跳过
try:
    from dotenv import load_dotenv
    # 加载环境变量
    load_dotenv()
except ImportError:
    # 在 GitHub Actions 环境中，环境变量由工作流直接提供
    pass

# 任务ID映射
TASK_MAPPING = {
    'property_image': 1,
    'real_estate_auckland': 2,
    'real_estate_wellington': 3,
    'real_estate_rent': 4,
    'property_wellington': 5,
    'migrate_property_history': 6
}

def force_reset_expired_status(task_id=None):
    """
    重置指定任务的running状态为idle
    
    Args:
        task_id (int, optional): 要重置的任务ID，如果为None则重置所有任务
    """
    supabase = create_supabase_client()
    
    try:
        # 构建查询条件
        query = supabase.table('scraping_progress').select('*').eq('status', 'running')
        
        if task_id is not None:
            query = query.eq('id', task_id)
            print(f"🔍 检查任务 ID {task_id} 的状态...")
        else:
            print("🔍 检查所有任务的状态...")
        
        response = query.execute()
        
        if not response.data:
            if task_id is not None:
                print(f"✓ 任务 ID {task_id} 没有运行中的状态")
            else:
                print("✓ 没有发现运行中的任务")
            return
        
        reset_count = 0
        for record in response.data:
            script_id = record['id']
            updated_at = record.get('updated_at')
            
            # 检查是否超过30分钟未更新
            if updated_at:
                from dateutil import parser
                last_update = parser.parse(updated_at)
                now = datetime.now(timezone.utc)
                time_diff = now - last_update
                
                # 只有超过30分钟未更新的任务才重置
                if time_diff <= timedelta(minutes=30):
                    print(f"⚠️ 任务 ID {script_id} 正在运行中 ({time_diff} 前更新)，跳过重置")
                    continue
            
            # 重置为idle
            supabase.table('scraping_progress').update({
                'status': 'idle',
                'updated_at': 'now()'
            }).eq('id', script_id).execute()
            
            print(f"✓ 重置任务状态: ID {script_id} (running → idle)")
            reset_count += 1
        
        print(f"✓ 总共重置了 {reset_count} 个过期的运行状态")
        print("✓ 状态重置完成，任务可以开始执行")
            
    except Exception as e:
        print(f"✗ 重置失败: {e}")
        sys.exit(1)

def get_task_id():
    """
    从命令行参数或环境变量获取任务ID
    """
    # 从命令行参数获取
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
        print(f"📝 从命令行参数获取任务: {task_name}")
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name]
        else:
            try:
                return int(task_name)
            except ValueError:
                print(f"✗ 未知的任务名称: {task_name}")
                sys.exit(1)
    
    # 从GitHub工作流名称推断
    github_workflow = os.getenv('GITHUB_WORKFLOW')
    if github_workflow:
        print(f"🔍 从GitHub工作流推断: {github_workflow}")
        workflow_mapping = {
            'Update Property Images': 'property_image',
            'Scrape RealEstate Auckland': 'real_estate_auckland',
            'Scrape RealEstate Wellington': 'real_estate_wellington',
            'Scrape RealEstate Rent': 'real_estate_rent',
            'Scrape Wellington Properties stopped': 'property_wellington',
            'Migrate Property History Data': 'migrate_property_history'
        }
        if github_workflow in workflow_mapping:
            task_name = workflow_mapping[github_workflow]
            return TASK_MAPPING[task_name]
    
    print("⚠️ 无法确定任务ID，将重置所有任务")
    return None

if __name__ == "__main__":
    task_id = get_task_id()
    force_reset_expired_status(task_id)