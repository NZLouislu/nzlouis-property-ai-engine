#!/usr/bin/env python3
"""
自动化脚本，用于将 property_history 表的记录合并到 properties 表的 property_history 字段中
参考 property_image.py 的成功模式，支持自动运行、防止并发执行、进度跟踪等功能
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Optional

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_config import create_supabase_client

# 加载环境变量
load_dotenv()

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("property_history_migration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_property_history_column_exists(supabase: Client) -> bool:
    """
    检查 properties 表中是否已存在 property_history 字段
    """
    try:
        response = supabase.table('properties').select('property_history').limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"检查 property_history 字段时出错: {e}")
        return False

def ensure_property_history_column(supabase: Client) -> bool:
    """
    确保 properties 表中存在 property_history 字段
    如果不存在，尝试创建（需要数据库权限）
    """
    if check_property_history_column_exists(supabase):
        logger.info("✓ properties 表中已存在 property_history 字段")
        return True
    
    logger.warning("properties 表中不存在 property_history 字段")
    logger.info("请确保在 Supabase SQL 编辑器中执行以下 SQL 语句:")
    logger.info("ALTER TABLE properties ADD COLUMN IF NOT EXISTS property_history TEXT;")
    
    # 在 GitHub Actions 环境中，我们假设字段已经存在或会被手动添加
    # 这里不阻塞执行，而是继续尝试
    return False

def get_last_processed_id() -> Optional[str]:
    """
    获取上次处理的最后一个 property ID
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').select('last_processed_id').eq('id', 6).execute()
        if response.data and len(response.data) > 0:
            last_processed_id = response.data[0].get('last_processed_id')
            if last_processed_id:
                logger.info(f"从 ID {last_processed_id} 继续处理")
                return last_processed_id
        
        logger.info("从头开始处理")
        return None
    except Exception as e:
        logger.error(f"获取上次处理 ID 时出错: {e}")
        return None

def update_migration_progress(processed_count: int, status: str = 'running') -> bool:
    """
    更新迁移进度到 scraping_progress 表
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').upsert({
            'id': 6,
            'last_processed_id': str(processed_count),
            'batch_size': 1000,
            'status': status,
            'updated_at': 'now()'
        }).execute()
        
        return True
    except Exception as e:
        logger.error(f"更新迁移进度时出错: {e}")
        return False

def is_already_running() -> bool:
    """
    检查是否已有实例在运行或任务状态
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').select('updated_at, status').eq('id', 6).execute()
        if response.data and len(response.data) > 0:
            updated_at_str = response.data[0].get('updated_at')
            status = response.data[0].get('status', 'idle')
            
            # 检查任务是否已完成
            if status == 'complete':
                logger.info("任务已完成，无需执行")
                return True
            
            # 检查任务是否被手动停止
            if status == 'stop':
                logger.info("任务被手动停止，跳过执行")
                return True
            
            # 检查是否有实例正在运行
            if updated_at_str and status == 'running':
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                time_diff = current_time - updated_at
                if time_diff.total_seconds() < 30 * 60:  # 30分钟内
                    logger.info("另一个实例正在运行，跳过执行")
                    return True
                else:
                    logger.info("发现过期锁，清除并继续")
                    clear_lock()
        return False
    except Exception as e:
        logger.error(f"检查运行状态时出错: {e}")
        return False

def update_lock_timestamp():
    """
    更新锁时间戳，表示进程正在运行
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').upsert({
            'id': 6,
            'status': 'running',
            'updated_at': 'now()'
        }).execute()
        
        logger.debug("锁时间戳更新成功")
    except Exception as e:
        logger.error(f"更新锁时间戳时出错: {e}")

def clear_lock():
    """
    清除锁，表示进程已暂停
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'idle',
            'updated_at': 'now()'
        }).eq('id', 6).execute()
        
        logger.info("锁已清除，状态设为idle")
    except Exception as e:
        logger.error(f"清除锁时出错: {e}")

def mark_complete():
    """
    标记任务为完成状态
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').update({
            'status': 'complete',
            'updated_at': 'now()'
        }).eq('id', 6).execute()
        
        logger.info("任务标记为完成")
    except Exception as e:
        logger.error(f"标记完成状态时出错: {e}")

def trigger_next_workflow():
    """
    触发下一次工作流运行
    """
    github_token = os.getenv('GITHUB_TOKEN')
    github_repo = os.getenv('GITHUB_REPOSITORY')
    
    if not github_token or not github_repo:
        logger.warning("GitHub token 或 repository 不可用，无法触发下一次工作流")
        return
    
    try:
        import requests
        
        url = f"https://api.github.com/repos/{github_repo}/actions/workflows/migrate_property_history.yml/dispatches"
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        data = {
            'ref': 'main'
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 204:
            logger.info("成功触发下一次工作流运行")
        else:
            logger.warning(f"触发下一次工作流失败，状态码: {response.status_code}")
            
    except Exception as e:
        logger.error(f"触发下一次工作流时出错: {e}")

def get_properties_batch(supabase: Client, last_processed_id: Optional[str], batch_size: int = 100):
    """
    分批获取需要处理的属性
    """
    try:
        query = supabase.table('properties').select('id, address')
        
        if last_processed_id:
            query = query.gt('id', last_processed_id)
        
        response = query.order('id').limit(batch_size).execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"获取属性批次时出错: {e}")
        return []

def aggregate_property_history_for_property(supabase: Client, property_id: str) -> Optional[str]:
    """
    为单个属性聚合历史记录
    """
    try:
        response = supabase.table('property_history').select(
            'event_date', 'event_description', 'interval_since_last_event'
        ).eq('property_id', property_id).order('event_date').execute()
        
        if not response.data:
            return None
            
        history_events = []
        for event in response.data:
            event_str = f"{event['event_date']}: {event['event_description']} ({event['interval_since_last_event']})"
            history_events.append(event_str)
            
        return "; ".join(history_events)
    except Exception as e:
        logger.error(f"聚合属性 {property_id} 的历史记录时出错: {e}")
        return None

def update_property_history_field(supabase: Client, property_id: str, history_str: str) -> bool:
    """
    更新单个属性的 property_history 字段
    """
    try:
        response = supabase.table('properties').update({
            'property_history': history_str
        }).eq('id', property_id).execute()
        
        return bool(response.data)
    except Exception as e:
        logger.error(f"更新属性 {property_id} 的 property_history 字段时出错: {e}")
        return False

def migrate_property_history(batch_size: int = 100, max_runtime_hours: float = 5.5) -> None:
    """
    主迁移函数，支持分批处理和时间限制
    """
    # 检查是否已有实例在运行
    if is_already_running():
        logger.info("另一个实例正在运行，退出")
        return
    
    # 更新锁时间戳
    update_lock_timestamp()
    
    should_continue = False
    
    try:
        logger.info("开始迁移 property_history 数据到 properties 表的 property_history 字段...")
        
        # 创建 Supabase 客户端
        supabase = create_supabase_client()
        logger.info("✓ 成功连接到 Supabase 数据库")
        
        # 检查 property_history 字段是否存在
        if not ensure_property_history_column(supabase):
            logger.warning("property_history 字段可能不存在，但继续尝试处理")
        
        # 获取上次处理的位置
        last_processed_id = get_last_processed_id()
        
        start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600
        processed_count = 0
        updated_count = 0
        failed_count = 0
        last_id_in_batch = None
        
        logger.info(f"开始处理，最大运行时间: {max_runtime_hours} 小时")
        
        while True:
            # 检查运行时间
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                logger.info(f"达到最大运行时间 {max_runtime_hours} 小时，停止处理...")
                if last_id_in_batch:
                    update_migration_progress(last_id_in_batch, 'idle')
                should_continue = True
                break
            
            # 更新锁时间戳
            update_lock_timestamp()
            
            # 获取一批属性
            properties = get_properties_batch(supabase, last_processed_id, batch_size)
            
            if not properties:
                logger.info("没有更多属性需要处理")
                should_continue = False
                break
            
            logger.info(f"处理 {len(properties)} 个属性")
            
            # 处理每个属性
            for prop in properties:
                # 检查运行时间
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime_seconds:
                    logger.info(f"达到最大运行时间，停止处理...")
                    if last_id_in_batch:
                        update_migration_progress(last_id_in_batch, 'idle')
                    should_continue = True
                    break
                
                property_id = prop['id']
                last_id_in_batch = property_id
                address = prop.get('address', 'Unknown')
                
                # 获取并聚合该属性的历史记录
                history_str = aggregate_property_history_for_property(supabase, property_id)
                
                # 如果有历史记录，则更新字段
                if history_str:
                    if update_property_history_field(supabase, property_id, history_str):
                        updated_count += 1
                        logger.info(f"✓ 已更新属性 {address[:50]}... 的历史记录")
                    else:
                        failed_count += 1
                        logger.warning(f"✗ 更新属性 {address[:50]}... 的历史记录失败")
                else:
                    logger.debug(f"- 属性 {address[:50]}... 没有历史记录")
                
                processed_count += 1
                
                # 每处理10个属性更新一次进度
                if processed_count % 10 == 0:
                    update_migration_progress(last_id_in_batch, 'running')
                
                # 显示进度
                if processed_count % 20 == 0:
                    elapsed_time = time.time() - start_time
                    logger.info(f"已处理: {processed_count} 个属性 - 耗时: {elapsed_time:.2f}秒")
                
                # 添加小延迟
                time.sleep(0.1)
            
            # 更新最后处理的 ID
            if last_id_in_batch:
                last_processed_id = last_id_in_batch
                update_migration_progress(last_id_in_batch, 'running')
            
            # 如果处理的属性少于批次大小，说明已到末尾
            if len(properties) < batch_size:
                logger.info("已处理完所有属性")
                should_continue = False
                break
        
        # 显示最终统计信息
        elapsed_time = time.time() - start_time
        logger.info("="*50)
        logger.info("迁移完成!")
        logger.info(f"总处理记录数: {processed_count}")
        logger.info(f"成功更新记录数: {updated_count}")
        logger.info(f"失败记录数: {failed_count}")
        logger.info(f"耗时: {elapsed_time:.2f} 秒")
        logger.info("="*50)
        
        # 根据完成情况设置状态
        if should_continue:
            # 由于时间限制需要继续，设置为idle等待下次运行
            clear_lock()
            logger.info("触发下一次工作流运行以继续处理...")
            trigger_next_workflow()
        else:
            # 任务完全完成，标记为complete
            mark_complete()
            logger.info("所有数据迁移完成，任务状态设为complete")
            
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 保存进度
        if last_id_in_batch:
            update_migration_progress(last_id_in_batch, 'idle')
        # 清除锁
        clear_lock()
        # 重新抛出异常以便在 GitHub Actions 中可见
        raise

def main():
    """
    主函数 - 支持自动运行模式
    """
    logger.info("Property History Migration Tool")
    logger.info("="*30)
    
    try:
        # 检查是否在 GitHub Actions 环境中运行
        if os.getenv('GITHUB_ACTIONS'):
            logger.info("在 GitHub Actions 环境中运行，自动开始迁移")
            migrate_property_history(batch_size=50, max_runtime_hours=5.5)
        else:
            # 本地运行时询问确认
            try:
                confirmation = input("此操作将把 property_history 表中的数据迁移到 properties 表中，是否继续? (y/N): ")
                if confirmation.lower() != 'y':
                    logger.info("操作已取消")
                    return
            except KeyboardInterrupt:
                logger.info("\n操作已取消")
                return
            
            migrate_property_history(batch_size=50, max_runtime_hours=1.0)
            
    except KeyboardInterrupt:
        logger.info("\n操作被用户中断")
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()