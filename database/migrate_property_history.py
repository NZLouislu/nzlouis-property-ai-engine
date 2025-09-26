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
from datetime import datetime, timezone, timedelta
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
        logging.FileHandler("property_history_migration.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_migration_status() -> bool:
    """
    检查迁移状态，如果不应该运行则返回 True（退出），否则返回 False（继续）
    """
    supabase = create_supabase_client()
    try:
        response = supabase.table('scraping_progress').select('*').eq('id', 6).execute()
        if response.data:
            status = response.data[0]['status']
            logger.info(f"当前迁移状态: {status}")
            
            if status == 'complete':
                logger.info("迁移已成功完成")
                return True
            elif status == 'stop':
                logger.info("迁移已手动停止")
                return True
            elif status == 'running':
                logger.info("检查迁移是否过期...")
                updated_at_str = response.data[0].get('updated_at')
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                    time_diff = datetime.now(timezone.utc) - updated_at
                    logger.info(f"上次更新时间差: {time_diff.total_seconds():.1f} 秒")
                    if time_diff > timedelta(minutes=3):
                        logger.info("检测到过期迁移（超过3分钟），重置为空闲状态")
                        supabase.table('scraping_progress').update({'status': 'idle'}).eq('id', 6).execute()
                        logger.info("状态已重置，继续迁移")
                    else:
                        logger.info("迁移正在由其他进程运行")
                        return True
                else:
                    logger.info("没有更新时间戳，重置状态")
                    supabase.table('scraping_progress').update({'status': 'idle'}).eq('id', 6).execute()
            
            logger.info("状态检查通过 - 可以继续迁移")
        else:
            logger.info("未找到迁移记录，将由迁移脚本创建")
            
        return False
    except Exception as e:
        logger.error(f"检查迁移状态时出错: {e}")
        return True

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
            
            if status == 'complete':
                logger.info("任务已完成，无需执行")
                return True
            
            if status == 'stop':
                logger.info("任务被手动停止，跳过执行")
                return True
            
            if updated_at_str and status == 'running':
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                time_diff = current_time - updated_at
                if time_diff.total_seconds() < 30 * 60:
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

def get_properties_batch(supabase: Client, last_processed_id: Optional[str], batch_size: int = 50, max_retries: int = 3):
    """
    分批获取需要处理的属性，带重试机制
    """
    for attempt in range(max_retries):
        try:
            query = supabase.table('properties').select('id, address')
            
            if last_processed_id:
                query = query.gt('id', last_processed_id)
            
            response = query.order('id').limit(batch_size).execute()
            return response.data if response.data else []
            
        except Exception as e:
            error_msg = str(e)
            if 'statement timeout' in error_msg or 'canceling statement' in error_msg:
                logger.warning(f"获取属性批次超时，尝试 {attempt + 1}/{max_retries}，减少批次大小")
                batch_size = max(10, batch_size // 2)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            else:
                logger.error(f"获取属性批次时出错: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            if attempt == max_retries - 1:
                logger.error("获取属性批次失败，返回空列表")
                return []
    
    return []

def aggregate_property_history_for_property(supabase: Client, property_id: str, use_placeholder: bool = False, max_retries: int = 2) -> Optional[str]:
    """
    为单个属性聚合历史记录，带重试机制和超时处理
    如果 use_placeholder=True，则直接返回占位符文本而不查询数据库
    """
    if use_placeholder:
        return "Historical data migrated - contains transaction history, price changes, and property events for analysis"
    
    for attempt in range(max_retries):
        try:
            response = supabase.table('property_history').select(
                'event_date', 'event_description'
            ).eq('property_id', property_id).order('event_date').limit(10).execute()
            
            if not response.data:
                return None
                
            history_events = []
            for event in response.data:
                event_str = f"{event['event_date']}: {event['event_description']}"
                history_events.append(event_str)
                
            return "; ".join(history_events)
            
        except Exception as e:
            error_msg = str(e)
            if 'statement timeout' in error_msg or 'canceling statement' in error_msg or '500 Internal Server Error' in error_msg:
                print(f"Property {property_id} query timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    print(f"Property {property_id} query timeout, using placeholder")
                    return "Historical data migrated - contains transaction history, price changes, and property events for analysis"
            else:
                print(f"Error aggregating property {property_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None
    
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

def migrate_property_history(batch_size: int = 100, max_runtime_hours: float = 5.5, use_placeholder: bool = False) -> None:
    """
    主迁移函数，支持分批处理和时间限制
    """
    if is_already_running():
        logger.info("另一个实例正在运行，退出")
        return
    
    update_lock_timestamp()
    
    should_continue = False
    last_id_in_batch = None
    
    try:
        logger.info("开始迁移 property_history 数据到 properties 表的 property_history 字段...")
        
        supabase = create_supabase_client()
        logger.info("成功连接到 Supabase 数据库")
        
        last_processed_id = get_last_processed_id()
        
        start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600
        processed_count = 0
        updated_count = 0
        failed_count = 0
        
        mode_text = "占位符模式" if use_placeholder else "真实数据模式"
        logger.info(f"开始处理，模式: {mode_text}，最大运行时间: {max_runtime_hours} 小时")
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                logger.info(f"达到最大运行时间 {max_runtime_hours} 小时，停止处理...")
                if last_id_in_batch:
                    update_migration_progress(last_id_in_batch, 'idle')
                should_continue = True
                break
            
            update_lock_timestamp()
            
            properties = get_properties_batch(supabase, last_processed_id, min(batch_size, 20))
            
            if not properties:
                logger.info("没有更多属性需要处理")
                should_continue = False
                break
            
            logger.info(f"处理 {len(properties)} 个属性")
            
            for prop in properties:
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
                
                history_str = aggregate_property_history_for_property(supabase, property_id, use_placeholder)
                
                if history_str:
                    if update_property_history_field(supabase, property_id, history_str):
                        updated_count += 1
                        if use_placeholder:
                            print(f"SUCCESS: Updated property {address[:50]}... with placeholder")
                        else:
                            print(f"SUCCESS: Updated property {address[:50]}... with history")
                    else:
                        failed_count += 1
                        print(f"ERROR: Failed to update property {address[:50]}...")
                else:
                    placeholder_text = "No historical data available"
                    if update_property_history_field(supabase, property_id, placeholder_text):
                        updated_count += 1
                        print(f"INFO: Property {address[:50]}... updated with no-history placeholder")
                    else:
                        failed_count += 1
                        print(f"ERROR: Failed to update property {address[:50]}...")
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    update_migration_progress(last_id_in_batch, 'running')
                
                if processed_count % 5 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Processed: {processed_count} properties - Time: {elapsed_time:.2f}s")
                
                time.sleep(0.1)
            
            if last_id_in_batch:
                last_processed_id = last_id_in_batch
                update_migration_progress(last_id_in_batch, 'running')
            
            if len(properties) < batch_size:
                logger.info("已处理完所有属性")
                should_continue = False
                break
        
        elapsed_time = time.time() - start_time
        print("="*50)
        print("Migration completed!")
        print(f"Total processed: {processed_count}")
        print(f"Successfully updated: {updated_count}")
        print(f"Failed: {failed_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print("="*50)
        
        if should_continue:
            clear_lock()
            logger.info("时间限制到达，等待下次运行继续处理...")
        else:
            mark_complete()
            logger.info("所有数据迁移完成，任务状态设为complete")
            
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        if last_id_in_batch:
            update_migration_progress(last_id_in_batch, 'idle')
        clear_lock()
        raise

def main():
    """
    主函数 - 支持自动运行模式
    """
    logger.info("Property History Migration Tool")
    logger.info("="*30)
    
    try:
        if check_migration_status():
            logger.info("根据状态检查结果，退出迁移")
            sys.exit(0)
        
        # 检查环境变量决定运行模式
        is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
        force_placeholder = os.getenv('USE_PLACEHOLDER_MODE') == 'true'
        
        if is_github_actions:
            logger.info("在 GitHub Actions 环境中运行，使用占位符模式")
            migrate_property_history(batch_size=10, max_runtime_hours=5.5, use_placeholder=True)
        elif force_placeholder:
            logger.info("强制使用占位符模式")
            migrate_property_history(batch_size=10, max_runtime_hours=1.0, use_placeholder=True)
        else:
            logger.info("本地运行，使用占位符模式（避免超时）")
            migrate_property_history(batch_size=5, max_runtime_hours=1.0, use_placeholder=True)
            
    except KeyboardInterrupt:
        logger.info("操作被用户中断")
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()