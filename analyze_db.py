import os
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)

print('=== 数据库连接测试 ===')
print('连接成功!')

print('\n=== Properties表分析 ===')
try:
    props = supabase.table('properties').select('*').limit(5).execute()
    if props.data:
        df = pd.DataFrame(props.data)
        print(f'列名: {list(df.columns)}')
        print(f'数据量: {len(props.data)}')
    else:
        print('无数据')
except Exception as e:
    print(f'错误: {e}')

print('\n=== 训练数据视图 ===')
try:
    train = supabase.table('properties_with_is_listed').select('*').limit(3).execute()
    if train.data:
        df_train = pd.DataFrame(train.data)
        print(f'列名: {list(df_train.columns)}')
        print(f'数据量: {len(train.data)}')
    else:
        print('无数据')
except Exception as e:
    print(f'错误: {e}')

print('\n=== 预测数据视图 ===')
try:
    pred = supabase.table('properties_to_predict').select('*').limit(3).execute()
    if pred.data:
        df_pred = pd.DataFrame(pred.data)
        print(f'列名: {list(df_pred.columns)}')
        print(f'数据量: {len(pred.data)}')
    else:
        print('无数据')
except Exception as e:
    print(f'错误: {e}')