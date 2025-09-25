import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from supabase import create_client, Client
from dotenv import load_dotenv
import re
from datetime import datetime
import joblib

# 加载.env文件中的环境变量
load_dotenv()

def create_supabase_client() -> Client:
    """创建Supabase客户端"""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL 和 SUPABASE_KEY 环境变量必须设置")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def to_json_serializable(records):
    """
    将 list[dict] 中的 numpy 数据类型转换为 Python 原生类型，
    以便 json.dumps() 或 Supabase 客户端序列化不报错。
    """
    new_recs = []
    for rec in records:
        new = {}
        for k, v in rec.items():
            if isinstance(v, np.integer):        # 包括 int64, int32…
                new[k] = int(v)                  # 转为 Python int
            elif isinstance(v, np.floating):     # 包括 float64, float32…
                new[k] = float(v)                # 转为 Python float
            else:
                new[k] = v
        new_recs.append(new)
    return new_recs

def fetch_training_data():
    """
    获取训练数据：从 properties_with_is_listed（status=1）和 properties_to_predict（status=0）视图中获取数据，并合并为训练集。
    """
    print("开始获取训练数据...")
    client = create_supabase_client()
    try:
        # 获取 status=1 的房产数据
        res_listed = (
            client
            .from_('properties_with_is_listed')
            .select('*')
            .execute()
        )
        df_listed = pd.DataFrame(res_listed.data) if res_listed.data else pd.DataFrame()

        # 获取 status=0 的房产数据
        res_unlisted = (
            client
            .from_('properties_to_predict')
            .select('*')
            .execute()
        )
        df_unlisted = pd.DataFrame(res_unlisted.data) if res_unlisted.data else pd.DataFrame()

        # 合并两个数据集
        df_combined = pd.concat([df_listed, df_unlisted], ignore_index=True)

        if df_combined.empty:
            print("警告: 没有获取到训练数据。")
            return None

        print(f"获取的训练数据列: {df_combined.columns.tolist()}")
        return df_combined

    except Exception as e:
        print(f"错误: 获取训练数据时发生错误: {e}")
        return None

def fetch_prediction_data(city_filter=None, limit=None):
    """
    获取预测数据：查询 properties_to_predict 视图，
    并可按 city 条件过滤。
    
    参数:
    city_filter: 城市过滤条件 (可选)
    limit: 限制返回记录数 (可选)
    """
    print("开始获取预测数据...")
    client = create_supabase_client()
    try:
        # 构建查询
        query = client.from_('properties_to_predict').select('*')
        
        # 应用过滤条件
        if city_filter:
            query = query.eq('city', city_filter)
            
        # 应用限制
        if limit:
            query = query.limit(limit)
            
        res = query.execute()
        
        print("成功从 Supabase 获取预测数据")
        if res.data:
            df = pd.DataFrame(res.data)
            print(f"获取的预测数据列: {df.columns.tolist()}")
            print(f"获取的预测数据行数: {len(df)}")
            return df
        else:
            print("警告: 没有获取到需要预测的数据。")
            return None
    except Exception as e:
        print(f"错误: 获取预测数据时发生错误: {e}")
        return None

def clear_previous_predictions():
    """
    清除 property_status 表中的旧预测数据
    """
    print("正在清空 property_status 表中的旧数据...")
    client = create_supabase_client()
    
    try:
        # 删除所有记录
        delete_result = client.table('property_status').delete().neq('id', 0).execute()
        print(f"已清空 property_status 表，共删除 {len(delete_result.data) if delete_result.data else 0} 条记录。")
        return True
    except Exception as e:
        print(f"警告: 删除旧数据时发生错误 - {e}")
        return False

def extract_property_history_features(history_str):
    """
    从 property_history 字段中提取有用的特征：
    1. 交易次数
    2. 最近一次交易距今的天数
    3. 是否包含建造记录
    """
    if pd.isnull(history_str) or not isinstance(history_str, str):
        return 0, -1, 0

    # 提取交易事件
    events = history_str.split('; ')
    transaction_count = len(events)

    # 检查是否包含"Property Built"事件
    has_built_event = int(any('Property Built' in event for event in events))

    # 提取最近一次交易距今的天数
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = [re.search(date_pattern, event) for event in events]
    dates = [pd.to_datetime(match.group(1)) for match in dates if match]

    if dates:
        most_recent_date = max(dates)
        days_since_last_transaction = (pd.Timestamp.now() - most_recent_date).days
    else:
        days_since_last_transaction = -1  # 没有找到日期时使用默认值

    return transaction_count, days_since_last_transaction, has_built_event

def preprocess_data(df, for_prediction=False):
    print("开始数据预处理...")
    feature_columns = ['year_built', 'bedrooms', 'bathrooms', 'car_spaces', 'floor_size',
                       'land_area', 'last_sold_price', 'capital_value',
                       'land_value', 'improvement_value', 'suburb',
                       'has_rental_history', 'is_currently_rented', 'property_history']

    # 确保所有需要的列都存在
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    X = df[feature_columns].copy()

    # 处理数值型特征
    X['floor_size'] = pd.to_numeric(X['floor_size'].replace({'m²': '', ',': ''}, regex=True), errors='coerce')
    X['land_area'] = pd.to_numeric(X['land_area'].replace({'m²': '', ',': ''}, regex=True), errors='coerce')
    X['has_rental_history'] = X['has_rental_history'].astype(int) if 'has_rental_history' in X.columns else 0
    X['is_currently_rented'] = X['is_currently_rented'].astype(int) if 'is_currently_rented' in X.columns else 0

    # 提取 property_history 特征
    if 'property_history' in X.columns:
        transaction_features = X['property_history'].apply(extract_property_history_features)
        X['transaction_count'] = [x[0] for x in transaction_features]
        X['days_since_last_transaction'] = [x[1] for x in transaction_features]
        X['has_built_event'] = [x[2] for x in transaction_features]
        X = X.drop(columns=['property_history'])
    else:
        X['transaction_count'] = 0
        X['days_since_last_transaction'] = -1
        X['has_built_event'] = 0

    # 处理类别型特征（如 suburb）
    if 'suburb' in X.columns:
        # 使用'Unknown'填充缺失的suburb值，然后进行编码
        X['suburb'] = X['suburb'].fillna('Unknown').astype('category').cat.codes

    # 将所有特征转为数值型，并填充缺失值
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 处理无穷大值，将其替换为数据集中的最大合理值或默认值
    X.replace([float('inf'), -float('inf')], -1, inplace=True)

    # 更好地处理缺失值：使用均值填充数值特征，使用-1填充其他特征
    numeric_columns = ['year_built', 'bedrooms', 'bathrooms', 'car_spaces', 'floor_size', 
                      'land_area', 'last_sold_price', 'capital_value', 'land_value', 
                      'improvement_value', 'transaction_count', 'days_since_last_transaction']
    
    for col in X.columns:
        if X[col].isna().any():
            if col in numeric_columns:
                # 使用均值填充数值特征，如果均值也是NaN则使用0
                mean_val = X[col].mean()
                if pd.isna(mean_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(mean_val)
            else:
                # 使用-1填充其他特征
                X[col] = X[col].fillna(-1)

    # 最后检查并确保没有NaN值
    if X.isna().sum().sum() > 0:
        print(f"警告：仍然存在 {X.isna().sum().sum()} 个NaN值，将用0填充")
        X = X.fillna(0)

    print("数据预处理完成，特征列：", X.columns.tolist())
    print("数据形状：", X.shape)
    print("缺失值统计：", X.isna().sum().sum())

    if not for_prediction:
        if 'status' in df.columns:
            y = df['status'].astype(int)  # 0/1
            # 改为警告而非抛错
            uniques = y.unique()
            print(">> 标签唯一值：", uniques)
            if len(uniques) < 2:
                print("警告：训练集只有单一标签，无法训练有区分度模型，请检查数据视图或查询。")
            return X, y, None
        else:
            print("错误：训练数据中缺少 status 字段")
            return None, None, None
    else:
        return X

def train_model(X, y):
    print("开始模型训练...")
    # 增加 stratify 以保持正负样本比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 修复语法，增加 class_weight
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    joblib.dump((model, X.columns.tolist()), 'property_status_model.joblib')
    print("模型和特征名称已保存为 'property_status_model.joblib'")

    return model, X.columns.tolist()

def predict_and_store(model, feature_names, prediction_df, clear_old_data=True):
    """
    预测并将结果存储到数据库。
    
    参数:
    model: 训练好的模型
    feature_names: 特征名称列表
    prediction_df: 需要预测的数据
    clear_old_data: 是否在预测前清除旧数据
    """
    print("开始预测并存储结果...")

    # 创建 Supabase 客户端
    supabase_client = create_supabase_client()

    # 如果需要，清空旧的预测数据
    if clear_old_data:
        clear_previous_predictions()

    # 预处理数据
    X_pred = preprocess_data(prediction_df, for_prediction=True)
    X_pred = X_pred.reindex(columns=feature_names, fill_value=0)  # 确保所有列都匹配

    # 预测结果和置信度
    predictions = model.predict(X_pred)
    confidence_scores = model.predict_proba(X_pred).max(axis=1)

    # 映射预测结果
    mapping = {1: 'for Sale', 0: 'not for Sale'}
    predicted_statuses = [mapping.get(int(p), str(p)) for p in predictions]

    # 当前时间精确到秒
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 准备要插入的数据
    records_to_insert = []
    for idx, row in prediction_df.iterrows():
        property_id = row['id']
        predicted_status = predicted_statuses[idx]
        confidence_score = float(confidence_scores[idx])

        records_to_insert.append({
            'property_id': property_id,
            'predicted_status': predicted_status,
            'confidence_score': confidence_score,
            'predicted_at': current_time
        })

    # 转换所有 numpy 类型为原生类型
    records_to_insert = to_json_serializable(records_to_insert)

    # 分批插入数据（如果数据量很大）
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(records_to_insert), batch_size):
        batch = records_to_insert[i:i+batch_size]
        
        # 批量插入数据
        try:
            result = supabase_client.table('property_status').insert(batch).execute()

            # 检查插入结果并记录日志
            if result.data:
                batch_inserted = len(result.data)
                total_inserted += batch_inserted
                print(f"成功插入批次 {i//batch_size + 1}，共 {batch_inserted} 条记录")
                
                # 显示前几条记录作为示例
                for j, inserted_record in enumerate(result.data[:3]):  # 只显示前3条
                    inserted_id = inserted_record.get('property_id')
                    predicted_at = inserted_record.get('predicted_at', current_time)
                    print(f"  成功存储预测结果 - ID: {inserted_id}, 预测状态: {inserted_record['predicted_status']}, 置信度: {inserted_record['confidence_score']:.4f}, 预测时间: {predicted_at}")
                    
                if len(result.data) > 3:
                    print(f"  ... 还有 {len(result.data) - 3} 条记录")
            else:
                print(f"警告: 批次 {i//batch_size + 1} 插入成功，但未能检索到插入的记录。")
        except Exception as insert_error:
            print(f"错误: 批量存储预测结果时发生错误 - 错误详情: {insert_error}")
    
    print(f"总共插入 {total_inserted} 条预测记录")

def main():
    print("程序开始运行...")
    try:
        # 获取训练数据
        training_df = fetch_training_data()
        if training_df is None or training_df.empty:
            print("错误: 没有获取到有效的训练数据，程序终止。")
            return

        # 数据预处理
        X, y, le = preprocess_data(training_df)
        if X is None or y is None:
            print("错误: 数据预处理失败，程序终止。")
            return

        # 训练模型
        model, feature_names = train_model(X, y)

        # 获取预测数据 - 这里我们演示获取所有需要预测的数据
        # 可以根据需要修改过滤条件，例如获取特定城市的所有数据
        print("\n=== 获取所有待预测数据 ===")
        prediction_df = fetch_prediction_data()
        
        if prediction_df is None or prediction_df.empty:
            print("警告: 没有获取到需要预测的数据，程序终止。")
            return

        # 预测并存储结果
        # 注意：这里设置 clear_old_data=True，表示在第一次运行时会删除之前的预测数据
        predict_and_store(model, feature_names, prediction_df, clear_old_data=True)
        print("预测完成，结果已存储到数据库。")

        # 额外演示：获取特定区域（如Wellington）的数据进行预测，限制数量以避免超时
        print("\n=== 获取Wellington地区待预测数据（限制100条）===")
        wellington_df = fetch_prediction_data(city_filter="Wellington", limit=100)
        
        if wellington_df is not None and not wellington_df.empty:
            print(f"获取到 {len(wellington_df)} 条 Wellington 地区的待预测数据")
            # 注意：这里设置 clear_old_data=False，表示不清除之前的数据
            # 如果需要清除，可以设置为True，或者单独调用 clear_previous_predictions()
            predict_and_store(model, feature_names, wellington_df, clear_old_data=False)
            print("Wellington地区预测完成，结果已存储到数据库。")
        else:
            print("没有获取到 Wellington 地区的待预测数据")

    except Exception as e:
        print(f"错误: 程序运行时发生错误：{e}")
        import traceback
        print("错误详情:")
        print(traceback.format_exc())

    print("程序运行结束。")

if __name__ == "__main__":
    main()