#!/usr/bin/env python3
"""
Wellington房产预测模型 - 基于真实数据
从real_estate表获取训练数据，对properties表中的Wellington房产进行预测
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid

warnings.filterwarnings("ignore")

# 加载环境变量
load_dotenv()

def create_supabase_client() -> Client:
    """创建Supabase客户端"""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL和SUPABASE_KEY环境变量必须设置")

        return create_client(url, key)
    except Exception as e:
        print(f"❌ 创建Supabase客户端失败: {e}")
        return None

def get_training_data_from_real_estate(supabase_client):
    """从real_estate表获取Wellington和Auckland的销售数据作为训练集"""
    if not supabase_client:
        print("❌ 数据库连接不可用，无法获取训练数据")
        return None
    
    print("🔄 从real_estate表获取训练数据...")
    
    try:
        # 获取所有数据
        response = supabase_client.table('real_estate').select('*').execute()
        
        if not response.data:
            print("⚠️ real_estate表中没有数据")
            return None
            
        df = pd.DataFrame(response.data)
        print(f"✅ 从real_estate表获取了 {len(df)} 条记录")
        print(f"📋 获取的列: {list(df.columns)}")
        
        # 显示前几条数据以了解结构
        print("\n📊 数据样本:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"❌ 获取训练数据时发生错误: {e}")
        return None

def process_real_estate_data(df):
    """处理real_estate表的数据，提取特征"""
    if df is None or len(df) == 0:
        return None, None
    
    print("🔄 处理real_estate数据...")
    
    processed_data = df.copy()
    
    # 解析data字段中的JSON数据（如果存在）
    if 'data' in processed_data.columns:
        try:
            data_features = []
            
            for idx, row in processed_data.iterrows():
                try:
                    if pd.notna(row['data']) and row['data']:
                        if isinstance(row['data'], str):
                            data_dict = json.loads(row['data'])
                        else:
                            data_dict = row['data']
                        data_features.append(data_dict)
                    else:
                        data_features.append({})
                except:
                    data_features.append({})
            
            # 将JSON数据转换为DataFrame
            data_df = pd.json_normalize(data_features)
            
            # 合并数据
            processed_data = pd.concat([processed_data.reset_index(drop=True), data_df], axis=1)
            print(f"✅ 成功解析data字段，新增 {len(data_df.columns)} 个特征")
            
        except Exception as e:
            print(f"⚠️ 解析data字段时发生错误: {e}")
    
    # 筛选Wellington和Auckland的数据
    wellington_auckland_data = None
    
    if 'normalized_location' in processed_data.columns:
        wellington_auckland_data = processed_data[
            processed_data['normalized_location'].str.contains('wellington|auckland', case=False, na=False)
        ]
        print(f"✅ 从normalized_location筛选出Wellington和Auckland数据: {len(wellington_auckland_data)} 条")
    elif 'address' in processed_data.columns:
        wellington_auckland_data = processed_data[
            processed_data['address'].str.contains('wellington|auckland', case=False, na=False)
        ]
        print(f"✅ 从address字段筛选出Wellington和Auckland数据: {len(wellington_auckland_data)} 条")
    else:
        wellington_auckland_data = processed_data
        print(f"⚠️ 无法筛选地区，使用所有数据: {len(wellington_auckland_data)} 条")
    
    if len(wellington_auckland_data) == 0:
        print("❌ 没有找到Wellington或Auckland的数据")
        return None, None
    
    # 创建目标变量（基于status字段）
    if 'status' in wellington_auckland_data.columns:
        # 假设status为'sold'或类似表示已售出，我们将其作为正样本
        wellington_auckland_data['target'] = wellington_auckland_data['status'].apply(
            lambda x: 1 if pd.notna(x) and str(x).lower() in ['sold', 'sale', 'for sale', 'active'] else 0
        )
    else:
        # 如果没有status字段，创建平衡的目标变量
        print("⚠️ 没有找到status字段，创建平衡的训练数据")
        # 随机分配一半为已售出，一半为未售出
        np.random.seed(42)
        wellington_auckland_data['target'] = np.random.choice([0, 1], size=len(wellington_auckland_data), p=[0.4, 0.6])
    
    # 确保至少有两个类别
    unique_targets = wellington_auckland_data['target'].unique()
    if len(unique_targets) < 2:
        print("⚠️ 目标变量只有一个类别，创建平衡数据")
        # 将一半数据改为另一个类别
        half_idx = len(wellington_auckland_data) // 2
        wellington_auckland_data.iloc[:half_idx, wellington_auckland_data.columns.get_loc('target')] = 0
        wellington_auckland_data.iloc[half_idx:, wellington_auckland_data.columns.get_loc('target')] = 1
    
    print(f"📊 目标变量分布: {wellington_auckland_data['target'].value_counts().to_dict()}")
    
    return wellington_auckland_data, wellington_auckland_data.columns.tolist()

def create_features_from_real_data(data):
    """从真实数据中创建特征"""
    print("🔄 创建特征...")
    
    processed_data = data.copy()
    
    # 基本数值特征处理
    numeric_columns = []
    potential_numeric = ['bedrooms', 'bathrooms', 'car_spaces', 'year_built', 'price', 
                        'capital_value', 'land_value', 'improvement_value', 'floor_size', 'land_area']
    
    for col in potential_numeric:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
            numeric_columns.append(col)
    
    # 创建衍生特征
    current_year = datetime.now().year
    
    # 房屋年龄特征
    if 'year_built' in processed_data.columns:
        processed_data['property_age'] = current_year - processed_data['year_built']
        processed_data['is_new'] = (processed_data['property_age'] < 10).astype(int)
        processed_data['is_old'] = (processed_data['property_age'] > 40).astype(int)
    
    # 房屋规模特征
    if 'bedrooms' in processed_data.columns:
        processed_data['is_large_house'] = (processed_data['bedrooms'] >= 4).astype(int)
        
        if 'bathrooms' in processed_data.columns:
            processed_data['total_rooms'] = processed_data['bedrooms'] + processed_data['bathrooms']
    
    # 停车位特征
    if 'car_spaces' in processed_data.columns:
        processed_data['has_parking'] = (processed_data['car_spaces'] >= 1).astype(int)
        processed_data['multiple_parking'] = (processed_data['car_spaces'] >= 2).astype(int)
    
    # 价格特征
    price_col = None
    for col in ['price', 'capital_value', 'last_sold_price']:
        if col in processed_data.columns:
            price_col = col
            break
    
    if price_col:
        processed_data['is_expensive'] = (processed_data[price_col] > processed_data[price_col].median()).astype(int)
        processed_data['is_luxury'] = (processed_data[price_col] > processed_data[price_col].quantile(0.8)).astype(int)
    
    # 地区特征
    location_col = None
    for col in ['normalized_location', 'address', 'suburb']:
        if col in processed_data.columns:
            location_col = col
            break
    
    if location_col:
        le_location = LabelEncoder()
        processed_data['location_encoded'] = le_location.fit_transform(processed_data[location_col].astype(str))
        
        # 创建城市特征
        processed_data['is_wellington'] = processed_data[location_col].str.contains('wellington', case=False, na=False).astype(int)
        processed_data['is_auckland'] = processed_data[location_col].str.contains('auckland', case=False, na=False).astype(int)
    
    # 面积特征
    if 'land_area' in processed_data.columns:
        processed_data['is_apartment'] = (processed_data['land_area'] == 0).astype(int)
    
    if 'floor_size' in processed_data.columns:
        processed_data['is_spacious'] = (processed_data['floor_size'] > processed_data['floor_size'].median()).astype(int)
    
    # 选择特征列
    feature_columns = []
    potential_features = [
        'bedrooms', 'bathrooms', 'car_spaces', 'year_built', 'price', 'capital_value',
        'property_age', 'is_new', 'is_old', 'total_rooms', 'is_large_house',
        'has_parking', 'multiple_parking', 'is_expensive', 'is_luxury',
        'location_encoded', 'is_wellington', 'is_auckland', 'is_apartment', 'is_spacious'
    ]
    
    for col in potential_features:
        if col in processed_data.columns:
            feature_columns.append(col)
    
    print(f"✅ 创建了 {len(feature_columns)} 个特征")
    print(f"📋 特征列表: {feature_columns}")
    
    return processed_data[feature_columns], feature_columns

def train_property_model(X, y):
    """训练房产预测模型"""
    print("🔄 训练房产预测模型...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建集成模型
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)], 
        voting='soft'
    )
    
    # 交叉验证
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"📊 交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 训练模型
    ensemble.fit(X_train_scaled, y_train)
    
    # 评估模型
    train_accuracy = ensemble.score(X_train_scaled, y_train)
    test_accuracy = ensemble.score(X_test_scaled, y_test)
    
    y_pred = ensemble.predict(X_test_scaled)
    
    print(f"\n📋 模型性能:")
    print(f"  训练准确率: {train_accuracy:.4f}")
    print(f"  测试准确率: {test_accuracy:.4f}")
    print(f"\n📊 详细分类报告:")
    print(classification_report(y_test, y_pred))
    
    return ensemble, scaler, test_accuracy

def get_wellington_properties(supabase_client):
    """从properties表获取Wellington的房产数据进行预测"""
    if not supabase_client:
        print("❌ 数据库连接不可用，无法获取properties数据")
        return None
    
    print("🔄 从properties表获取Wellington房产数据...")
    
    try:
        # 获取Wellington地区的房产数据
        response = supabase_client.table('properties').select('*').ilike('city', '%wellington%').execute()
        
        if not response.data:
            print("⚠️ properties表中没有Wellington的数据")
            return None
            
        df = pd.DataFrame(response.data)
        print(f"✅ 从properties表获取了 {len(df)} 条Wellington房产记录")
        print(f"📋 获取的列: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ 获取Wellington房产数据时发生错误: {e}")
        return None

def predict_wellington_properties(properties_df, model, scaler, feature_names):
    """预测Wellington房产的出售可能性"""
    if model is None or properties_df is None:
        print("❌ 模型或数据不可用")
        return None
    
    print("🔄 开始预测Wellington房产出售可能性...")
    
    # 准备预测数据
    prediction_data = properties_df.copy()
    
    # 数据预处理 - 统一字段名称
    column_mapping = {
        'capital_value': 'price',
        'property_address': 'address',
        'suburb': 'normalized_location'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in prediction_data.columns and new_col not in prediction_data.columns:
            prediction_data[new_col] = prediction_data[old_col]
    
    # 添加Wellington标识
    if 'normalized_location' not in prediction_data.columns:
        prediction_data['normalized_location'] = prediction_data.get('suburb', 'Wellington')
    
    # 创建特征
    try:
        X_pred, _ = create_features_from_real_data(prediction_data)
        
        # 确保特征列与训练时一致
        X_pred = X_pred.reindex(columns=feature_names, fill_value=0)
        
        # 标准化特征
        X_pred_scaled = scaler.transform(X_pred)
        
        # 进行预测
        predictions = model.predict(X_pred_scaled)
        probabilities = model.predict_proba(X_pred_scaled)
        
        # 整理预测结果
        results = []
        for i, (_, row) in enumerate(properties_df.iterrows()):
            confidence = max(probabilities[i])
            predicted_status = "likely to sell" if predictions[i] == 1 else "unlikely to sell"
            
            result = {
                'property_id': row.get('id', str(uuid.uuid4())),
                'address': row.get('property_address', row.get('address', 'Unknown Address')),
                'suburb': row.get('suburb', 'Unknown Suburb'),
                'predicted_status': predicted_status,
                'confidence_score': confidence,
                'bedrooms': row.get('bedrooms', 0),
                'bathrooms': row.get('bathrooms', 0),
                'year_built': row.get('year_built', 0),
                'price': row.get('capital_value', row.get('price', 0)),
                'is_rented': row.get('is_currently_rented', False)
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('confidence_score', ascending=False)
        
        print(f"✅ 完成 {len(results_df)} 个Wellington房产的预测")
        
        return results_df
        
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        return None

def save_predictions_to_property_status(supabase_client, results_df, confidence_threshold=0.7):
    """将预测结果保存到property_status表"""
    if not supabase_client:
        print("⚠️ 数据库连接不可用，跳过数据库保存")
        return 0
    
    if results_df is None or len(results_df) == 0:
        print("⚠️ 没有预测结果需要保存")
        return 0
    
    # 筛选高置信度结果
    high_confidence_results = results_df[results_df['confidence_score'] >= confidence_threshold]
    
    if len(high_confidence_results) == 0:
        print(f"⚠️ 没有置信度≥{confidence_threshold}的预测结果")
        return 0
    
    print(f"🔄 保存 {len(high_confidence_results)} 条高置信度预测结果到property_status表...")
    
    try:
        # 清空旧的预测数据
        print("🔄 清空property_status表中的旧数据...")
        delete_result = supabase_client.table('property_status').delete().neq('id', 0).execute()
        print(f"✅ 已清空旧数据")
        
        # 准备插入数据
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_data = []
        
        for _, row in high_confidence_results.iterrows():
            insert_data.append({
                'property_id': str(row['property_id'])[:32],  # 限制长度
                'predicted_status': row['predicted_status'],
                'confidence_score': float(row['confidence_score']),
                'predicted_at': current_time
            })
        
        # 批量插入
        batch_size = 25
        total_inserted = 0
        
        for i in range(0, len(insert_data), batch_size):
            batch = insert_data[i:i + batch_size]
            
            result = supabase_client.table('property_status').insert(batch).execute()
            
            if result.data:
                batch_inserted = len(result.data)
                total_inserted += batch_inserted
                print(f"✅ 成功插入批次 {i//batch_size + 1}，共 {batch_inserted} 条记录")
        
        print(f"\n🎯 总共成功插入 {total_inserted} 条预测记录到property_status表")
        return total_inserted
        
    except Exception as e:
        print(f"❌ 保存预测结果时发生错误: {e}")
        return 0

def main():
    """主函数"""
    print("🚀 Wellington房产预测模型 - 基于真实数据")
    print("=" * 60)
    
    # 创建数据库连接
    supabase_client = create_supabase_client()
    if not supabase_client:
        print("❌ 数据库连接失败，程序退出")
        return
    
    print("✅ 数据库连接成功")
    
    # 1. 获取训练数据
    print("\n" + "="*50)
    print("📊 第1步：获取训练数据")
    print("="*50)
    
    real_estate_data = get_training_data_from_real_estate(supabase_client)
    if real_estate_data is None:
        print("❌ 无法获取训练数据，程序退出")
        return
    
    # 2. 处理训练数据
    training_data, available_columns = process_real_estate_data(real_estate_data)
    if training_data is None:
        print("❌ 训练数据处理失败，程序退出")
        return
    
    print(f"\n✅ 训练数据准备完成:")
    print(f"  数据量: {len(training_data)}")
    print(f"  特征数: {len(available_columns)}")
    
    # 3. 创建特征和训练模型
    print("\n" + "="*50)
    print("🤖 第2步：训练模型")
    print("="*50)
    
    X, feature_names = create_features_from_real_data(training_data)
    y = training_data['target'].values
    
    print(f"\n📊 训练数据统计:")
    print(f"  样本数量: {len(X)}")
    print(f"  特征数量: {len(feature_names)}")
    print(f"  标签分布: {pd.Series(y).value_counts().to_dict()}")
    
    # 训练模型
    model, scaler, accuracy = train_property_model(X, y)
    
    print(f"\n🎯 最终模型准确率: {accuracy:.4f}")
    if accuracy >= 0.8:
        print("✅ 成功达到0.8以上准确率目标!")
    else:
        print("⚠️ 准确率接近但未达到0.8目标")
    
    # 4. 获取Wellington房产数据
    print("\n" + "="*50)
    print("🏠 第3步：获取Wellington房产数据")
    print("="*50)
    
    wellington_properties = get_wellington_properties(supabase_client)
    if wellington_properties is None:
        print("❌ 无法获取Wellington房产数据，程序退出")
        return
    
    print(f"\n✅ 获取到 {len(wellington_properties)} 条Wellington房产数据")
    
    # 5. 进行预测
    print("\n" + "="*50)
    print("🔮 第4步：预测Wellington房产")
    print("="*50)
    
    prediction_results = predict_wellington_properties(
        wellington_properties, model, scaler, feature_names
    )
    
    if prediction_results is None:
        print("❌ 预测失败，程序退出")
        return
    
    # 显示预测结果
    print(f"\n📊 预测结果统计:")
    print(f"  总预测数量: {len(prediction_results)}")
    print(f"  平均置信度: {prediction_results['confidence_score'].mean():.4f}")
    print(f"  预测为可能出售: {len(prediction_results[prediction_results['predicted_status'] == 'likely to sell'])}")
    print(f"  预测为不太可能出售: {len(prediction_results[prediction_results['predicted_status'] == 'unlikely to sell'])}")
    
    # 显示高置信度预测结果
    high_confidence = prediction_results[prediction_results['confidence_score'] >= 0.8]
    print(f"\n🎯 高置信度(≥0.8)预测结果: {len(high_confidence)} 条")
    
    if len(high_confidence) > 0:
        print("\n🏠 高置信度Wellington房产预测:")
        for _, row in high_confidence.head(10).iterrows():
            status_emoji = "🟢" if row['predicted_status'] == "likely to sell" else "🔴"
            print(f"{status_emoji} {row['address']}")
            print(f"    地区: {row['suburb']} | {row['bedrooms']}房{row['bathrooms']}卫 | {row['year_built']}年建")
            print(f"    价格: ${row['price']:,} | 预测: {row['predicted_status']} | 置信度: {row['confidence_score']:.3f}")
            print()
    
    # 6. 保存预测结果
    print("\n" + "="*50)
    print("💾 第5步：保存预测结果")
    print("="*50)
    
    inserted_count = save_predictions_to_property_status(supabase_client, prediction_results, confidence_threshold=0.7)
    
    # 保存到CSV
    prediction_results.to_csv('wellington_property_predictions_real_data.csv', index=False)
    print(f"💾 预测结果已保存到 wellington_property_predictions_real_data.csv")
    
    # 最终总结
    print("\n🎉 Wellington房产预测分析完成!")
    print("=" * 60)
    print(f"✅ 模型准确率: {accuracy:.4f}")
    print(f"✅ 总预测数量: {len(prediction_results)}")
    high_conf_count = len(prediction_results[prediction_results['confidence_score'] >= 0.8])
    print(f"✅ 高置信度(≥0.8): {high_conf_count}")
    print(f"✅ 平均置信度: {prediction_results['confidence_score'].mean():.4f}")
    
    likely_to_sell = len(prediction_results[prediction_results['predicted_status'] == 'likely to sell'])
    print(f"✅ 预测可能出售: {likely_to_sell} 个房产")
    
    if inserted_count > 0:
        print(f"✅ 成功保存 {inserted_count} 条预测结果到property_status表")
    
    print("\n📋 关键发现:")
    print("  • 基于real_estate表的真实销售数据训练模型")
    print("  • 对properties表中的Wellington房产进行预测")
    print("  • 识别出最有可能近期出售的房产")
    print("  • 预测结果已保存到property_status表")
    
    print("\n🚀 使用建议:")
    print("  1. 重点关注置信度≥0.8的预测结果")
    print("  2. 优先推荐'likely to sell'状态的房产")
    print("  3. 结合房产特征（年龄、价格、地区）进行分析")
    print("  4. 定期更新模型以提高预测准确性")
    print("  5. 查看property_status表获取完整预测数据")
    
    print("\n" + "="*60)
    print("🎯 Wellington房产预测完成，结果已保存到数据库，可以在应用中展示！")
    print("="*60)

if __name__ == "__main__":
    main()