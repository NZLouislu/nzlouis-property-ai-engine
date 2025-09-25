import os
from supabase import create_client, Client
import pandas as pd
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    print("错误: 请检查 .env 文件中的 SUPABASE_URL 和 SUPABASE_ANON_KEY")
    exit(1)

supabase: Client = create_client(url, key)

print("=== 房产预测系统专家分析报告 ===\n")

print("1. 数据质量分析")
print("-" * 50)

try:
    properties = supabase.table('properties').select('*').execute()
    df = pd.DataFrame(properties.data)
    
    if len(df) > 0:
        print(f"总数据量: {len(df)} 条记录")
        print(f"数据时间范围: {df['created_at'].min()} 到 {df['created_at'].max()}")
        
        print("\n关键特征缺失值分析:")
        key_features = ['bedrooms', 'bathrooms', 'floor_size', 'land_area', 'last_sold_price', 'capital_value']
        for feature in key_features:
            if feature in df.columns:
                missing_count = df[feature].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                print(f"  {feature}: {missing_count} 缺失 ({missing_pct:.1f}%)")
        
        print("\n数据分布分析:")
        if 'last_sold_price' in df.columns:
            prices = df['last_sold_price'].dropna()
            if len(prices) > 0:
                print(f"  房价范围: ${prices.min():,.0f} - ${prices.max():,.0f}")
                print(f"  房价中位数: ${prices.median():,.0f}")
        
        if 'bedrooms' in df.columns:
            bedrooms = df['bedrooms'].dropna()
            if len(bedrooms) > 0:
                print(f"  卧室数量分布: {dict(bedrooms.value_counts().sort_index())}")
        
        print("\n地理分布:")
        if 'city' in df.columns:
            city_dist = df['city'].value_counts()
            print(f"  城市分布: {dict(city_dist)}")
        
        if 'suburb' in df.columns:
            suburb_count = df['suburb'].nunique()
            print(f"  涉及郊区数量: {suburb_count}")
    
    else:
        print("警告: Properties表中没有数据!")

except Exception as e:
    print(f"数据分析错误: {e}")

print("\n2. 预测模型架构分析")
print("-" * 50)

try:
    training_data = supabase.table('properties_with_is_listed').select('*').execute()
    prediction_data = supabase.table('properties_to_predict').select('*').execute()
    
    train_df = pd.DataFrame(training_data.data)
    pred_df = pd.DataFrame(prediction_data.data)
    
    print(f"训练数据量: {len(train_df)} 条")
    print(f"预测数据量: {len(pred_df)} 条")
    
    if len(train_df) > 0:
        print("\n训练数据标签分布:")
        if 'is_listed' in train_df.columns:
            label_dist = train_df['is_listed'].value_counts()
            print(f"  已上市: {label_dist.get(True, 0)} 条")
            print(f"  未上市: {label_dist.get(False, 0)} 条")
            
            if len(label_dist) > 1:
                balance_ratio = min(label_dist) / max(label_dist)
                print(f"  数据平衡度: {balance_ratio:.2f} (1.0为完全平衡)")
                if balance_ratio < 0.3:
                    print("  警告: 数据严重不平衡，建议使用SMOTE或调整类权重")

except Exception as e:
    print(f"模型数据分析错误: {e}")

print("\n3. 特征工程评估")
print("-" * 50)

feature_recommendations = [
    "价格相关特征:",
    "  - 房价/土地面积比率 (price_per_land_sqm)",
    "  - 房价/建筑面积比率 (price_per_floor_sqm)", 
    "  - 资本价值/改善价值比率 (capital_improvement_ratio)",
    "",
    "时间特征:",
    "  - 距离上次销售天数 (days_since_last_sale)",
    "  - 房屋年龄 (property_age)",
    "  - 销售季节性特征 (sale_month, sale_quarter)",
    "",
    "地理特征:",
    "  - 郊区平均房价 (suburb_avg_price)",
    "  - 郊区房价排名 (suburb_price_rank)",
    "  - 距离CBD距离 (distance_to_cbd)",
    "",
    "房产特征:",
    "  - 房间总数 (total_rooms = bedrooms + bathrooms)",
    "  - 房间密度 (room_density = total_rooms / floor_size)",
    "  - 土地利用率 (land_utilization = floor_size / land_area)"
]

for rec in feature_recommendations:
    print(rec)

print("\n4. 模型性能优化建议")
print("-" * 50)

optimization_suggestions = [
    "数据增强策略:",
    "  - 收集更多历史数据 (目标: >1000条记录)",
    "  - 增加外部数据源 (学校评分、交通便利性、犯罪率)",
    "  - 使用SMOTE处理类别不平衡",
    "",
    "模型选择:",
    "  - 尝试集成方法: XGBoost, LightGBM, CatBoost",
    "  - 考虑深度学习: TabNet, Wide&Deep",
    "  - 实现模型堆叠 (Stacking) 提高准确性",
    "",
    "验证策略:",
    "  - 使用时间序列分割 (TimeSeriesSplit)",
    "  - 实现交叉验证 (StratifiedKFold)",
    "  - 添加业务指标评估 (精确率、召回率、F1-score)",
    "",
    "部署优化:",
    "  - 实现模型版本管理",
    "  - 添加模型监控和漂移检测",
    "  - 设置自动重训练机制"
]

for suggestion in optimization_suggestions:
    print(suggestion)

print("\n5. 系统架构改进建议")
print("-" * 50)

architecture_improvements = [
    "数据管道:",
    "  - 实现ETL流水线自动化",
    "  - 添加数据质量检查",
    "  - 设置异常数据告警",
    "",
    "模型管道:",
    "  - 使用MLflow进行实验跟踪",
    "  - 实现A/B测试框架",
    "  - 添加模型解释性工具 (SHAP, LIME)",
    "",
    "API设计:",
    "  - 实现RESTful API",
    "  - 添加批量预测接口",
    "  - 实现预测置信度返回"
]

for improvement in architecture_improvements:
    print(improvement)

print("\n6. 立即可执行的改进措施")
print("-" * 50)

immediate_actions = [
    "高优先级 (本周内):",
    "  ✓ 增加数据收集频率和覆盖范围",
    "  ✓ 实现基础特征工程",
    "  ✓ 添加模型评估指标",
    "",
    "中优先级 (本月内):",
    "  ✓ 尝试不同算法并比较性能",
    "  ✓ 实现交叉验证",
    "  ✓ 添加预测置信度",
    "",
    "低优先级 (长期):",
    "  ✓ 集成外部数据源",
    "  ✓ 实现深度学习模型",
    "  ✓ 构建完整MLOps流水线"
]

for action in immediate_actions:
    print(action)

print(f"\n分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")