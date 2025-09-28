"""
Wellington房产预测 - Colab简化版本
适合在Google Colab中直接运行的Python脚本
"""

# 安装依赖 (在Colab中运行)
# !pip install pandas numpy scikit-learn joblib matplotlib seaborn

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

warnings.filterwarnings("ignore")

def create_training_data():
    """创建训练数据"""
    print("🔄 创建训练数据...")
    np.random.seed(42)
    
    suburbs = {
        'Oriental Bay': {'base_price': 2000000, 'sale_rate': 0.90},
        'Thorndon': {'base_price': 1500000, 'sale_rate': 0.85},
        'Kelburn': {'base_price': 1300000, 'sale_rate': 0.80},
        'Khandallah': {'base_price': 1400000, 'sale_rate': 0.82},
        'Wellington Central': {'base_price': 1000000, 'sale_rate': 0.75},
        'Mount Victoria': {'base_price': 1100000, 'sale_rate': 0.70},
        'Te Aro': {'base_price': 900000, 'sale_rate': 0.65},
        'Newtown': {'base_price': 700000, 'sale_rate': 0.40},
        'Island Bay': {'base_price': 800000, 'sale_rate': 0.45},
        'Karori': {'base_price': 950000, 'sale_rate': 0.55}
    }
    
    data = []
    for i in range(2000):
        suburb = np.random.choice(list(suburbs.keys()))
        suburb_info = suburbs[suburb]
        
        year_built = np.random.randint(1950, 2024)
        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.2, 0.35, 0.3, 0.08, 0.02])
        bathrooms = min(bedrooms, np.random.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.25, 0.05]))
        car_spaces = np.random.choice([0, 1, 2, 3], p=[0.15, 0.4, 0.35, 0.1])
        
        floor_size = max(50, 60 + bedrooms * 25 + np.random.randint(-20, 30))
        
        if suburb in ['Wellington Central', 'Te Aro']:
            land_area = 0 if np.random.random() < 0.6 else np.random.randint(200, 400)
        else:
            land_area = np.random.randint(300, 1000)
        
        # 价格计算
        base_price = suburb_info['base_price']
        property_age = 2024 - year_built
        
        if property_age < 5:
            age_factor = 1.2
        elif property_age < 15:
            age_factor = 1.1
        elif property_age < 30:
            age_factor = 1.0
        elif property_age < 50:
            age_factor = 0.9
        else:
            age_factor = 0.8
        
        size_factor = 1 + (floor_size - 120) * 0.005
        bedroom_factor = 1 + (bedrooms - 3) * 0.12
        
        last_sold_price = int(base_price * age_factor * size_factor * bedroom_factor * np.random.uniform(0.85, 1.15))
        capital_value = int(last_sold_price * np.random.uniform(0.95, 1.25))
        
        land_value = int(capital_value * np.random.uniform(0.4, 0.7)) if land_area > 0 else 0
        improvement_value = capital_value - land_value
        
        has_rental_history = np.random.random() < 0.35
        is_currently_rented = np.random.random() < 0.25 if has_rental_history else False
        
        # 目标变量计算
        sale_probability = suburb_info['sale_rate']
        
        # 影响因子
        if property_age < 5:
            sale_probability += 0.35
        elif property_age < 15:
            sale_probability += 0.25
        elif property_age > 60:
            sale_probability -= 0.30
        
        if last_sold_price > 2000000:
            sale_probability += 0.30
        elif last_sold_price < 600000:
            sale_probability -= 0.25
        
        if bedrooms >= 5:
            sale_probability += 0.25
        elif bedrooms <= 1:
            sale_probability -= 0.20
        
        if car_spaces >= 3:
            sale_probability += 0.20
        elif car_spaces == 0:
            sale_probability -= 0.25
        
        if is_currently_rented:
            sale_probability -= 0.50
        elif has_rental_history and not is_currently_rented:
            sale_probability += 0.15
        
        sale_probability = np.clip(sale_probability, 0.05, 0.95)
        
        if sale_probability > 0.8:
            target = 1 if np.random.random() < 0.95 else 0
        elif sale_probability > 0.6:
            target = 1 if np.random.random() < 0.85 else 0
        elif sale_probability > 0.4:
            target = 1 if np.random.random() < sale_probability else 0
        else:
            target = 1 if np.random.random() < 0.15 else 0
        
        data.append({
            'suburb': suburb, 'year_built': year_built, 'bedrooms': bedrooms,
            'bathrooms': bathrooms, 'car_spaces': car_spaces, 'floor_size': floor_size,
            'land_area': land_area, 'last_sold_price': last_sold_price,
            'capital_value': capital_value, 'land_value': land_value,
            'improvement_value': improvement_value, 'has_rental_history': has_rental_history,
            'is_currently_rented': is_currently_rented, 'target': target
        })
    
    df = pd.DataFrame(data)
    print(f"✅ 创建了 {len(df)} 条训练数据")
    return df

def create_features(data):
    """创建特征"""
    print("🔄 创建特征...")
    
    processed_data = data.copy()
    
    # 数值特征处理
    numeric_columns = ['year_built', 'bedrooms', 'bathrooms', 'car_spaces', 
                      'floor_size', 'land_area', 'last_sold_price', 
                      'capital_value', 'land_value', 'improvement_value']
    
    for col in numeric_columns:
        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        processed_data[col] = processed_data[col].fillna(processed_data[col].median())
    
    # 布尔特征
    boolean_columns = ['has_rental_history', 'is_currently_rented']
    for col in boolean_columns:
        processed_data[col] = processed_data[col].astype(bool).astype(int)
    
    # Suburb特征
    le_suburb = LabelEncoder()
    processed_data['suburb_encoded'] = le_suburb.fit_transform(processed_data['suburb'].astype(str))
    
    suburb_tiers = {
        'Oriental Bay': 10, 'Thorndon': 9, 'Kelburn': 8, 'Khandallah': 8,
        'Wellington Central': 6, 'Mount Victoria': 6, 'Karori': 5,
        'Te Aro': 4, 'Island Bay': 3, 'Newtown': 2
    }
    processed_data['suburb_tier'] = processed_data['suburb'].map(suburb_tiers).fillna(3)
    
    # 时间特征
    current_year = datetime.now().year
    processed_data['property_age'] = current_year - processed_data['year_built']
    processed_data['is_very_new'] = (processed_data['property_age'] < 5).astype(int)
    processed_data['is_new'] = (processed_data['property_age'] < 15).astype(int)
    processed_data['is_old'] = (processed_data['property_age'] > 40).astype(int)
    
    # 房屋特征
    processed_data['total_rooms'] = processed_data['bedrooms'] + processed_data['bathrooms']
    processed_data['is_large_house'] = (processed_data['bedrooms'] >= 4).astype(int)
    processed_data['has_parking'] = (processed_data['car_spaces'] >= 1).astype(int)
    processed_data['multiple_parking'] = (processed_data['car_spaces'] >= 2).astype(int)
    
    # 面积特征
    processed_data['is_apartment'] = (processed_data['land_area'] == 0).astype(int)
    processed_data['is_spacious'] = (processed_data['floor_size'] > 150).astype(int)
    
    # 价格特征
    processed_data['price_per_sqm'] = processed_data['last_sold_price'] / processed_data['floor_size']
    processed_data['is_expensive'] = (processed_data['last_sold_price'] > 1200000).astype(int)
    processed_data['is_luxury'] = (processed_data['last_sold_price'] >= 2000000).astype(int)
    
    # 综合评分
    processed_data['luxury_score'] = (
        processed_data['suburb_tier'] * 1.5 +
        processed_data['is_very_new'] * 4 +
        processed_data['is_new'] * 2 +
        processed_data['is_large_house'] * 2 +
        processed_data['multiple_parking'] * 2 +
        processed_data['is_luxury'] * 3 -
        processed_data['is_currently_rented'] * 6 -
        processed_data['is_old'] * 3
    )
    
    # 选择特征
    feature_columns = [
        'year_built', 'bedrooms', 'bathrooms', 'car_spaces', 'floor_size', 'land_area',
        'last_sold_price', 'capital_value', 'land_value', 'improvement_value',
        'suburb_encoded', 'suburb_tier', 'has_rental_history', 'is_currently_rented',
        'property_age', 'is_very_new', 'is_new', 'is_old', 'total_rooms',
        'is_large_house', 'has_parking', 'multiple_parking', 'is_apartment',
        'is_spacious', 'price_per_sqm', 'is_expensive', 'is_luxury', 'luxury_score'
    ]
    
    available_features = [col for col in feature_columns if col in processed_data.columns]
    print(f"✅ 创建了 {len(available_features)} 个特征")
    
    return processed_data[available_features], available_features

def train_model(X, y):
    """训练模型"""
    print("🔄 训练模型...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 集成模型
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
    
    ensemble.fit(X_train_scaled, y_train)
    accuracy = ensemble.score(X_test_scaled, y_test)
    
    print(f"✅ 模型训练完成，准确率: {accuracy:.4f}")
    
    return ensemble, scaler, accuracy

def create_wellington_data():
    """创建Wellington测试数据"""
    return pd.DataFrame([
        {
            'suburb': 'Khandallah', 'year_built': 2020, 'bedrooms': 4, 'bathrooms': 3,
            'car_spaces': 2, 'floor_size': 200, 'land_area': 650, 'last_sold_price': 1800000,
            'capital_value': 1950000, 'land_value': 1200000, 'improvement_value': 750000,
            'has_rental_history': False, 'is_currently_rented': False,
            'address': "15 Agra Crescent, Khandallah, Wellington"
        },
        {
            'suburb': 'Oriental Bay', 'year_built': 2022, 'bedrooms': 3, 'bathrooms': 2,
            'car_spaces': 2, 'floor_size': 140, 'land_area': 0, 'last_sold_price': 2500000,
            'capital_value': 2600000, 'land_value': 0, 'improvement_value': 2600000,
            'has_rental_history': False, 'is_currently_rented': False,
            'address': "45 Oriental Parade, Oriental Bay, Wellington"
        },
        {
            'suburb': 'Newtown', 'year_built': 1965, 'bedrooms': 2, 'bathrooms': 1,
            'car_spaces': 0, 'floor_size': 85, 'land_area': 400, 'last_sold_price': 550000,
            'capital_value': 650000, 'land_value': 400000, 'improvement_value': 250000,
            'has_rental_history': True, 'is_currently_rented': True,
            'address': "78 Riddiford Street, Newtown, Wellington"
        },
        {
            'suburb': 'Kelburn', 'year_built': 2021, 'bedrooms': 5, 'bathrooms': 4,
            'car_spaces': 3, 'floor_size': 250, 'land_area': 800, 'last_sold_price': 2200000,
            'capital_value': 2300000, 'land_value': 1400000, 'improvement_value': 900000,
            'has_rental_history': False, 'is_currently_rented': False,
            'address': "23 Kelburn Parade, Kelburn, Wellington"
        },
        {
            'suburb': 'Wellington Central', 'year_built': 2019, 'bedrooms': 2, 'bathrooms': 2,
            'car_spaces': 1, 'floor_size': 95, 'land_area': 0, 'last_sold_price': 950000,
            'capital_value': 1000000, 'land_value': 0, 'improvement_value': 1000000,
            'has_rental_history': True, 'is_currently_rented': False,
            'address': "12 The Terrace, Wellington Central, Wellington"
        }
    ])

def main():
    """主函数"""
    print("🏠 Wellington房产预测开始...")
    
    # 创建训练数据
    training_data = create_training_data()
    
    # 创建特征
    X, feature_names = create_features(training_data)
    y = training_data['target'].values
    
    # 训练模型
    model, scaler, accuracy = train_model(X, y)
    
    # 创建Wellington测试数据
    wellington_data = create_wellington_data()
    
    # 预测
    print("🔄 开始预测...")
    X_wellington, _ = create_features(wellington_data)
    X_wellington = X_wellington.reindex(columns=feature_names, fill_value=0)
    X_wellington_scaled = scaler.transform(X_wellington)
    
    predictions = model.predict(X_wellington_scaled)
    probabilities = model.predict_proba(X_wellington_scaled)
    
    # 结果处理
    results = []
    for i, (_, row) in enumerate(wellington_data.iterrows()):
        confidence = max(probabilities[i])
        predicted_status = "for Sale" if predictions[i] == 1 else "not for Sale"
        
        results.append({
            'address': row['address'],
            'suburb': row['suburb'],
            'predicted_status': predicted_status,
            'confidence_score': confidence,
            'bedrooms': row['bedrooms'],
            'year_built': row['year_built'],
            'price': row['last_sold_price'],
            'is_rented': row['is_currently_rented']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('confidence_score', ascending=False)
    
    # 显示结果
    print(f"\n🎉 预测完成!")
    print(f"✅ 模型准确率: {accuracy:.4f}")
    print(f"✅ 总预测数量: {len(results_df)}")
    print(f"✅ 高置信度(≥0.8): {len(results_df[results_df['confidence_score'] >= 0.8])}")
    
    print(f"\n🏠 Wellington房产预测结果:")
    for _, row in results_df.iterrows():
        rent_status = "正在出租" if row['is_rented'] else "空置"
        status_emoji = "🟢" if row['predicted_status'] == "for Sale" else "🔴"
        print(f"\n{status_emoji} {row['address']}")
        print(f"    地区: {row['suburb']} | {row['bedrooms']}房 | {row['year_built']}年 | {rent_status}")
        print(f"    价格: ${row['price']:,}")
        print(f"    预测: {row['predicted_status']} | 置信度: {row['confidence_score']:.3f}")
    
    # 保存结果
    results_df.to_csv('wellington_predictions.csv', index=False)
    print(f"\n💾 结果已保存到 wellington_predictions.csv")
    
    return results_df

if __name__ == "__main__":
    results = main()