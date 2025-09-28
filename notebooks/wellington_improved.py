"""
改进的Wellington房产预测脚本
解决准确率低的问题，生成高置信度预测结果
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config.supabase_config import create_supabase_client

def get_supabase_client():
    return create_supabase_client()

class WellingtonPropertyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def engineer_features(self, df):
        """高级特征工程"""
        df = df.copy()
        
        # 处理数值列
        numeric_cols = ['year_built', 'bedrooms', 'bathrooms', 'car_spaces', 
                       'floor_size', 'land_area', 'last_sold_price', 'capital_value',
                       'land_value', 'improvement_value']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 使用中位数填充缺失值
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # 处理布尔列
        bool_cols = ['has_rental_history', 'is_currently_rented']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(int)
        
        # 创建新的组合特征
        try:
            # 价格相关特征
            if 'last_sold_price' in df.columns and 'capital_value' in df.columns:
                df['price_to_capital_ratio'] = df['last_sold_price'] / (df['capital_value'] + 1)
                df['value_appreciation'] = (df['capital_value'] - df['last_sold_price']) / (df['last_sold_price'] + 1)
            
            # 面积相关特征
            if 'floor_size' in df.columns and 'land_area' in df.columns:
                df['floor_to_land_ratio'] = df['floor_size'] / (df['land_area'] + 1)
            
            # 房间比例特征
            if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
                df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.5)
            
            # 房屋年龄特征
            if 'year_built' in df.columns:
                current_year = 2024
                df['property_age'] = current_year - df['year_built']
                df['is_new_property'] = (df['property_age'] <= 10).astype(int)
                df['is_old_property'] = (df['property_age'] >= 50).astype(int)
            
            # 每平米价格
            if 'last_sold_price' in df.columns and 'floor_size' in df.columns:
                df['price_per_sqm'] = df['last_sold_price'] / (df['floor_size'] + 1)
            
            # 价值密度
            if 'capital_value' in df.columns and 'land_area' in df.columns:
                df['value_density'] = df['capital_value'] / (df['land_area'] + 1)
                
        except Exception as e:
            print(f"特征工程警告: {e}")
        
        # 处理分类变量
        categorical_cols = ['suburb', 'p_status']
        for col in categorical_cols:
            if col in df.columns:
                # 填充缺失值
                df[col] = df[col].fillna('Unknown')
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # 处理新的类别
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else 'Unknown')
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def prepare_features(self, df):
        """准备特征矩阵"""
        df_processed = self.engineer_features(df)
        
        # 选择特征
        base_features = [
            'year_built', 'bedrooms', 'bathrooms', 'car_spaces',
            'floor_size', 'land_area', 'last_sold_price', 'capital_value',
            'land_value', 'improvement_value', 'has_rental_history',
            'is_currently_rented'
        ]
        
        # 添加工程特征
        engineered_features = [
            'price_to_capital_ratio', 'value_appreciation', 'floor_to_land_ratio',
            'bed_bath_ratio', 'property_age', 'is_new_property', 'is_old_property',
            'price_per_sqm', 'value_density'
        ]
        
        # 添加编码的分类特征
        encoded_features = [col for col in df_processed.columns if col.endswith('_encoded')]
        
        # 合并所有特征
        all_features = base_features + engineered_features + encoded_features
        
        # 过滤存在的特征
        available_features = [col for col in all_features if col in df_processed.columns]
        
        feature_matrix = df_processed[available_features]
        
        # 最终的缺失值处理
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        target = df_processed['status'] if 'status' in df_processed.columns else None
        
        return feature_matrix, target, available_features
    
    def train_enhanced_model(self, training_df):
        """训练增强模型"""
        print("开始训练增强模型...")
        
        # 过滤有标签的数据
        training_df = training_df.dropna(subset=['status'])
        print(f"有效训练数据: {len(training_df)} 条")
        
        if len(training_df) < 100:
            print("警告: 训练数据太少，可能影响模型性能")
        
        # 准备特征
        feature_matrix, target, feature_names = self.prepare_features(training_df)
        self.feature_names = feature_names
        
        print(f"使用特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")
        
        # 检查标签分布
        label_counts = target.value_counts()
        print(f"标签分布: {dict(label_counts)}")
        
        # 分割数据
        test_size = min(0.3, max(0.1, 500 / len(training_df)))  # 动态调整测试集大小
        
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, target, 
            test_size=test_size, 
            random_state=42, 
            stratify=target
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 尝试多个模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 测试集评估
            test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            
            print(f"{name} - 测试准确率: {test_accuracy:.4f}")
            print(f"{name} - 交叉验证准确率: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 选择最佳模型
            combined_score = (test_accuracy + cv_mean) / 2
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
                best_name = name
        
        self.model = best_model
        
        # 最终评估
        final_pred = self.model.predict(X_test_scaled)
        final_accuracy = accuracy_score(y_test, final_pred)
        
        print(f"\n=== 最佳模型: {best_name} ===")
        print(f"最终准确率: {final_accuracy:.4f}")
        print(f"综合评分: {best_score:.4f}")
        
        print("\n详细分类报告:")
        print(classification_report(y_test, final_pred))
        
        return final_accuracy
    
    def predict_wellington(self, wellington_df, min_confidence=0.7):
        """预测Wellington房产状态"""
        print(f"开始预测Wellington房产 (最小置信度: {min_confidence})...")
        
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train_enhanced_model()")
        
        # 准备特征
        feature_matrix, _, _ = self.prepare_features(wellington_df)
        
        # 标准化
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        # 预测
        predictions = self.model.predict(feature_matrix_scaled)
        probabilities = self.model.predict_proba(feature_matrix_scaled)
        
        # 计算置信度
        confidence_scores = np.max(probabilities, axis=1)
        
        # 创建结果DataFrame
        results = wellington_df.copy()
        results['predicted_status'] = predictions
        results['confidence_score'] = confidence_scores
        
        # 过滤高置信度结果
        high_confidence_results = results[confidence_scores >= min_confidence]
        
        print(f"总预测数量: {len(results)}")
        print(f"高置信度预测数量: {len(high_confidence_results)}")
        
        if len(high_confidence_results) > 0:
            print(f"平均置信度: {high_confidence_results['confidence_score'].mean():.4f}")
            print(f"置信度范围: {high_confidence_results['confidence_score'].min():.4f} - {high_confidence_results['confidence_score'].max():.4f}")
            
            # 预测分布
            pred_dist = high_confidence_results['predicted_status'].value_counts()
            print(f"预测分布: {dict(pred_dist)}")
        
        return high_confidence_results

def main():
    """主执行函数"""
    print("=== Wellington房产状态预测系统 (改进版) ===")
    
    try:
        # 初始化预测器
        predictor = WellingtonPropertyPredictor()
        
        # 连接数据库
        supabase = get_supabase_client()
        print("数据库连接成功")
        
        # 获取训练数据
        print("\n1. 获取训练数据...")
        training_response = supabase.table('properties').select('*').not_.is_('status', 'null').limit(15000).execute()
        
        if not training_response.data:
            print("错误: 无法获取训练数据")
            return
        
        training_df = pd.DataFrame(training_response.data)
        print(f"原始训练数据: {len(training_df)} 条")
        
        # 训练模型
        print("\n2. 训练模型...")
        accuracy = predictor.train_enhanced_model(training_df)
        
        # 保存模型
        model_path = 'wellington_enhanced_model.joblib'
        joblib.dump({
            'model': predictor.model,
            'scaler': predictor.scaler,
            'label_encoders': predictor.label_encoders,
            'feature_names': predictor.feature_names
        }, model_path)
        print(f"模型已保存: {model_path}")
        
        # 获取Wellington预测数据
        print("\n3. 获取Wellington待预测数据...")
        wellington_response = supabase.table('properties').select('*').eq('city', 'Wellington').is_('status', 'null').limit(2000).execute()
        
        if not wellington_response.data:
            print("没有找到Wellington待预测数据")
            return
        
        wellington_df = pd.DataFrame(wellington_response.data)
        print(f"Wellington待预测数据: {len(wellington_df)} 条")
        
        # 尝试不同置信度阈值
        confidence_thresholds = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
        
        best_results = None
        best_threshold = None
        
        for threshold in confidence_thresholds:
            print(f"\n4. 尝试置信度阈值: {threshold}")
            
            high_conf_results = predictor.predict_wellington(wellington_df, threshold)
            
            if len(high_conf_results) >= 10:  # 至少需要10条结果
                best_results = high_conf_results
                best_threshold = threshold
                print(f"找到足够的高质量预测结果，使用阈值: {threshold}")
                break
            elif len(high_conf_results) > 0:
                if best_results is None or len(high_conf_results) > len(best_results):
                    best_results = high_conf_results
                    best_threshold = threshold
        
        if best_results is not None and len(best_results) > 0:
            print(f"\n5. 存储预测结果 (阈值: {best_threshold})...")
            
            # 清空旧数据
            print("清空旧预测数据...")
            supabase.table('property_status').delete().neq('id', 0).execute()
            
            # 准备插入数据
            insert_data = []
            for _, row in best_results.iterrows():
                insert_data.append({
                    'property_id': row['id'],
                    'predicted_status': int(row['predicted_status']),
                    'confidence_score': float(row['confidence_score']),
                    'model_version': f'wellington_enhanced_v1.0_threshold_{best_threshold}',
                    'prediction_date': pd.Timestamp.now().isoformat(),
                    'city': 'Wellington',
                    'suburb': row.get('suburb', 'Unknown')
                })
            
            # 批量插入
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(insert_data), batch_size):
                batch = insert_data[i:i + batch_size]
                try:
                    supabase.table('property_status').insert(batch).execute()
                    total_inserted += len(batch)
                    print(f"已插入 {len(batch)} 条记录 (总计: {total_inserted})")
                except Exception as e:
                    print(f"插入批次失败: {e}")
            
            print(f"\n=== 预测完成 ===")
            print(f"成功插入 {total_inserted} 条Wellington预测结果")
            print(f"使用置信度阈值: {best_threshold}")
            print(f"平均置信度: {best_results['confidence_score'].mean():.4f}")
            
            # 显示样本结果
            print("\n样本预测结果:")
            sample = best_results[['property_address', 'suburb', 'predicted_status', 'confidence_score']].head(10)
            for _, row in sample.iterrows():
                status_text = "Active" if row['predicted_status'] == 1 else "Inactive"
                addr = str(row['property_address'])[:40] if pd.notna(row['property_address']) else "Unknown"
                suburb = str(row['suburb'])[:15] if pd.notna(row['suburb']) else "Unknown"
                print(f"  {addr:<40} | {suburb:<15} | {status_text:<8} | {row['confidence_score']:.3f}")
        
        else:
            print("\n没有生成足够的高置信度预测结果")
            print("建议:")
            print("1. 增加更多训练数据")
            print("2. 检查数据质量")
            print("3. 调整特征工程策略")
        
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()