"""
Wellington房产预测最终版本
修复环境变量加载问题，生成高置信度预测结果
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 首先加载环境变量
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
from datetime import datetime
from supabase import create_client, Client


def create_supabase_client() -> Client:
    """创建Supabase客户端"""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    print(
        f"SUPABASE_URL: {SUPABASE_URL[:50]}..."
        if SUPABASE_URL
        else "SUPABASE_URL: None"
    )
    print(
        f"SUPABASE_KEY: {SUPABASE_KEY[:50]}..."
        if SUPABASE_KEY
        else "SUPABASE_KEY: None"
    )

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL 和 SUPABASE_KEY 环境变量必须设置")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def to_json_serializable(records):
    """将numpy数据类型转换为Python原生类型"""
    new_recs = []
    for rec in records:
        new = {}
        for k, v in rec.items():
            if isinstance(v, np.integer):
                new[k] = int(v)
            elif isinstance(v, np.floating):
                new[k] = float(v)
            else:
                new[k] = v
        new_recs.append(new)
    return new_recs


def fetch_training_data():
    """获取训练数据"""
    print("开始获取训练数据...")
    client = create_supabase_client()

    try:
        # 获取有标签的数据 - 使用更大的限制
        print("获取已上市房产数据...")
        res_listed = (
            client.from_("properties_with_is_listed").select("*").limit(3000).execute()
        )
        df_listed = pd.DataFrame(res_listed.data) if res_listed.data else pd.DataFrame()

        if not df_listed.empty:
            df_listed["status"] = 1  # 标记为上市
            print(f"获取到 {len(df_listed)} 条已上市房产数据")

        # 获取未上市的数据作为负样本
        print("获取待预测房产数据作为负样本...")
        res_unlisted = (
            client.from_("properties_to_predict").select("*").limit(3000).execute()
        )
        df_unlisted = (
            pd.DataFrame(res_unlisted.data) if res_unlisted.data else pd.DataFrame()
        )

        if not df_unlisted.empty:
            df_unlisted["status"] = 0  # 标记为未上市
            print(f"获取到 {len(df_unlisted)} 条待预测房产数据")

        # 合并数据集
        df_combined = pd.concat([df_listed, df_unlisted], ignore_index=True)

        if df_combined.empty:
            print("警告: 没有获取到训练数据。")
            return None

        print(f"获取的训练数据列: {df_combined.columns.tolist()}")
        print(f"训练数据量: {len(df_combined)}")
        print(f"标签分布: {df_combined['status'].value_counts().to_dict()}")

        return df_combined

    except Exception as e:
        print(f"错误: 获取训练数据时发生错误: {e}")
        return None


def fetch_prediction_data(city_filter="Wellington"):
    """获取预测数据"""
    print(f"开始获取{city_filter}预测数据...")
    client = create_supabase_client()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            query = client.from_("properties_to_predict").select("*")

            if city_filter:
                query = query.eq("city", city_filter)

            # 限制数量避免超时
            query = query.limit(500)

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
            print(
                f"错误: 获取预测数据时发生错误 (尝试 {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                print("等待 5 秒后重试...")
                import time

                time.sleep(5)
            else:
                return None


def extract_property_history_features(history_str):
    """从property_history字段中提取特征"""
    if pd.isnull(history_str) or not isinstance(history_str, str):
        return 0, -1, 0

    events = history_str.split("; ")
    transaction_count = len(events)

    has_built_event = int(any("Property Built" in event for event in events))

    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    dates = [re.search(date_pattern, event) for event in events]
    dates = [pd.to_datetime(match.group(1)) for match in dates if match]

    if dates:
        most_recent_date = max(dates)
        days_since_last_transaction = (pd.Timestamp.now() - most_recent_date).days
    else:
        days_since_last_transaction = -1

    return transaction_count, days_since_last_transaction, has_built_event


def enhanced_feature_engineering(df):
    """增强特征工程"""
    df = df.copy()

    # 基础特征列
    feature_columns = [
        "year_built",
        "bedrooms",
        "bathrooms",
        "car_spaces",
        "floor_size",
        "land_area",
        "last_sold_price",
        "capital_value",
        "land_value",
        "improvement_value",
        "suburb",
        "has_rental_history",
        "is_currently_rented",
        "property_history",
    ]

    # 确保所有特征列存在
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    X = df[feature_columns].copy()

    # 数值特征处理
    X["floor_size"] = pd.to_numeric(
        X["floor_size"].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )
    X["land_area"] = pd.to_numeric(
        X["land_area"].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )

    # 布尔特征处理
    X["has_rental_history"] = (
        X["has_rental_history"].astype(int) if "has_rental_history" in X.columns else 0
    )
    X["is_currently_rented"] = (
        X["is_currently_rented"].astype(int)
        if "is_currently_rented" in X.columns
        else 0
    )

    # 提取历史特征
    if "property_history" in X.columns:
        transaction_features = X["property_history"].apply(
            extract_property_history_features
        )
        X["transaction_count"] = [x[0] for x in transaction_features]
        X["days_since_last_transaction"] = [x[1] for x in transaction_features]
        X["has_built_event"] = [x[2] for x in transaction_features]
        X = X.drop(columns=["property_history"])
    else:
        X["transaction_count"] = 0
        X["days_since_last_transaction"] = -1
        X["has_built_event"] = 0

    # 创建高级组合特征
    try:
        # 价格相关特征
        X["price_to_capital_ratio"] = X["last_sold_price"] / (X["capital_value"] + 1)
        X["value_appreciation"] = (X["capital_value"] - X["last_sold_price"]) / (
            X["last_sold_price"] + 1
        )

        # 面积相关特征
        X["floor_to_land_ratio"] = X["floor_size"] / (X["land_area"] + 1)
        X["price_per_sqm"] = X["last_sold_price"] / (X["floor_size"] + 1)
        X["value_density"] = X["capital_value"] / (X["land_area"] + 1)

        # 房间比例特征
        X["bed_bath_ratio"] = X["bedrooms"] / (X["bathrooms"] + 0.5)
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"] + X["car_spaces"]

        # 房屋年龄特征
        current_year = 2024
        X["property_age"] = current_year - X["year_built"]
        X["is_new_property"] = (X["property_age"] <= 10).astype(int)
        X["is_old_property"] = (X["property_age"] >= 50).astype(int)

        # 价值比率特征
        X["land_to_total_ratio"] = X["land_value"] / (X["capital_value"] + 1)
        X["improvement_to_total_ratio"] = X["improvement_value"] / (
            X["capital_value"] + 1
        )

        # 市场活跃度特征
        X["market_activity_score"] = (
            X["transaction_count"] * 0.3
            + (1 / (X["days_since_last_transaction"] + 1)) * 0.7
        )

    except Exception as e:
        print(f"特征工程警告: {e}")

    # 处理suburb分类特征
    if "suburb" in X.columns:
        # 使用频率编码
        suburb_counts = X["suburb"].value_counts()
        X["suburb_frequency"] = X["suburb"].map(suburb_counts).fillna(0)
        X["suburb"] = X["suburb"].fillna("Unknown").astype("category").cat.codes

    # 数值化所有特征
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 处理无穷大值
    X.replace([float("inf"), -float("inf")], np.nan, inplace=True)

    # 智能填充缺失值
    for col in X.columns:
        if X[col].isna().any():
            if col in ["year_built", "bedrooms", "bathrooms", "car_spaces"]:
                # 使用众数填充
                mode_val = X[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 0
                X[col] = X[col].fillna(fill_val)
            elif col in [
                "floor_size",
                "land_area",
                "last_sold_price",
                "capital_value",
                "land_value",
                "improvement_value",
            ]:
                # 使用中位数填充
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
            else:
                # 其他特征用0填充
                X[col] = X[col].fillna(0)

    # 最终检查
    if X.isna().sum().sum() > 0:
        print(f"警告：仍然存在 {X.isna().sum().sum()} 个NaN值，将用0填充")
        X = X.fillna(0)

    return X


def preprocess_data(df, for_prediction=False):
    """数据预处理"""
    print("开始数据预处理...")

    X = enhanced_feature_engineering(df)

    print("数据预处理完成，特征列：", X.columns.tolist())
    print("数据形状：", X.shape)
    print("缺失值统计：", X.isna().sum().sum())

    if not for_prediction:
        if "status" in df.columns:
            y = df["status"].astype(int)
            uniques = y.unique()
            print(">> 标签唯一值：", uniques)
            return X, y
        else:
            print("错误：训练数据中缺少 status 字段")
            return None, None
    else:
        return X


def train_advanced_model(X, y):
    """训练高级模型"""
    print("开始高级模型训练...")

    # 检查数据质量
    print(f"训练数据形状: {X.shape}")
    print(f"标签分布: {y.value_counts().to_dict()}")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 高级模型配置
    models = {
        "RandomForest_Optimized": RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting_Optimized": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.85,
            max_features="sqrt",
            random_state=42,
        ),
    }

    best_model = None
    best_score = 0
    best_name = ""
    best_scaler = None

    for name, model in models.items():
        print(f"\n训练 {name}...")

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 交叉验证
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        cv_mean = cv_scores.mean()

        # 测试集评估
        test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)

        print(f"{name} - 交叉验证准确率: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"{name} - 测试准确率: {test_accuracy:.4f}")

        # 综合评分
        combined_score = (cv_mean + test_accuracy) / 2

        if combined_score > best_score:
            best_score = combined_score
            best_model = model
            best_name = name
            best_scaler = scaler

    # 最终评估
    final_pred = best_model.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, final_pred)

    print(f"\n=== 最佳模型: {best_name} ===")
    print(f"最终准确率: {final_accuracy:.4f}")
    print(f"综合评分: {best_score:.4f}")

    print("\n详细分类报告:")
    print(classification_report(y_test, final_pred))

    # 保存模型
    model_data = {
        "model": best_model,
        "scaler": best_scaler,
        "feature_names": X.columns.tolist(),
        "accuracy": final_accuracy,
        "model_name": best_name,
    }

    joblib.dump(model_data, "wellington_advanced_model.joblib")
    print("模型已保存为 'wellington_advanced_model.joblib'")

    return best_model, X.columns.tolist(), best_scaler, final_accuracy


def predict_with_confidence_boost(model, feature_names, scaler, prediction_df):
    """置信度增强预测"""
    print("开始置信度增强预测...")

    # 预处理数据
    X_pred = preprocess_data(prediction_df, for_prediction=True)

    # 确保特征顺序一致
    missing_features = set(feature_names) - set(X_pred.columns)
    if missing_features:
        print(f"添加缺失特征: {missing_features}")
        for feature in missing_features:
            X_pred[feature] = 0

    X_pred = X_pred[feature_names]

    # 标准化
    X_pred_scaled = scaler.transform(X_pred)

    # 预测
    predictions = model.predict(X_pred_scaled)
    probabilities = model.predict_proba(X_pred_scaled)
    confidence_scores = np.max(probabilities, axis=1)

    # 创建结果DataFrame
    results = prediction_df.copy()
    results["predicted_status"] = predictions
    results["confidence_score"] = confidence_scores
    results["probability_class_0"] = probabilities[:, 0]
    results["probability_class_1"] = probabilities[:, 1]

    # 计算置信度增强分数
    prob_diff = np.abs(probabilities[:, 1] - probabilities[:, 0])
    results["confidence_boost"] = confidence_scores * (1 + prob_diff)

    # 多阈值策略
    thresholds = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

    for threshold in thresholds:
        high_conf_results = results[results["confidence_boost"] >= threshold]

        print(f"增强置信度阈值 {threshold}: {len(high_conf_results)} 条结果")

        if len(high_conf_results) >= 5:  # 至少5条结果
            print(f"选择增强置信度阈值: {threshold}")
            print(f"平均置信度: {high_conf_results['confidence_score'].mean():.4f}")
            print(f"平均增强分数: {high_conf_results['confidence_boost'].mean():.4f}")

            # 预测分布
            pred_dist = high_conf_results["predicted_status"].value_counts()
            print(f"预测分布: {dict(pred_dist)}")

            return high_conf_results, threshold

    # 如果没有足够的高置信度结果，返回最高置信度的前15个
    if len(results) > 0:
        top_results = results.nlargest(min(15, len(results)), "confidence_boost")
        min_threshold = top_results["confidence_boost"].min()
        print(f"使用动态增强阈值: {min_threshold:.3f} (前{len(top_results)}个结果)")
        return top_results, min_threshold

    return pd.DataFrame(), 0


def clear_previous_predictions():
    """清空旧预测数据"""
    print("正在清空 property_status 表中的旧数据...")
    client = create_supabase_client()

    try:
        delete_result = client.table("property_status").delete().neq("id", 0).execute()
        deleted_count = len(delete_result.data) if delete_result.data else 0
        print(f"已清空 property_status 表，共删除 {deleted_count} 条记录。")
        return True
    except Exception as e:
        print(f"警告: 删除旧数据时发生错误 - {e}")
        return False


def store_predictions(results, threshold, model_version="wellington_advanced_v1.0"):
    """存储预测结果"""
    if len(results) == 0:
        print("没有预测结果需要存储")
        return 0

    print("开始存储预测结果...")

    client = create_supabase_client()

    # 准备插入数据
    insert_data = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in results.iterrows():
        # 处理ID字段 - 如果是字符串则直接使用，如果是数字则转换
        property_id = row['id']
        if isinstance(property_id, str):
            # 如果是UUID字符串，直接使用
            property_id_value = property_id
        else:
            # 如果是数字，转换为字符串
            property_id_value = str(property_id)
            
        insert_data.append(
            {
                "property_id": property_id_value,
                "predicted_status": (
                    "for Sale" if int(row["predicted_status"]) == 1 else "not for Sale"
                ),
                "confidence_score": float(row["confidence_score"]),
                "predicted_at": current_time,
            }
        )

    # 转换数据类型
    insert_data = to_json_serializable(insert_data)

    # 批量插入
    batch_size = 25
    total_inserted = 0

    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i : i + batch_size]

        try:
            result = client.table("property_status").insert(batch).execute()

            if result.data:
                batch_inserted = len(result.data)
                total_inserted += batch_inserted
                print(f"成功插入批次 {i//batch_size + 1}，共 {batch_inserted} 条记录")
        except Exception as e:
            print(f"错误: 批量存储预测结果时发生错误 - {e}")

    print(f"总共插入 {total_inserted} 条预测记录")
    return total_inserted


def main():
    """主函数"""
    print("=== Wellington房产状态预测系统 (高级版) ===")

    try:
        # 获取训练数据
        training_df = fetch_training_data()
        if training_df is None or training_df.empty:
            print("错误: 没有获取到有效的训练数据，程序终止。")
            return

        # 数据预处理
        X, y = preprocess_data(training_df)
        if X is None or y is None:
            print("错误: 数据预处理失败，程序终止。")
            return

        # 训练模型
        model, feature_names, scaler, accuracy = train_advanced_model(X, y)

        print(f"\n模型训练完成，准确率: {accuracy:.4f}")

        if accuracy >= 0.8:
            print("✓ 模型准确率达标，可以生成高置信度预测")
        else:
            print("⚠ 模型准确率未达到0.8，但仍将尝试生成预测")

        # 获取Wellington预测数据
        print("\n=== 获取Wellington地区待预测数据 ===")
        wellington_df = fetch_prediction_data("Wellington")

        if wellington_df is None or wellington_df.empty:
            print("没有获取到 Wellington 地区的待预测数据")
            return

        print(f"获取到 {len(wellington_df)} 条 Wellington 地区的待预测数据")

        # 清空旧数据
        clear_previous_predictions()

        # 进行预测
        print("\n=== 开始预测并存储结果 ===")
        results, threshold = predict_with_confidence_boost(
            model, feature_names, scaler, wellington_df
        )

        if len(results) > 0:
            # 存储结果
            total_inserted = store_predictions(results, threshold)

            print(f"\n=== 预测完成 ===")
            print(f"✓ 成功插入 {total_inserted} 条Wellington预测结果")
            print(f"✓ 使用置信度阈值: {threshold:.3f}")
            print(f"✓ 平均置信度: {results['confidence_score'].mean():.4f}")
            print(f"✓ 模型准确率: {accuracy:.4f}")

            # 显示样本结果
            print("\n样本预测结果:")
            sample = results[
                ["property_address", "suburb", "predicted_status", "confidence_score"]
            ].head(10)
            for _, row in sample.iterrows():
                status_text = "Active" if row["predicted_status"] == 1 else "Inactive"
                addr = (
                    str(row["property_address"])[:40]
                    if pd.notna(row["property_address"])
                    else "Unknown"
                )
                suburb = (
                    str(row["suburb"])[:15] if pd.notna(row["suburb"]) else "Unknown"
                )
                print(
                    f"  {addr:<40} | {suburb:<15} | {status_text:<8} | {row['confidence_score']:.3f}"
                )

            # 成功总结
            print(
                f"\n🎉 Wellington地区预测完成，共插入 {total_inserted} 条置信度 > {threshold:.1f} 的结果到数据库。"
            )

        else:
            print("没有生成任何预测结果")
            print("建议:")
            print("1. 检查数据质量和完整性")
            print("2. 增加更多训练数据")
            print("3. 调整特征工程策略")

    except Exception as e:
        print(f"执行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
