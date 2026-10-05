"""
===============================================================================
FILE: train_model.py
MỤC ĐÍCH: Huấn luyện mô hình Machine Learning dự báo nguy cơ Cảnh báo học tập
===============================================================================
"""

import os
import sys
import logging
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

PROJECT_ROOT = Path(__file__).resolve().parent


def load_processed_data(file_path: str = None) -> pd.DataFrame:
    """Tải dữ liệu đã gán nhãn và trích xuất đặc trưng."""
    if file_path is None:
        file_path = PROJECT_ROOT / "data" / "processed" / "processed_student_features.csv"

    if not os.path.exists(file_path):
        logging.error(f"Không tìm thấy file dữ liệu tại {file_path}. Hãy chạy run_data_pipeline.py trước!")
        sys.exit(1)

    logging.info(f"Đang tải dữ liệu huấn luyện từ: {file_path}")
    return pd.read_csv(file_path)


def prepare_features_and_target(df: pd.DataFrame):
    """Chuẩn bị tập đặc trưng (X) và nhãn mục tiêu (y)."""
    feature_cols = [
        'gpa_gap_hk_vs_tl',
        'dtbchk_trend',
        'dtbctl_safety_margin',
        'dtbchk_safety_margin',
        'credit_load_ratio',
        'credit_fail_ratio_hk',
        'no_dong_risk_ratio',
        'credit_accumulation_speed'
    ]

    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].copy()

    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    if 'is_canh_bao_hoc_vu' in df.columns:
        y = df['is_canh_bao_hoc_vu'].astype(int)
    elif 'is_canh_bao_hk' in df.columns:
        y = df['is_canh_bao_hk'].astype(int)
    else:
        y = (df['dtbctl_safety_margin'] < 0).astype(int)

    logging.info(f"-> Sử dụng {len(available_features)} đặc trưng làm đầu vào mô hình.")
    return X, y, available_features


def train_academic_risk_model(X: pd.DataFrame, y: pd.Series):
    """Huấn luyện mô hình Random Forest."""
    if len(X) < 10 or y.nunique() < 2:
        logging.warning("Dữ liệu nhỏ hoặc đơn lớp, huấn luyện mô hình mẫu trên toàn bộ tập...")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH DỰ BÁO ===")
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    return model


def save_model(model, feature_names: list, model_path: str = None):
    """Lưu mô hình đã huấn luyện ra file pkl."""
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "academic_warning_rf_model.pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_artifacts = {
        'model': model,
        'feature_names': feature_names
    }

    joblib.dump(model_artifacts, model_path)
    logging.info(f"-> Mô hình đã lưu tại: {os.path.abspath(model_path)}")


def run_training_pipeline():
    logging.info("==================================================")
    logging.info("   BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH STUDENTPROGRESSAI   ")
    logging.info("==================================================")

    df = load_processed_data()
    X, y, feature_names = prepare_features_and_target(df)
    model = train_academic_risk_model(X, y)
    save_model(model, feature_names)

    logging.info("Hoàn tất quy trình huấn luyện mô hình!")


if __name__ == '__main__':
    run_training_pipeline()