"""
===============================================================================
FILE: src/data_engineering/run_data_pipeline.py
MỤC ĐÍCH: Chạy Pipeline Trích xuất Dữ liệu Tổng hợp ra File CSV/Excel 1 dòng/SV
===============================================================================
"""

import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engineering.unified_feature_engineering import FullFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


def run_full_pipeline(raw_csv_path: str = None):
    logging.info("🚀 BẮT ĐẦU CHẠY PIPELINE TRÍCH XUẤT MA TRẬN SINH VIÊN 1 DÒNG")
    
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    os.makedirs(processed_dir, exist_ok=True)

    input_path = raw_csv_path if raw_csv_path else str(raw_dir / "DIEMTHAYCANG_FINAL.csv")
    output_path = str(processed_dir / "student_full_matrix_wide.csv")

    engineer = FullFeatureEngineer(
        raw_filepath=input_path, 
        output_filepath=output_path,
        max_semesters=12  # Cấu hình tối đa số học kỳ mở rộng dạng cột
    )
    engineer.run()

    logging.info(f"✅ Hoàn tất! File dữ liệu kết quả nằm tại: {output_path}")


if __name__ == '__main__':
    run_full_pipeline()