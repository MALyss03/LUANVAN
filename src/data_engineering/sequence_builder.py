"""
===============================================================================
FILE: src/data_engineering/sequence_builder.py
MỤC ĐÍCH: Khởi tạo Tensor Chuỗi [Students x Semesters x Dynamic_Features]
          Phục vụ huấn luyện Transformer / RNN RecSys
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

FEATURE_EXCLUDE_COLS = [
    'MSSV', 'MaLop', 'NganhHoc', 'HocKy', 'Trang_Thai_Hoc_Vu', 
    'Target_Label', 'Risk_Level', 'Rule_Evidence'
]

def build_student_sequence_tensor(
    df_labeled: pd.DataFrame, max_semesters: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Tạo Tensor chuỗi với số chiều đặc trưng (feature_dim) tự động cập nhật."""
    feature_cols = [c for c in df_labeled.columns if c not in FEATURE_EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df_labeled[c])]
    
    students = df_labeled['MSSV'].unique()
    num_students = len(students)
    feature_dim = len(feature_cols)

    X_seq = np.zeros((num_students, max_semesters, feature_dim), dtype=np.float32)
    y_target = np.zeros(num_students, dtype=np.int64)
    y_risk_score = np.zeros(num_students, dtype=np.float32)

    student_map = {mssv: i for i, mssv in enumerate(students)}

    for mssv, group in df_labeled.groupby('MSSV'):
        s_idx = student_map[mssv]
        sorted_group = group.sort_values('HocKySo')

        for _, row in sorted_group.iterrows():
            sem_idx = int(row['HocKySo']) - 1
            if sem_idx < max_semesters:
                vec = row[feature_cols].values.astype(np.float32)
                # Thay thế NaN bằng 0.0
                vec = np.nan_to_num(vec, nan=0.0)
                X_seq[s_idx, sem_idx, :] = vec

        # Lấy nhãn của học kỳ gần nhất làm Target
        last_row = sorted_group.iloc[-1]
        y_target[s_idx] = int(last_row['Target_Label'])
        y_risk_score[s_idx] = float(last_row['Risk_Score'])

    return X_seq, y_target, y_risk_score, feature_cols