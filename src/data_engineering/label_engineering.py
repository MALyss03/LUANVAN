"""
===============================================================================
FILE: src/data_engineering/label_engineering.py
MỤC ĐÍCH: Gán nhãn 4 Lớp Học vụ & Xuất Bằng chứng Quy tắc (Rule Evidence)
===============================================================================
"""

import pandas as pd
from src.rules.academic_rules import RuleEngine

def run_label_engineering(df_semester_features: pd.DataFrame) -> pd.DataFrame:
    """Gán nhãn phân loại 4 lớp kèm Rule Evidence giải thích lý do."""
    df = df_semester_features.sort_values(['MSSV', 'HocKySo']).copy()
    rule_engine = RuleEngine()
    
    labeled_records = []

    for mssv, group in df.groupby('MSSV'):
        consecutive_warnings = 0

        for _, row in group.iterrows():
            row_dict = row.to_dict()
            
            # Tính safety margin so với ngưỡng quy chế
            nam_hoc = row_dict['NamHoc']
            min_gpa_tl = {1: 1.20, 2: 1.40, 3: 1.60, 4: 1.80}.get(nam_hoc, 1.80)
            row_dict['GPA_Safety_Margin'] = round(row_dict['GPA_TL_4'] - min_gpa_tl, 2)

            # Đánh giá bằng chứng vi phạm quy tắc
            evidences = rule_engine.evaluate_student_semester(row_dict)
            evidence_descriptions = [e['description'] for e in evidences]
            
            official_violations = [e for e in evidences if e['source'] == 'OFFICIAL_CTU_RULE']
            
            if len(official_violations) > 0:
                consecutive_warnings += 1
            else:
                consecutive_warnings = 0

            row_dict['Consecutive_Warnings'] = consecutive_warnings
            row_dict['Rule_Evidence'] = evidence_descriptions

            # Logic Phân loại 4 Nhãn Trạng thái
            if consecutive_warnings >= 2:
                target_label = 3
                status_name = "Nguy cơ buộc thôi học"
            elif len(official_violations) > 0 or consecutive_warnings == 1:
                target_label = 2
                status_name = "Nguy cơ bị cảnh báo học vụ"
            elif row_dict['Credit_Deficit'] > 12 or row_dict['Prerequisite_Block_Count'] >= 2:
                target_label = 1
                status_name = "Trễ tiến độ"
            else:
                target_label = 0
                status_name = "Đúng tiến độ"

            row_dict['Target_Label'] = target_label
            row_dict['Trang_Thai_Hoc_Vu'] = status_name
            
            # Tính Risk Score phân rã (0 - 100)
            risk_score = min(100, int(
                (max(0, 2.0 - row_dict['GPA_HK_4']) * 20) +
                (row_dict['F_Rate_HK'] * 30) +
                (min(1.0, row_dict['Credit_Deficit'] / 30.0) * 25) +
                (min(1.0, row_dict['Prerequisite_Block_Count'] / 5.0) * 25)
            ))
            
            row_dict['Risk_Score'] = risk_score
            row_dict['Risk_Level'] = "HIGH" if risk_score >= 60 else ("MEDIUM" if risk_score >= 35 else "LOW")

            labeled_records.append(row_dict)

    return pd.DataFrame(labeled_records)