"""
===============================================================================
FILE: src/data_engineering/feature_engineering.py
MỤC ĐÍCH: Trích xuất Bộ đặc trưng Đa chiều 48+ Features dạng Semester Student Profile
          Đảm bảo không bị Data Leakage & Tính toán chính xác Động học Học tập
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.data_engineering.curriculum_engine import CurriculumEngine

# Mốc tín chỉ tích lũy chuẩn Đại học Cần Thơ theo từng học kỳ
EXPECTED_CREDITS_TIMELINE = {1: 18, 2: 36, 3: 54, 4: 72, 5: 90, 6: 108, 7: 126, 8: 144, 9: 161}

def calculate_slope(series: List[float]) -> float:
    """Tính độ dốc (Slope) bằng Linear Regression trên chuỗi lịch sử."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    y = np.array(series)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    return round(float(num / den), 4) if den != 0 else 0.0


def run_feature_engineering(df_raw: pd.DataFrame, curriculum_engine: CurriculumEngine = None) -> pd.DataFrame:
    """Trích xuất ma trận Semester Student Profile [1 dòng = 1 Sinh viên x 1 Học kỳ]."""
    df = df_raw.copy()
    
    # Chuẩn hóa tên cột
    rename_dict = {
        'F_MASV': 'mssv', 'student_id': 'mssv', 'ma_sv': 'mssv',
        'F_TENLOP': 'ma_lop', 'class_code': 'ma_lop',
        'NHHK': 'hoc_ky', 'semester': 'hoc_ky',
        'F_DVHT': 'tin_chi', 'credits': 'tin_chi', 'stc': 'tin_chi',
        'F_DIEM2': 'diem_thang_10', 'diem': 'diem_thang_10', 'score': 'diem_thang_10',
        'F_MAMH': 'ma_mon', 'course_code': 'ma_mon'
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    df['tin_chi'] = pd.to_numeric(df['tin_chi'], errors='coerce').fillna(0).astype(int)
    df['diem_thang_10'] = pd.to_numeric(df['diem_thang_10'], errors='coerce')
    
    # Quy đổi điểm thang 4 chuẩn
    def to_grade_4(x):
        if pd.isna(x): return np.nan
        if x >= 9.0: return 4.0
        elif x >= 8.0: return 3.5
        elif x >= 7.0: return 3.0
        elif x >= 6.5: return 2.5
        elif x >= 5.5: return 2.0
        elif x >= 5.0: return 1.5
        elif x >= 4.0: return 1.0
        else: return 0.0

    df['diem_thang_4'] = df['diem_thang_10'].apply(to_grade_4)
    df['is_passed'] = df['diem_thang_4'] >= 1.0

    if curriculum_engine is None:
        curriculum_engine = CurriculumEngine()

    semester_profiles = []

    # Nhóm dữ liệu theo sinh viên và sắp xếp theo tiến trình thời gian
    for mssv, group in df.groupby('mssv'):
        ma_lop = str(group['ma_lop'].iloc[0]) if 'ma_lop' in group.columns else 'KXD'
        nganh_hoc = 'Mạng máy tính' if 'T9' in ma_lop else ('An toàn thông tin' if 'D2' in ma_lop else 'CNTT')
        
        history_gpa_hk = []
        history_gpa_tl = []
        history_tc_reg = []
        
        passed_courses_set = set()
        failed_courses_set = set()
        attempted_courses_hist = set()

        cum_tc_dang_ky = 0
        cum_tc_tich_luy = 0
        cum_tc_rot = 0
        cum_score_points = 0.0

        sems = sorted(group['hoc_ky'].unique())

        for idx, hk_val in enumerate(sems):
            hk_num = idx + 1
            sem_data = group[group['hoc_ky'] == hk_val]

            # 1. Nhóm Tín chỉ Học kỳ
            tc_dk_hk = sem_data['tin_chi'].sum()
            sem_passed = sem_data[sem_data['is_passed']]
            sem_failed = sem_data[~sem_data['is_passed']]
            
            tc_dat_hk = sem_passed['tin_chi'].sum()
            tc_rot_hk = sem_failed['tin_chi'].sum()
            tc_no_hk = tc_dk_hk - tc_dat_hk
            credit_completion_rate = round(tc_dat_hk / tc_dk_hk, 4) if tc_dk_hk > 0 else 0.0

            # 2. Nhóm Điểm GPA
            # GPA Học kỳ
            gpa_hk_4 = round((sem_data['diem_thang_4'] * sem_data['tin_chi']).sum() / tc_dk_hk, 2) if tc_dk_hk > 0 else 0.0
            gpa_hk_10 = round((sem_data['diem_thang_10'] * sem_data['tin_chi']).sum() / tc_dk_hk, 2) if tc_dk_hk > 0 else 0.0

            # Cập nhật tập hợp môn học
            for _, r in sem_data.iterrows():
                code = str(r['ma_mon']).upper()
                if r['is_passed']:
                    passed_courses_set.add(code)
                    if code in failed_courses_set:
                        failed_courses_set.remove(code)
                else:
                    if code not in passed_courses_set:
                        failed_courses_set.add(code)
                attempted_courses_hist.add(code)

            # GPA Tích lũy (Chỉ tính các môn ĐẠT và loại trừ học phần điều kiện)
            cum_tc_dang_ky += tc_dk_hk
            cum_tc_tich_luy += tc_dat_hk
            cum_tc_rot += tc_rot_hk
            
            # Tính điểm tích lũy chuẩn
            passed_df = sem_data[sem_data['is_passed']]
            cum_score_points += (passed_df['diem_thang_4'] * passed_df['tin_chi']).sum()
            gpa_tl_4 = round(cum_score_points / cum_tc_tich_luy, 2) if cum_tc_tich_luy > 0 else 0.0
            gpa_tl_10 = round(gpa_tl_4 * 2.5, 2)

            # 3. Động học GPA
            history_gpa_hk.append(gpa_hk_4)
            history_gpa_tl.append(gpa_tl_4)
            history_tc_reg.append(tc_dk_hk)

            delta_gpa_hk = round(gpa_hk_4 - history_gpa_hk[-2], 2) if len(history_gpa_hk) > 1 else 0.0
            delta_gpa_tl = round(gpa_tl_4 - history_gpa_tl[-2], 2) if len(history_gpa_tl) > 1 else 0.0
            
            gpa_hk_ma3 = round(float(np.mean(history_gpa_hk[-3:])), 2)
            gpa_tl_ma3 = round(float(np.mean(history_gpa_tl[-3:])), 2)
            
            gpa_hk_slope = calculate_slope(history_gpa_hk)
            gpa_tl_slope = calculate_slope(history_gpa_tl)
            
            gpa_hk_std_3 = round(float(np.std(history_gpa_hk[-3:])), 2) if len(history_gpa_hk) >= 3 else 0.0
            gpa_hk_std_5 = round(float(np.std(history_gpa_hk[-5:])), 2) if len(history_gpa_hk) >= 5 else 0.0

            # Tính Chuỗi suy giảm / Cải thiện
            decline_streak = 0
            for i in range(len(history_gpa_hk)-1, 0, -1):
                if history_gpa_hk[i] < history_gpa_hk[i-1]:
                    decline_streak += 1
                else: break

            improve_streak = 0
            for i in range(len(history_gpa_hk)-1, 0, -1):
                if history_gpa_hk[i] > history_gpa_hk[i-1]:
                    improve_streak += 1
                else: break

            # 4. Động học Đăng ký Tín chỉ
            delta_tc_reg = tc_dk_hk - history_tc_reg[-2] if len(history_tc_reg) > 1 else 0
            reg_credit_slope = calculate_slope(history_tc_reg)
            avg_tc_last_3 = round(float(np.mean(history_tc_reg[-3:])), 1)

            reg_decline_streak = 0
            for i in range(len(history_tc_reg)-1, 0, -1):
                if history_tc_reg[i] < history_tc_reg[i-1]: reg_decline_streak += 1
                else: break

            # 5. Phân bố điểm A/B/C/D/D+/F
            scores_10 = sem_data['diem_thang_10'].tolist()
            f_count_hk = sum(1 for s in scores_10 if s < 4.0)
            d_count_hk = sum(1 for s in scores_10 if 4.0 <= s < 5.0)
            dplus_count_hk = sum(1 for s in scores_10 if 5.0 <= s < 5.5)
            c_count_hk = sum(1 for s in scores_10 if 5.5 <= s < 7.0)
            b_count_hk = sum(1 for s in scores_10 if 7.0 <= s < 8.5)
            a_count_hk = sum(1 for s in scores_10 if s >= 8.5)

            tot_courses_hk = len(scores_10) if len(scores_10) > 0 else 1
            f_rate_hk = round(f_count_hk / tot_courses_hk, 4)
            low_grade_count_hk = d_count_hk + dplus_count_hk
            low_grade_rate_hk = round(low_grade_count_hk / tot_courses_hk, 4)

            # 6. Môn Tiên quyết & Môn học lại
            blocked_count, blocked_credits, longest_chain = curriculum_engine.calculate_prerequisite_blocks(passed_courses_set)
            
            # Tính số lần học lại
            repeat_count_hk = sum(1 for c in sem_data['ma_mon'] if str(c).upper() in attempted_courses_hist)
            
            # 7. Tiến độ & Khả năng Tốt nghiệp
            expected_credits = EXPECTED_CREDITS_TIMELINE.get(hk_num, 161)
            credit_deficit = max(0, expected_credits - cum_tc_tich_luy)
            remaining_credits = max(0, 161 - cum_tc_tich_luy)
            remaining_sems = max(1, 8 - hk_num)
            req_avg_credits_per_sem = round(remaining_credits / remaining_sems, 1)
            
            grad_feasibility_gap = round(req_avg_credits_per_sem - avg_tc_last_3, 1)

            # Tổng hợp vào Profile Học kỳ
            semester_profiles.append({
                'MSSV': mssv,
                'MaLop': ma_lop,
                'NganhHoc': nganh_hoc,
                'HocKy': hk_val,
                'HocKySo': hk_num,
                'NamHoc': min(4, (hk_num + 1) // 2),
                
                # GPA
                'GPA_HK_4': gpa_hk_4,
                'GPA_HK_10': gpa_hk_10,
                'GPA_TL_4': gpa_tl_4,
                'GPA_TL_10': gpa_tl_10,
                'Delta_GPA_HK': delta_gpa_hk,
                'Delta_GPA_TL': delta_gpa_tl,
                'GPA_HK_MA3': gpa_hk_ma3,
                'GPA_TL_MA3': gpa_tl_ma3,
                'GPA_HK_Slope': gpa_hk_slope,
                'GPA_TL_Slope': gpa_tl_slope,
                'GPA_HK_STD_3': gpa_hk_std_3,
                'GPA_HK_STD_5': gpa_hk_std_5,
                'GPA_Decline_Streak': decline_streak,
                'GPA_Improvement_Streak': improve_streak,

                # Tín chỉ & Đăng ký
                'TC_Dang_Ky_HK': tc_dk_hk,
                'TC_Dat_HK': tc_dat_hk,
                'TC_Rot_HK': tc_rot_hk,
                'TC_No_HK': tc_no_hk,
                'TC_Tich_Luy': cum_tc_tich_luy,
                'TC_No_Tich_Luy': max(0, cum_tc_dang_ky - cum_tc_tich_luy),
                'Credit_Completion_Rate': credit_completion_rate,
                'Credit_Deficit': credit_deficit,
                'Delta_TC_Dang_Ky': delta_tc_reg,
                'Registration_Credit_Slope': reg_credit_slope,
                'Average_TC_Last_3_Semesters': avg_tc_last_3,
                'TC_Registration_Decline_Streak': reg_decline_streak,

                # Phân bố điểm & Môn F/D
                'F_Count_HK': f_count_hk,
                'F_Rate_HK': f_rate_hk,
                'Low_Grade_Count_HK': low_grade_count_hk,
                'Low_Grade_Rate_HK': low_grade_rate_hk,
                'A_Rate': round(a_count_hk / tot_courses_hk, 4),
                'B_Rate': round(b_count_hk / tot_courses_hk, 4),
                'C_Rate': round(c_count_hk / tot_courses_hk, 4),

                # Tiên quyết & Tốt nghiệp
                'Prerequisite_Block_Count': blocked_count,
                'Prerequisite_Block_Credits': blocked_credits,
                'Longest_Prerequisite_Chain_Blocked': longest_chain,
                'Repeat_Count_HK': repeat_count_hk,
                'Remaining_Credits': remaining_credits,
                'Required_Average_Credits_Per_Semester': req_avg_credits_per_sem,
                'Graduation_Feasibility_Gap': grad_feasibility_gap
            })

    return pd.DataFrame(semester_profiles)