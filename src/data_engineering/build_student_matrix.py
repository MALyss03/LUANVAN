import pandas as pd
import numpy as np
from pathlib import Path

def convert_ctu_grade(score: float) -> tuple[str, float]:
    """Quy đổi điểm theo Điều 22.2 - Quyết định 3266/QĐ-ĐHCT."""
    if pd.isna(score):
        return np.nan, np.nan
    if score >= 9.0: return 'A', 4.0
    elif score >= 8.0: return 'B+', 3.5
    elif score >= 7.0: return 'B', 3.0
    elif score >= 6.5: return 'C+', 2.5
    elif score >= 5.5: return 'C', 2.0
    elif score >= 5.0: return 'D+', 1.5
    elif score >= 4.0: return 'D', 1.0
    else: return 'F', 0.0

def check_ctu_academic_warning(
    dtb_hk: float, 
    dtb_tl: float, 
    academic_year: int, 
    failed_credits_hk: int, 
    registered_credits_hk: int,
    total_accumulated_failed_credits: int,
    is_first_hk: bool = False
) -> bool:
    """Kiểm tra cảnh báo học tập theo Điều 18.2 - Quyết định 3266/QĐ-ĐHCT."""
    # 1. Cảnh báo theo ĐTBCHK
    hk_threshold = 0.80 if is_first_hk else 1.00
    if dtb_hk < hk_threshold:
        return True

    # 2. Cảnh báo theo tín chỉ không đạt hoặc nợ đọng
    if registered_credits_hk > 0 and (failed_credits_hk / registered_credits_hk) > 0.5:
        return True
    if total_accumulated_failed_credits > 24:
        return True

    # 3. Cảnh báo theo ĐTBCTL từng năm học (Áp dụng từ Khóa 47)
    cpa_thresholds = {1: 1.20, 2: 1.40, 3: 1.60}
    cpa_limit = cpa_thresholds.get(academic_year, 1.80)
    if dtb_tl < cpa_limit:
        return True

    return False

def calculate_ctu_averages(student_df: pd.DataFrame):
    """
    Tính ĐTBCHK và ĐTBCTL đúng chuẩn Điều 24 - Quyết định 3266/QĐ-ĐHCT:
    - ĐTBCHK: Tính tất cả môn trong kỳ (gồm F và học phần điều kiện, TRỪ GDTC).
    - ĐTBCTL: Chỉ tính các môn ĐÃ TÍCH LŨY (không tính F và các học phần điều kiện).
    """
    CONDITION_COURSES = {'GDQP', 'GDTC', 'NN_CAN_BAN', 'TIN_HOC_CAN_BAN'}

    # 1. Tính ĐTBCHK (Loại trừ GDTC)
    hk_df = student_df[student_df['loai_hp'] != 'GDTC']
    total_hk_credits = hk_df['so_tin_chi'].sum()
    dtb_hk = (hk_df['diem_4'] * hk_df['so_tin_chi']).sum() / total_hk_credits if total_hk_credits > 0 else 0.0

    # 2. Tính ĐTBCTL (Loại trừ môn bị F và TẤT CẢ học phần điều kiện)
    tl_df = student_df[
        (student_df['diem_4'] >= 1.0) & 
        (~student_df['loai_hp'].isin(CONDITION_COURSES))
    ]
    total_tl_credits = tl_df['so_tin_chi'].sum()
    dtb_tl = (tl_df['diem_4'] * tl_df['so_tin_chi']).sum() / total_tl_credits if total_tl_credits > 0 else 0.0

    return dtb_hk, dtb_tl