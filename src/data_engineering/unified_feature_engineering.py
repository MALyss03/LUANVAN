"""
===============================================================================
FILE: src/data_engineering/unified_feature_engineer.py
MỤC ĐÍCH: Trích xuất Ma trận Hồ sơ Sinh viên 1 Dòng (Wide Format)
          Bao gồm: Thông tin SV + Kết quả theo từng Học kỳ + Chi tiết môn học
===============================================================================
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger('UnifiedFeatureEngineer')


def convert_grade_10_to_4(score: float) -> float:
    """Quy đổi điểm thang 10 sang thang 4 chuẩn CTU."""
    if pd.isna(score):
        return np.nan
    try:
        val = float(score)
    except (ValueError, TypeError):
        return np.nan

    if val >= 9.0: return 4.0
    elif val >= 8.0: return 3.5
    elif val >= 7.0: return 3.0
    elif val >= 6.5: return 2.5
    elif val >= 5.5: return 2.0
    elif val >= 5.0: return 1.5
    elif val >= 4.0: return 1.0
    else: return 0.0


def classify_academic_rank(gpa_4: float, tc_registered: int) -> str:
    """Xếp loại Học kỳ dựa trên ĐTBCHK (Thang 4)."""
    if tc_registered == 0 or pd.isna(gpa_4):
        return "Chưa rõ"
    if gpa_4 >= 3.6: return "Xuất sắc"
    elif gpa_4 >= 3.2: return "Giỏi"
    elif gpa_4 >= 2.5: return "Khá"
    elif gpa_4 >= 2.0: return "Trung bình"
    elif gpa_4 >= 1.0: return "Yếu"
    else: return "Kém"


def classify_major_from_class_code(ma_lop: str) -> str:
    """Xác định ngành học từ mã lớp."""
    if pd.isna(ma_lop) or str(ma_lop).strip().upper() in ['KXD', 'NAN', 'NONE', '']:
        return 'Chưa xác định'
    
    ma_lop_clean = str(ma_lop).strip().upper()
    if 'T9' in ma_lop_clean:
        return 'Mạng máy tính và Truyền thông dữ liệu'
    elif 'D2' in ma_lop_clean:
        return 'An toàn thông tin'
    return 'Công nghệ thông tin'


def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột linh hoạt từ file thô."""
    rename_map = {
        'F_MASV': 'mssv', 'student_id': 'mssv', 'ma_sv': 'mssv',
        'F_TENLOP': 'ma_lop', 'class_code': 'ma_lop',
        'F_MAMH': 'ma_mon', 'course_code': 'ma_mon',
        'F_DVHT': 'tin_chi', 'credits': 'tin_chi', 'stc': 'tin_chi',
        'F_DIEM2': 'diem_thang_10', 'diem': 'diem_thang_10', 'score': 'diem_thang_10',
        'NHHK': 'hoc_ky', 'semester': 'hoc_ky'
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


class FullFeatureEngineer:
    def __init__(self, raw_filepath: str = None, output_filepath: str = None, max_semesters: int = 12):
        project_root = Path(__file__).resolve().parents[2]
        self.raw_filepath = (
            Path(raw_filepath) if raw_filepath 
            else project_root / 'data' / 'raw' / 'DIEMTHAYCANG_FINAL.csv'
        )
        self.output_filepath = (
            Path(output_filepath) if output_filepath 
            else project_root / 'data' / 'processed' / 'full_student_features_wide.csv'
        )
        self.max_semesters = max_semesters

    def generate_semester_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất 9 trường dữ liệu theo TỪNG HỌC KỲ thành các CỘT NGANG cho mỗi SV:
        - Số TC Đăng ký
        - Số TC Tích lũy (Đạt)
        - Số TC Nợ đọng
        - ĐTBCHK (Thang 10) & (Thang 4)
        - CPA Tích lũy (Thang 10) & (Thang 4)
        - Tỷ lệ rớt môn (%)
        - Xếp loại HK
        """
        logger.info('[STEP 2] Đang xoay dữ liệu theo từng Học kỳ thành các cột ngang...')
        
        student_sem_records = []
        
        # Sắp xếp theo học kỳ
        df_sorted = df.sort_values(['mssv', 'hoc_ky'])

        for mssv, group in df_sorted.groupby('mssv'):
            semesters = sorted(group['hoc_ky'].unique())
            sv_dict = {'mssv': mssv}

            cum_tc_dang_ky = 0
            cum_tc_tich_luy = 0
            cum_score_points_10 = 0.0
            cum_score_points_4 = 0.0

            # Duyệt qua các học kỳ
            for idx, hk_val in enumerate(semesters[:self.max_semesters]):
                hk_num = idx + 1
                prefix = f"HK{hk_num}_"

                sem_data = group[group['hoc_ky'] == hk_val]

                # 1. Tín chỉ học kỳ
                tc_dk_hk = sem_data['tin_chi'].sum()
                sem_passed = sem_data[sem_data['is_passed']]
                sem_failed = sem_data[~sem_data['is_passed']]

                tc_dat_hk = sem_passed['tin_chi'].sum()
                tc_rot_hk = sem_failed['tin_chi'].sum()
                tc_no_hk = max(0, tc_dk_hk - tc_dat_hk)

                # 2. ĐTBCHK (Loại trừ môn không tính điểm nếu có)
                dtb_hk_10 = round((sem_data['diem_thang_10'] * sem_data['tin_chi']).sum() / tc_dk_hk, 2) if tc_dk_hk > 0 else 0.0
                dtb_hk_4 = round((sem_data['diem_thang_4'] * sem_data['tin_chi']).sum() / tc_dk_hk, 2) if tc_dk_hk > 0 else 0.0

                # 3. CPA Tích lũy (Chỉ tính các môn ĐẠT)
                cum_tc_dang_ky += tc_dk_hk
                cum_tc_tich_luy += tc_dat_hk
                
                cum_score_points_10 += (sem_passed['diem_thang_10'] * sem_passed['tin_chi']).sum()
                cum_score_points_4 += (sem_passed['diem_thang_4'] * sem_passed['tin_chi']).sum()

                cpa_tl_10 = round(cum_score_points_10 / cum_tc_tich_luy, 2) if cum_tc_tich_luy > 0 else 0.0
                cpa_tl_4 = round(cum_score_points_4 / cum_tc_tich_luy, 2) if cum_tc_tich_luy > 0 else 0.0

                # 4. Tỷ lệ rớt môn HK (%)
                total_courses = len(sem_data)
                failed_courses = len(sem_failed)
                fail_rate_hk = round((failed_courses / total_courses) * 100, 2) if total_courses > 0 else 0.0

                # 5. Xếp loại học kỳ
                xep_loai_hk = classify_academic_rank(dtb_hk_4, tc_dk_hk)

                # Gán các cột cho học kỳ hiện tại
                sv_dict[f"{prefix}So_TC_Dang_Ky"] = tc_dk_hk
                sv_dict[f"{prefix}So_TC_Tich_Luy_Dat"] = tc_dat_hk
                sv_dict[f"{prefix}So_TC_No_Dong"] = tc_no_hk
                sv_dict[f"{prefix}DTBCHK_Thang_10"] = dtb_hk_10
                sv_dict[f"{prefix}DTBCHK_Thang_4"] = dtb_hk_4
                sv_dict[f"{prefix}CPA_Tich_Luy_Thang_10"] = cpa_tl_10
                sv_dict[f"{prefix}CPA_Tich_Luy_Thang_4"] = cpa_tl_4
                sv_dict[f"{prefix}Ty_Le_Rot_Mon"] = f"{fail_rate_hk}%"
                sv_dict[f"{prefix}Xep_Loai_HK"] = xep_loai_hk

            student_sem_records.append(sv_dict)

        return pd.DataFrame(student_sem_records)

    def generate_course_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất thông tin MÔN HỌC thành các CỘT NGANG cho từng SV:
        - Điểm môn học (DIEM_<MA_MON>)
        - Số lần học (SOLAN_HOC_<MA_MON>)
        - Số lần hoàn thành (SOLAN_HOAN_THANH_<MA_MON>)
        - Tỉ lệ hoàn thành (TILE_HOAN_THANH_<MA_MON>)
        """
        logger.info('[STEP 3] Đang tính điểm, số lần học, số lần hoàn thành và tỉ lệ hoàn thành từng môn...')

        df_valid = df.dropna(subset=['mssv', 'ma_mon']).copy()
        df_valid['ma_mon'] = df_valid['ma_mon'].astype(str).str.strip().str.upper()

        course_stats = df_valid.groupby(['mssv', 'ma_mon']).agg(
            diem_max=('diem_thang_10', 'max'),
            so_lan_hoc=('diem_thang_10', 'count'),
            so_lan_hoan_thanh=('is_passed', lambda x: int(x.sum()))
        ).reset_index()

        course_stats['ti_le_hoan_thanh'] = np.where(
            course_stats['so_lan_hoc'] > 0,
            (course_stats['so_lan_hoan_thanh'] / course_stats['so_lan_hoc']).round(4),
            0.0
        )

        piv_score = course_stats.pivot(index='mssv', columns='ma_mon', values='diem_max')
        piv_score.columns = [f'DIEM_{col}' for col in piv_score.columns]

        piv_attempts = course_stats.pivot(index='mssv', columns='ma_mon', values='so_lan_hoc').fillna(0).astype(int)
        piv_attempts.columns = [f'SOLAN_HOC_{col}' for col in piv_attempts.columns]

        piv_passed = course_stats.pivot(index='mssv', columns='ma_mon', values='so_lan_hoan_thanh').fillna(0).astype(int)
        piv_passed.columns = [f'SOLAN_HOAN_THANH_{col}' for col in piv_passed.columns]

        piv_rate = course_stats.pivot(index='mssv', columns='ma_mon', values='ti_le_hoan_thanh').fillna(0.0)
        piv_rate.columns = [f'TILE_HOAN_THANH_{col}' for col in piv_rate.columns]

        return pd.concat([piv_score, piv_attempts, piv_passed, piv_rate], axis=1).reset_index()

    def process_data(self) -> pd.DataFrame:
        logger.info(f'[STEP 1] Đọc dữ liệu từ: {self.raw_filepath}')
        if not self.raw_filepath.exists():
            raise FileNotFoundError(f'Không tìm thấy file dữ liệu tại: {self.raw_filepath}')

        df = pd.read_csv(self.raw_filepath, low_memory=False)
        df = auto_map_columns(df)

        df['mssv'] = df['mssv'].astype(str).str.strip()
        df['tin_chi'] = pd.to_numeric(df['tin_chi'], errors='coerce').fillna(0).astype(int)
        df['diem_thang_10'] = pd.to_numeric(df['diem_thang_10'], errors='coerce')
        df['diem_thang_4'] = df['diem_thang_10'].apply(convert_grade_10_to_4)
        df['is_passed'] = df['diem_thang_4'] >= 1.0

        if 'ma_lop' in df.columns:
            df['ma_lop'] = df['ma_lop'].fillna('KXD').astype(str).str.strip().str.upper()
            df['nganh_hoc'] = df['ma_lop'].apply(classify_major_from_class_code)
        else:
            df['ma_lop'] = 'KXD'
            df['nganh_hoc'] = 'Chưa xác định'

        # 1. Trích xuất thông tin định danh sinh viên
        df_sv_base = df.groupby('mssv').agg(
            MaLop=('ma_lop', 'first'),
            NganhHoc=('nganh_hoc', 'first')
        ).reset_index().rename(columns={'mssv': 'MSSV'})

        # 2. Tạo ma trận theo Học kỳ
        df_sem_matrix = self.generate_semester_matrix(df)

        # 3. Tạo ma trận theo Môn học
        df_course_matrix = self.generate_course_matrix(df)

        # 4. Gộp toàn bộ vào 1 Bảng chung
        logger.info('[STEP 4] Ghép nối thông tin SV + Bảng học kỳ + Bảng môn học...')
        df_final = pd.merge(df_sv_base, df_sem_matrix, left_on='MSSV', right_on='mssv', how='left')
        df_final.drop(columns=['mssv'], inplace=True, errors='ignore')

        df_final = pd.merge(df_final, df_course_matrix, left_on='MSSV', right_on='mssv', how='left')
        df_final.drop(columns=['mssv'], inplace=True, errors='ignore')

        return df_final

    def run(self):
        df_final = self.process_data()
        os.makedirs(self.output_filepath.parent, exist_ok=True)
        df_final.to_csv(self.output_filepath, index=False, encoding='utf-8-sig')
        logger.info(f'🎉 [HOÀN THÀNH] Đã xuất file thành công: {self.output_filepath}')


if __name__ == '__main__':
    engineer = FullFeatureEngineer()
    engineer.run()