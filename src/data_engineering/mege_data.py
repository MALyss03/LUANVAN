import pandas as pd
import numpy as np
from pathlib import Path

def merge_fl_courses_to_wide_matrix(
    raw_csv_path: str = "data/raw/DIEMTHAYCANG_FINAL.csv",
    wide_csv_path: str = "data/processed/student_full_matrix_wide.csv",
    output_csv_path: str = "data/processed/student_full_matrix_wide.csv"
):
    # 1. Tải dữ liệu
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    df_wide = pd.read_csv(wide_csv_path)

    # 2. Chuẩn hóa tên cột dữ liệu thô
    rename_map = {
        'F_MASV': 'MSSV', 'student_id': 'MSSV', 'ma_sv': 'MSSV',
        'F_MAMH': 'ma_mon', 'course_code': 'ma_mon',
        'F_DIEM2': 'diem_thang_10', 'diem': 'diem_thang_10', 'score': 'diem_thang_10',
        'F_DVHT': 'tin_chi', 'credits': 'tin_chi'
    }
    df_raw = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})

    # Làm sạch dữ liệu
    df_raw['MSSV'] = df_raw['MSSV'].astype(str).str.strip()
    df_raw['ma_mon'] = df_raw['ma_mon'].astype(str).str.strip().str.upper()
    df_raw['diem_thang_10'] = pd.to_numeric(df_raw['diem_thang_10'], errors='coerce')
    df_raw['is_passed'] = df_raw['diem_thang_10'] >= 4.0

    # 3. Lọc riêng các học phần Ngoại ngữ bắt đầu bằng "FL"
    df_fl = df_raw[df_raw['ma_mon'].str.startswith('FL', na=False)].copy()

    if df_fl.empty:
        print("Không tìm thấy học phần nào có mã bắt đầu bằng 'FL'.")
        return

    # 4. Gom nhóm tính chỉ số từng môn FL theo sinh viên
    fl_stats = df_fl.groupby(['MSSV', 'ma_mon']).agg(
        diem_max=('diem_thang_10', 'max'),
        so_lan_hoc=('diem_thang_10', 'count'),
        so_lan_hoan_thanh=('is_passed', lambda x: int(x.sum()))
    ).reset_index()

    fl_stats['ti_le_hoan_thanh'] = np.where(
        fl_stats['so_lan_hoc'] > 0,
        (fl_stats['so_lan_hoan_thanh'] / fl_stats['so_lan_hoc']).round(4),
        0.0
    )

    # 5. Xoay bảng (Pivot) thành các cột ngang chuẩn hóa cấu trúc Wide Matrix
    piv_score = fl_stats.pivot(index='MSSV', columns='ma_mon', values='diem_max')
    piv_score.columns = [f'DIEM_{col}' for col in piv_score.columns]

    piv_attempts = fl_stats.pivot(index='MSSV', columns='ma_mon', values='so_lan_hoc').fillna(0).astype(int)
    piv_attempts.columns = [f'SOLAN_HOC_{col}' for col in piv_attempts.columns]

    piv_passed = fl_stats.pivot(index='MSSV', columns='ma_mon', values='so_lan_hoan_thanh').fillna(0).astype(int)
    piv_passed.columns = [f'SOLAN_HOAN_THANH_{col}' for col in piv_passed.columns]

    piv_rate = fl_stats.pivot(index='MSSV', columns='ma_mon', values='ti_le_hoan_thanh').fillna(0.0)
    piv_rate.columns = [f'TILE_HOAN_THANH_{col}' for col in piv_rate.columns]

    fl_matrix = pd.concat([piv_score, piv_attempts, piv_passed, piv_rate], axis=1).reset_index()

    # 6. Loại bỏ các cột FL cũ trong Wide Matrix (nếu có) để ghi đè dữ liệu mới
    fl_prefixes = ('DIEM_FL', 'SOLAN_HOC_FL', 'SOLAN_HOAN_THANH_FL', 'TILE_HOAN_THANH_FL')
    cols_to_drop = [c for c in df_wide.columns if c.startswith(fl_prefixes)]
    if cols_to_drop:
        df_wide = df_wide.drop(columns=cols_to_drop)

    # 7. Merge dữ liệu học phần FL vào ma trận Wide chính
    df_wide['MSSV'] = df_wide['MSSV'].astype(str).str.strip()
    df_final = pd.merge(df_wide, fl_matrix, on='MSSV', how='left')

    # 8. Lưu kết quả ra file CSV
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Đã cập nhật thành công {len(fl_matrix.columns) - 1} cột thuộc tính mã FL vào file: {output_csv_path}")

if __name__ == '__main__':
    merge_fl_courses_to_wide_matrix()