import os
from pathlib import Path
import numpy as np
import pandas as pd

# Cấu hình chống đơ luồng NumPy/Pandas trên Windows Anaconda
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 1. Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Tự động tạo thư mục data/processed nếu chưa có
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = INTERIM_DIR / "processed_student_features.csv"
MATRIX_PATH = INTERIM_DIR / "featured_student_matrix.csv"
OUTPUT_PATH = PROCESSED_DIR / "merged_student_features_final.csv"


def convert_gpa_10_to_4(score):
  """Hàm bổ trợ quy đổi GPA từ Thang 10 sang Thang 4.0 (Nếu file chưa có sẵn cột Thang 4.0)."""
  if pd.isna(score) or score is None:
    return np.nan
  try:
    val = float(score)
  except (ValueError, TypeError):
    return np.nan

  if val >= 8.5:
    return 4.0
  elif val >= 8.0:
    return 3.5
  elif val >= 7.0:
    return 3.0
  elif val >= 6.5:
    return 2.5
  elif val >= 5.5:
    return 2.0
  elif val >= 5.0:
    return 1.5
  elif val >= 4.0:
    return 1.0
  else:
    return 0.0


def merge_student_features():
  print("🚀 Đang tải dữ liệu từ data/interim/...")

  if not PROCESSED_PATH.exists() or not MATRIX_PATH.exists():
    print(
        f"❌ Lỗi: Không tìm thấy file dữ liệu tại:\n - {PROCESSED_PATH}\n -"
        f" {MATRIX_PATH}"
    )
    return

  df_processed = pd.read_csv(PROCESSED_PATH)
  df_matrix = pd.read_csv(MATRIX_PATH)

  # -------------------------------------------------------------------------
  # 2. TỰ ĐỘNG BỔ SUNG CỘT THANG 4.0 NẾU CHƯA CÓ TRONG FILE PROCESSED
  # -------------------------------------------------------------------------
  if "dtbchk_thang_4" not in df_processed.columns and "dtbchk" in df_processed:
    df_processed["dtbchk_thang_4"] = df_processed["dtbchk"].apply(
        convert_gpa_10_to_4
    )

  if "dtbctl_thang_4" not in df_processed.columns and "dtbctl" in df_processed:
    df_processed["dtbctl_thang_4"] = df_processed["dtbctl"].apply(
        convert_gpa_10_to_4
    )

  if (
      "gpa_tich_luy_thang_4" not in df_processed.columns
      and "gpa_tich_luy" in df_processed
  ):
    df_processed["gpa_tich_luy_thang_4"] = df_processed["gpa_tich_luy"].apply(
        convert_gpa_10_to_4
    )

  if (
      "dtbchk_prev_thang_4" not in df_processed.columns
      and "dtbchk_prev" in df_processed
  ):
    df_processed["dtbchk_prev_thang_4"] = df_processed["dtbchk_prev"].apply(
        convert_gpa_10_to_4
    )

  # -------------------------------------------------------------------------
  # 3. SẮP XẾP DỮ LIỆU THEO MSSV VÀ HỌC KỲ
  # -------------------------------------------------------------------------
  df_processed_sorted = df_processed.sort_values(["mssv", "so_hk_da_hoc"])

  # 4. Trích xuất đặc trưng kỳ mới nhất (Đã bổ sung các cột Thang 4.0)
  latest_cols = [
      "mssv",
      "nhhk",
      "so_hk_da_hoc",
      "nam_thu",
      # GPA Thang 10 & Thang 4.0 Học kỳ
      "dtbchk",
      "dtbchk_thang_4",
      # GPA Thang 10 & Thang 4.0 Tích lũy
      "dtbctl",
      "dtbctl_thang_4",
      "gpa_tich_luy",
      "gpa_tich_luy_thang_4",
      # Tín chỉ
      "tc_dang_ky",
      "tc_dat_hk",
      "tc_khong_dat_hk",
      "tc_tich_luy",
      "tc_no_dong",
      # Điểm Rèn luyện & Kỳ trước
      "drl_hk",
      "dtbchk_prev",
      "dtbchk_prev_thang_4",
      "gpa_gap_hk_vs_tl",
      "dtbchk_trend",
      "dtbctl_safety_margin",
      "dtbchk_safety_margin",
      # Tỷ lệ & Chỉ số rủi ro
      "credit_load_ratio",
      "credit_fail_ratio_hk",
      "no_dong_risk_ratio",
      "credit_accumulation_speed",
      "drl_level",
      "is_drl_risk",
      "so_mon_hoc_hk",
  ]

  existing_latest_cols = [
      c for c in latest_cols if c in df_processed_sorted.columns
  ]
  df_latest = df_processed_sorted.groupby("mssv").last().reset_index()[
      existing_latest_cols
  ]

  rename_map = {
      "nhhk": "nhhk_hien_tai",
      "so_hk_da_hoc": "tong_so_hk_da_hoc",
      "nam_thu": "nam_thu_hien_tai",
      "dtbchk": "dtbchk_gannhat",
      "dtbchk_thang_4": "dtbchk_gannhat_thang_4",
      "dtbctl": "dtbctl_hien_tai",
      "dtbctl_thang_4": "dtbctl_hien_tai_thang_4",
      "drl_hk": "drl_gannhat",
      "tc_dang_ky": "tc_dang_ky_gannhat",
      "tc_dat_hk": "tc_dat_gannhat",
      "tc_khong_dat_hk": "tc_khong_dat_gannhat",
      "tc_tich_luy": "tc_tich_luy_hien_tai",
      "tc_no_dong": "tc_no_dong_hien_tai",
      "so_mon_hoc_hk": "so_mon_hoc_gannhat",
  }
  df_latest = df_latest.rename(columns=rename_map)

  # -------------------------------------------------------------------------
  # 5. TÍNH TOÁN CHỈ SỐ LỊCH SỬ (Đã bổ sung Thống kê Thang 4.0)
  # -------------------------------------------------------------------------
  agg_dict = {
      "dtbchk_mean": ("dtbchk", "mean"),
      "dtbchk_min": ("dtbchk", "min"),
      "dtbchk_max": ("dtbchk", "max"),
      "drl_mean": ("drl_hk", "mean"),
      "drl_min": ("drl_hk", "min"),
      "so_hk_bi_canh_bao_drl": ("is_drl_risk", "sum"),
      "tc_dang_ky_trung_binh_hk": ("tc_dang_ky", "mean"),
      "tc_fail_trung_binh_hk": ("tc_khong_dat_hk", "mean"),
  }

  # Thêm thống kê GPA Thang 4.0 nếu có cột dtbchk_thang_4
  if "dtbchk_thang_4" in df_processed_sorted.columns:
    agg_dict["dtbchk_thang_4_mean"] = ("dtbchk_thang_4", "mean")
    agg_dict["dtbchk_thang_4_min"] = ("dtbchk_thang_4", "min")
    agg_dict["dtbchk_thang_4_max"] = ("dtbchk_thang_4", "max")

  df_stats = (
      df_processed_sorted.groupby("mssv").agg(**agg_dict).reset_index()
  )

  # Làm tròn số thực cho đẹp bảng dữ liệu
  float_cols = df_stats.select_dtypes(include=["float64", "float32"]).columns
  df_stats[float_cols] = df_stats[float_cols].round(2)

  # -------------------------------------------------------------------------
  # 6. GỘP BẢNG VÀ XUẤT KẾT QUẢ
  # -------------------------------------------------------------------------
  print("🔄 Đang gộp dữ liệu...")
  df_final = pd.merge(df_matrix, df_latest, on="mssv", how="left")
  df_final = pd.merge(df_final, df_stats, on="mssv", how="left")

  # Xuất kết quả
  df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
  print(f"✅ Đã xuất file thành công tại: {OUTPUT_PATH}")
  print(
      f"📈 Kích thước ma trận cuối cùng: {df_final.shape[0]} hàng x"
      f" {df_final.shape[1]} cột"
  )


if __name__ == "__main__":
  merge_student_features()