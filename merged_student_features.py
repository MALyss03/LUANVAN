from pathlib import Path
import pandas as pd

# 1. Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Tự động tạo thư mục data/processed nếu chưa có
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = INTERIM_DIR / "processed_student_features.csv"
MATRIX_PATH = INTERIM_DIR / "featured_student_matrix.csv"
OUTPUT_PATH = PROCESSED_DIR / "merged_student_features_final.csv"


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

  # 2. Sắp xếp dữ liệu theo MSSV và học kỳ
  df_processed_sorted = df_processed.sort_values(["mssv", "so_hk_da_hoc"])

  # 3. Trích xuất đặc trưng kỳ mới nhất
  latest_cols = [
      "mssv",
      "nhhk",
      "so_hk_da_hoc",
      "nam_thu",
      "dtbchk",
      "dtbctl",
      "gpa_tich_luy",
      "tc_dang_ky",
      "tc_dat_hk",
      "tc_khong_dat_hk",
      "tc_tich_luy",
      "tc_no_dong",
      "drl_hk",
      "dtbchk_prev",
      "gpa_gap_hk_vs_tl",
      "dtbchk_trend",
      "dtbctl_safety_margin",
      "dtbchk_safety_margin",
      "credit_load_ratio",
      "credit_fail_ratio_hk",
      "no_dong_risk_ratio",
      "credit_accumulation_speed",
      "drl_level",
      "is_drl_risk",
      "so_mon_hoc_hk",
  ]

  existing_latest_cols = [c for c in latest_cols if c in df_processed.columns]
  df_latest = df_processed_sorted.groupby("mssv").last().reset_index()[
      existing_latest_cols
  ]

  rename_map = {
      "nhhk": "nhhk_hien_tai",
      "so_hk_da_hoc": "tong_so_hk_da_hoc",
      "nam_thu": "nam_thu_hien_tai",
      "dtbchk": "dtbchk_gannhat",
      "dtbctl": "dtbctl_hien_tai",
      "drl_hk": "drl_gannhat",
      "tc_dang_ky": "tc_dang_ky_gannhat",
      "tc_dat_hk": "tc_dat_gannhat",
      "tc_khong_dat_hk": "tc_khong_dat_gannhat",
      "tc_tich_luy": "tc_tich_luy_hien_tai",
      "tc_no_dong": "tc_no_dong_hien_tai",
      "so_mon_hoc_hk": "so_mon_hoc_gannhat",
  }
  df_latest = df_latest.rename(columns=rename_map)

  # 4. Tính toán chỉ số lịch sử
  df_stats = (
      df_processed_sorted.groupby("mssv")
      .agg(
          dtbchk_mean=("dtbchk", "mean"),
          dtbchk_min=("dtbchk", "min"),
          dtbchk_max=("dtbchk", "max"),
          drl_mean=("drl_hk", "mean"),
          drl_min=("drl_hk", "min"),
          so_hk_bi_canh_bao_drl=("is_drl_risk", "sum"),
          tc_dang_ky_trung_binh_hk=("tc_dang_ky", "mean"),
          tc_fail_trung_binh_hk=("tc_khong_dat_hk", "mean"),
      )
      .reset_index()
  )

  # 5. Gộp bảng
  print("🔄 Đang gộp dữ liệu...")
  df_final = pd.merge(df_matrix, df_latest, on="mssv", how="left")
  df_final = pd.merge(df_final, df_stats, on="mssv", how="left")

  # 6. Xuất kết quả
  df_final.to_csv(OUTPUT_PATH, index=False)
  print(f"✅ Đã xuất file thành công tại: {OUTPUT_PATH}")
  print(
      f"📈 Kích thước ma trận cuối cùng: {df_final.shape[0]} hàng x"
      f" {df_final.shape[1]} cột"
  )


if __name__ == "__main__":
  merge_student_features()