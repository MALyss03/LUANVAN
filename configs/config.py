from pathlib import Path

# ================================
# PROJECT
# ================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ================================
# DATA DIRECTORIES
# ================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# ================================
# OUTPUT DIRECTORIES
# ================================
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
SHAP_DIR = OUTPUT_DIR / "shap"

# ================================
# DATASETS & REPORTS
# ================================
RAW_DATASET = RAW_DATA_DIR / "DIEMTHAYCANG_FINAL.csv"
CLEAN_DATASET = INTERIM_DATA_DIR / "clean_dataset.csv"
VALIDATION_REPORT = REPORT_DIR / "validation_report.csv"
INVALID_ROWS = REPORT_DIR / "invalid_rows.csv"

# ================================
# CONFIG & API KEYS
# ================================
RANDOM_STATE = 42
OPENAI_API_KEY = "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"