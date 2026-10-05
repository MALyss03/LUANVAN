import os
# Khắc phục lỗi xung đột thư viện OpenMP (libiomp5md.dll) trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys
import re
import json
import hashlib
import base64
import mimetypes
from datetime import datetime
from io import BytesIO
import tempfile
import shutil
import textwrap
import unicodedata
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Groq API: ưu tiên cấu hình lưu bền vững của ADMIN, sau đó mới fallback về biến môi trường.
HARDCODED_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Import OpenAI library (Dùng để kết nối Groq API via base_url)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    openai = None
    HAS_OPENAI = False

# Import PyTorch an toàn
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    PdfReader = None
    HAS_PYPDF = False

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN IMPORT SYSTEM & BASE64 LOGO
# -----------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2] if len(FILE_PATH.parents) >= 3 else FILE_PATH.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def get_ctu_logo_b64():
    """Tải ảnh Logo CTU từ thư mục chỉ định và mã hóa Base64."""
    logo_path = Path(r"D:\StudentProgressAI\data\pics\LogoCTU.png")
    if not logo_path.exists():
        logo_path = PROJECT_ROOT / "data" / "pics" / "LogoCTU.png"
    if logo_path.exists():
        try:
            return base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        except Exception:
            return None
    return None


def get_login_hero_image_b64():
    """Đọc ảnh nền CTU cho màn hình đăng nhập.

    Ưu tiên đúng đường dẫn Windows mà hệ thống đang dùng:
    Đường dẫn Windows: D:\\StudentProgressAI\\data\\pics\\AnhCTU.jpg
    Sau đó mới tìm bản sao tương đối theo PROJECT_ROOT.
    """
    candidates = [
        Path(r"D:\StudentProgressAI\data\pics\AnhCTU.jpg"),
        PROJECT_ROOT / "data" / "pics" / "AnhCTU.jpg",
        PROJECT_ROOT / "data" / "pics" / "anhctu.jpg",
        PROJECT_ROOT / "data" / "pics" / "anh CTU.jpg",
    ]
    image_path = next((p for p in candidates if p.is_file()), None)
    if image_path is None:
        return None, None

    try:
        # Không để lỗi MIME làm mất toàn bộ ảnh nền.
        mime = mimetypes.guess_type(str(image_path))[0]
        if not mime:
            suffix = image_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return encoded, mime
    except (OSError, ValueError, TypeError):
        return None, None

# -----------------------------------------------------------------------------
# 1A. AUTHENTICATION / RBAC / ADMIN STORAGE
# -----------------------------------------------------------------------------
ADMIN_DATA_DIR = PROJECT_ROOT / "data" / "admin"
ADMIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
AUTH_FILE = ADMIN_DATA_DIR / "users.json"
PROMPT_FILE = ADMIN_DATA_DIR / "groq_prompt.json"
GROQ_CONFIG_FILE = ADMIN_DATA_DIR / "groq_config.json"
GROQ_MASTER_KEY_FILE = ADMIN_DATA_DIR / ".groq_master.key"
CURRICULUM_FILE = ADMIN_DATA_DIR / "curriculum_registry.json"
CERT_FILE = ADMIN_DATA_DIR / "language_certificates.json"
OFFERING_FILE = ADMIN_DATA_DIR / "course_offerings.json"
DEFAULT_GROQ_PROMPT = (
    "Bạn là Cố vấn học tập AI chuyên nghiệp của Đại học Cần Thơ (CTU). "
    "Dựa trên dữ liệu học tập chính thức được cung cấp, hãy phân tích khách quan, "
    "không bịa dữ liệu và đề xuất kế hoạch học tập khả thi. Phân biệt rõ dữ liệu hệ thống, "
    "suy luận của mô hình và lời khuyên. Ưu tiên CTĐT, học phần tiên quyết, môn nợ và điều kiện tốt nghiệp."
)
INITIAL_ADMIN_PASSWORD = os.getenv("ADMIN_INITIAL_PASSWORD", "admin123")

try:
    from cryptography.fernet import Fernet, InvalidToken
    HAS_FERNET = True
except Exception:
    Fernet = None
    InvalidToken = Exception
    HAS_FERNET = False


def _hash_password(password):
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


def _load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path, data):
    """Lưu JSON atomic để tránh file hỏng nếu app bị reset giữa lúc ghi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _get_groq_fernet():
    """Tạo bộ mã hóa local cho API key; có thể override bằng STUDENT_PROGRESSAI_SECRET."""
    if not HAS_FERNET:
        return None
    env_secret = os.getenv("STUDENT_PROGRESSAI_SECRET", "").strip()
    if env_secret:
        raw = hashlib.sha256(env_secret.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(raw))
    try:
        if GROQ_MASTER_KEY_FILE.exists():
            key = GROQ_MASTER_KEY_FILE.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            GROQ_MASTER_KEY_FILE.write_bytes(key)
        return Fernet(key)
    except Exception:
        return None


def load_persisted_groq_api_key():
    """Chỉ đọc key đã được ADMIN lưu trong storage; không fallback về environment."""
    data = _load_json(GROQ_CONFIG_FILE, {})
    if not isinstance(data, dict):
        data = {}
    encrypted = str(data.get("api_key_encrypted", "") or "")
    if encrypted:
        fernet = _get_groq_fernet()
        if fernet:
            try:
                return fernet.decrypt(encrypted.encode("utf-8")).decode("utf-8").strip()
            except Exception:
                pass
    legacy = str(data.get("api_key", "") or "").strip()
    if legacy:
        try:
            save_groq_api_key(legacy)
        except Exception:
            pass
        return legacy
    return ""


def load_groq_api_key():
    """Key hiệu lực: key ADMIN đã lưu, nếu chưa có thì dùng GROQ_API_KEY từ environment."""
    return load_persisted_groq_api_key() or str(HARDCODED_GROQ_API_KEY or "").strip()


def save_groq_api_key(api_key):
    api_key = str(api_key or "").strip()
    if not api_key:
        return False
    fernet = _get_groq_fernet()
    payload = {"schema_version": 2, "updated_at": datetime.now().isoformat(timespec="seconds")}
    if fernet:
        payload["api_key_encrypted"] = fernet.encrypt(api_key.encode("utf-8")).decode("utf-8")
    else:
        payload["api_key"] = api_key
        payload["storage_warning"] = "cryptography_unavailable"
    _save_json(GROQ_CONFIG_FILE, payload)
    try:
        cached_fn = globals().get("get_chatgpt_advice")
        if cached_fn is not None and hasattr(cached_fn, "clear"):
            cached_fn.clear()
    except Exception:
        pass
    return True


def delete_groq_api_key():
    if GROQ_CONFIG_FILE.exists():
        try:
            GROQ_CONFIG_FILE.unlink()
        except Exception:
            pass
    try:
        cached_fn = globals().get("get_chatgpt_advice")
        if cached_fn is not None and hasattr(cached_fn, "clear"):
            cached_fn.clear()
    except Exception:
        pass


def test_groq_api_key(api_key):
    """Kiểm tra key bằng models.list(), không gửi dữ liệu sinh viên."""
    key = str(api_key or "").strip()
    if not key:
        return False, "Chưa có API key."
    if not (HAS_OPENAI and hasattr(openai, "OpenAI")):
        return False, "Chưa cài thư viện openai."
    try:
        client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        page = client.models.list()
        return True, f"Kết nối thành công. Groq trả về {len(getattr(page, 'data', []) or [])} model(s)."
    except Exception as exc:
        return False, f"Không kết nối được Groq: {exc}"


def load_users():
    users = _load_json(AUTH_FILE, None)
    if not isinstance(users, dict) or not users:
        users = {"admin": {"password_hash": _hash_password(INITIAL_ADMIN_PASSWORD), "role": "admin", "active": True}}
        _save_json(AUTH_FILE, users)
    return users


def authenticate(username, password):
    item = load_users().get(str(username).strip())
    if not item or not item.get("active", True):
        return None
    if item.get("password_hash") != _hash_password(password):
        return None
    return {"username": str(username).strip(), "role": item.get("role", "user"), "mssv": item.get("mssv")}


def require_login():
    if st.session_state.get("auth_user"):
        return True

    logo_b64 = get_ctu_logo_b64()
    hero_b64, hero_mime = get_login_hero_image_b64()

    if hero_b64:
        hero_image_html = (
            f'<img class="login-hero-image" '
            f'src="data:{hero_mime};base64,{hero_b64}" '
            f'alt="Khuôn viên Đại học Cần Thơ">'
        )
    else:
        # Không render chữ placeholder trên nền xanh.
        # Khi có ảnh tại D:\StudentProgressAI\data\pics\AnhCTU.jpg, ảnh sẽ được hiển thị chìm dưới lớp overlay.
        hero_image_html = '<div class="login-hero-placeholder" aria-hidden="true"></div>'

    if logo_b64:
        logo_html = (
            f'<div class="login-mini-logo">'
            f'<img src="data:image/png;base64,{logo_b64}" alt="Logo Đại học Cần Thơ">'
            f'</div>'
        )
    else:
        logo_html = '<div class="login-mini-logo login-logo-fallback">CTU</div>'

    st.markdown(
        """
        <div class="login-page-marker"></div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    with left_col:
        login_visual_html = textwrap.dedent(f"""
            <section class="login-visual">
                <div class="login-visual-media">
                    <img class="login-hero-image-blur" src="data:{hero_mime};base64,{hero_b64}" alt="" aria-hidden="true">
                    {hero_image_html}
                    <div class="login-visual-overlay"></div>
                </div>
                <div class="login-visual-content">
                    <div class="login-visual-badge">ĐẠI HỌC CẦN THƠ</div>
                    <div class="login-visual-logo-wrap">{logo_html}</div>
                    <div class="login-visual-kicker">HỆ THỐNG HỖ TRỢ HỌC TẬP</div>
                    <h2>StudentProgressAI</h2>
                    <p class="login-visual-text">Đồng hành cùng sinh viên trong việc theo dõi tiến độ, phân tích kết quả học tập và xây dựng lộ trình học tập cá nhân hóa.</p>
                    <div class="login-feature-list">
                        <div class="login-feature"><span>01</span><div><b>Dự báo tiến độ</b><small>Phân tích dữ liệu học tập theo thời gian.</small></div></div>
                        <div class="login-feature"><span>02</span><div><b>XAI / SHAP</b><small>Giải thích các yếu tố ảnh hưởng đến dự đoán.</small></div></div>
                        <div class="login-feature"><span>03</span><div><b>Lộ trình cá nhân hóa</b><small>Đề xuất học phần theo hồ sơ từng sinh viên.</small></div></div>
                    </div>
                </div>
                <div class="login-visual-footer">Can Tho University · StudentProgressAI</div>
            </section>
        """).strip()
        st.markdown(login_visual_html, unsafe_allow_html=True)

    with right_col:
        login_panel_html = textwrap.dedent("""
            <section class="login-panel-head login-panel-head-no-logo">
                <div class="login-panel-kicker">CỔNG TRUY CẬP</div>
                <h1>Chào mừng trở lại</h1>
                <p>Đăng nhập để tiếp tục sử dụng hệ thống StudentProgressAI.</p>
            </section>
        """).strip()
        st.markdown(login_panel_html, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Tên đăng nhập",
                placeholder="Nhập tài khoản",
                key="login_username",
            )
            password = st.text_input(
                "Mật khẩu",
                type="password",
                placeholder="Nhập mật khẩu",
                key="login_password",
            )

            st.markdown(
                '<div class="login-form-note">Tài khoản được quản lý theo phân quyền của hệ thống.</div>',
                unsafe_allow_html=True,
            )

            submitted = st.form_submit_button(
                "Đăng nhập vào hệ thống",
                use_container_width=True,
            )

            if submitted:
                user = authenticate(username, password)
                if user:
                    st.session_state["auth_user"] = user
                    st.session_state["login_success_at"] = datetime.now().isoformat(timespec="seconds")
                    st.rerun()
                else:
                    st.error("Sai tài khoản, mật khẩu hoặc tài khoản đã bị khóa.")

        st.markdown(
            """
            <div class="login-security">
                <span>🔒</span>
                <div>
                    <b>Phiên đăng nhập được bảo vệ</b>
                    <small>Quyền truy cập sẽ được xác định theo vai trò tài khoản.</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class=\"login-page-end-marker\"></div>", unsafe_allow_html=True)
    return False


def is_admin():
    return st.session_state.get("auth_user", {}).get("role") == "admin"


def load_prompt_config():
    data = _load_json(PROMPT_FILE, {})
    if not isinstance(data, dict):
        data = {}
    prompt = str(data.get("prompt") or DEFAULT_GROQ_PROMPT)
    version = str(data.get("version") or hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12])
    return {"prompt": prompt, "version": version, "updated_at": data.get("updated_at", "")}


def load_prompt_template():
    return load_prompt_config()["prompt"]


def get_prompt_version():
    return load_prompt_config()["version"]


def save_prompt_template(prompt):
    prompt = str(prompt or "").strip() or DEFAULT_GROQ_PROMPT
    version = hashlib.sha1((prompt + "|" + datetime.now().isoformat()).encode("utf-8")).hexdigest()[:12]
    _save_json(PROMPT_FILE, {
        "schema_version": 2,
        "prompt": prompt,
        "version": version,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    try:
        cached_fn = globals().get("get_chatgpt_advice")
        if cached_fn is not None and hasattr(cached_fn, "clear"):
            cached_fn.clear()
    except Exception:
        pass


def load_curriculum_registry():
    data = _load_json(CURRICULUM_FILE, {})
    return data if isinstance(data, dict) else {}


def _invalidate_curriculum_caches():
    try:
        cached_fn = globals().get("load_personalized_recommendation_engine")
        if cached_fn is not None and hasattr(cached_fn, "clear"):
            cached_fn.clear()
    except Exception:
        pass
    try:
        cached_fn = globals().get("prepare_cohort_dataframe")
        if cached_fn is not None and hasattr(cached_fn, "clear"):
            cached_fn.clear()
    except Exception:
        pass


def save_curriculum_registry(data):
    _save_json(CURRICULUM_FILE, data)
    _invalidate_curriculum_caches()


def load_certificate_registry():
    data = _load_json(CERT_FILE, {})
    return data if isinstance(data, dict) else {}


def save_certificate_registry(data):
    _save_json(CERT_FILE, data)


# -----------------------------------------------------------------------------
# CURRICULUM KNOWLEDGE BASE / PDF + OCR / STUDENT MAPPING
# -----------------------------------------------------------------------------
DECISION_BATCH_MAP = {
    "3334/QĐ-ĐHCT": ["B21", "B22"],
    "2423/QĐ-ĐHCT": ["B23", "B24"],
    "3922/QĐ-ĐHCT": ["B25", "B26"],
}

def _normalize_major_key(text):
    s = str(text or "").upper()
    if any(k in s for k in ["AN TOÀN", "AN TOAN", "ATTT", "7480202"]): return "ATTT"
    if any(k in s for k in ["MẠNG", "MANG", "TRUYỀN THÔNG", "TRUYEN THONG", "7480102"]): return "MMT"
    return "UNKNOWN"

def _batch_to_bcode(value):
    s = str(value or "").upper()
    m = re.search(r"\bB(2[1-9]|3[0-9])\b", s)
    if m: return "B" + m.group(1)
    m = re.search(r"\bK(4[7-9]|5[0-9])\b", s)
    if not m: m = re.search(r"KH[ÓO]A\s*(4[7-9]|5[0-9])", s)
    if m: return "B" + str(int(m.group(1)) - 26)
    return ""

def _normalize_decision_number(text):
    s = str(text or "")
    m = re.search(r"(\d{2,5})\s*/\s*Q[ĐD]\s*[-–]?\s*(?:ĐHCT|DHCT)", s, re.I)
    return f"{m.group(1)}/QĐ-ĐHCT" if m else ""

def _extract_decision_metadata(text):
    decision = _normalize_decision_number(text)
    patterns = [
        r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(20\d{2})",
        r"ngày\s*(\d{1,2})[/-](\d{1,2})[/-](20\d{2})",
    ]
    date_iso = ""
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            d, mo, y = m.groups()
            date_iso = f"{int(d):02d}/{int(mo):02d}/{y}"
            break
    year = int(date_iso[-4:]) if date_iso else None
    candidates = DECISION_BATCH_MAP.get(decision, [])
    return decision, date_iso, year, candidates

def _extract_pdf_text(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    texts = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(texts), len(reader.pages)

def _render_pdf_pages(file_bytes, scale=1.8, page_numbers=None):
    """Render PDF pages without requiring PyMuPDF. Prefer PyMuPDF, then pdf2image/Poppler."""
    errors = []
    # Backend 1: PyMuPDF (optional)
    try:
        import fitz
        import numpy as _np
        import cv2
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for _idx, page in enumerate(doc, start=1):
            if page_numbers and _idx not in set(page_numbers):
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            arr = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n >= 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            pages.append(img)
        doc.close()
        if pages:
            return pages, "PYMUPDF", errors
    except Exception as e:
        errors.append(f"PyMuPDF unavailable: {e}")

    # Backend 2: pdf2image + Poppler.
    try:
        from pdf2image import convert_from_bytes
        import numpy as _np
        import cv2
        dpi = max(150, int(100 * scale))
        kwargs = {}
        if page_numbers:
            kwargs["first_page"] = min(page_numbers)
            kwargs["last_page"] = max(page_numbers)
        pil_pages = convert_from_bytes(file_bytes, dpi=dpi, fmt="png", thread_count=1, **kwargs)
        pages = [cv2.cvtColor(_np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR) for im in pil_pages]
        if pages:
            return pages, "PDF2IMAGE", errors
    except Exception as e:
        errors.append(f"pdf2image/Poppler unavailable: {e}")

    # Backend 3: pypdfium2.
    try:
        import pypdfium2 as pdfium
        import numpy as _np
        import cv2
        pdf = pdfium.PdfDocument(file_bytes)
        indices = page_numbers or list(range(1, len(pdf) + 1))
        pages = []
        for idx in indices:
            page = pdf[idx - 1]
            bitmap = page.render(scale=max(1.5, scale * 1.6))
            pil = bitmap.to_pil().convert("RGB")
            pages.append(cv2.cvtColor(_np.array(pil), cv2.COLOR_RGB2BGR))
        if pages:
            return pages, "PDFIUM", errors
    except Exception as e:
        errors.append(f"pypdfium2 unavailable: {e}")
    return [], "NONE", errors

def _configure_tesseract():
    """Try common Windows, Linux, and macOS installations before running OCR."""
    try:
        import pytesseract
        
        # 1. Lấy từ biến môi trường (nếu có cấu hình)
        env_cmd = os.getenv("TESSERACT_CMD")
        if env_cmd and Path(env_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = env_cmd
            return pytesseract
            
        # 2. Kiểm tra xem lệnh tesseract có sẵn trong PATH không
        if shutil.which("tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "tesseract"
            return pytesseract
            
        # 3. Quét đường dẫn mặc định trên các hệ điều hành
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ):
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        return pytesseract
    except Exception:
        return None

def _ocr_pdf_bytes(file_bytes, page_numbers=None):
    """OCR fallback for scanned PDFs. Supports PyMuPDF OR pdf2image rendering."""
    pytesseract = _configure_tesseract()
    if pytesseract is None:
        return "", ["OCR dependencies unavailable: pytesseract is not installed or not in PATH"]
    pages, backend, warnings = _render_pdf_pages(file_bytes, scale=1.5, page_numbers=page_numbers)
    if not pages:
        return "", warnings or ["Không thể render PDF để OCR"]
    chunks = []
    actual_page_numbers = list(page_numbers) if page_numbers else list(range(1, len(pages) + 1))
    for page_no, image in zip(actual_page_numbers, pages):
        try:
            try:
                page_text = pytesseract.image_to_string(
                    image, lang="vie+eng", config="--oem 1 --psm 6", timeout=45
                )
            except Exception:
                page_text = pytesseract.image_to_string(
                    image, lang="eng", config="--oem 1 --psm 6", timeout=45
                )
            chunks.append(f"\n--- PAGE {page_no} OCR ({backend}) ---\n{page_text}")
        except Exception as page_error:
            warnings.append(f"Page {page_no} OCR failed/skipped: {page_error}")
    return "\n".join(chunks), warnings

def _clean_ocr_line(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _course_type_from_context(name, group, row_text):
    x = f"{name} {group} {row_text}".lower()
    if any(k in x for k in ["quốc phòng", "giáo dục thể chất", "ngoại ngữ", "điều kiện", "(*)"]):
        return "Điều kiện"
    if any(k in x for k in ["luận văn", "tiểu luận tốt nghiệp", "thực tập"]):
        return "Tốt nghiệp / Thực tập"
    if "tự chọn" in x:
        return "Tự chọn"
    return "Bắt buộc"

def _parse_prereq_from_text(row_text, code):
    hits = [x.upper() for x in re.findall(r"\b[A-Z]{1,5}\d{2,5}[A-Z]*\b", row_text.upper())]
    return [x for x in dict.fromkeys(hits) if x != code]

def _extract_courses_from_text(full_text, filename):
    courses, seen = [], set()
    group = "Chưa phân nhóm"
    lines = [_clean_ocr_line(x) for x in full_text.splitlines() if _clean_ocr_line(x)]
    code_re = r"[A-Z]{1,5}\d{2,5}[A-Z]*"
    group_keywords = ("khối kiến thức", "nhóm học phần", "giáo dục đại cương", "cơ sở ngành", "chuyên ngành")

    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in group_keywords) and len(line) < 150:
            group = line
            continue

        m = re.search(rf"(?:^\d{{1,3}}\s+)?({code_re})\s+(.+)", line, re.I)
        if not m:
            continue
        code = m.group(1).upper()
        tail = m.group(2)
        if code in seen:
            continue

        nums = list(re.finditer(r"\b(\d{1,2}(?:[.,]\d+)?)\b", tail))
        if not nums:
            continue
        credit_match = nums[0]
        try:
            credits = float(credit_match.group(1).replace(",", "."))
        except ValueError:
            continue
        if not (0 < credits <= 20):
            continue

        name = tail[:credit_match.start()].strip(" |-:;")
        if len(name) < 2 or name.lower() in {"tên học phần", "học phần"}:
            continue

        row_context = " ".join(lines[i:min(len(lines), i+2)])
        prereq = _parse_prereq_from_text(row_context, code)
        min_credits = 0
        mc = re.search(r"[≥>=]\s*(\d{2,3})\s*TC", row_context, re.I)
        if mc:
            min_credits = int(mc.group(1))

        ctype = _course_type_from_context(name, group, row_context)
        elective_group = group if "tự chọn" in group.lower() else ""
        courses.append({
            "code": code, "name": name, "credits": credits, "type": ctype,
            "group": group, "elective_group": elective_group,
            "prereq": prereq, "coreq": [], "min_credits": min_credits,
            "semester_hint": "", "source": filename,
            "evidence": line
        })
        seen.add(code)

    return courses

def _dedupe_and_validate_courses(courses):
    merged = {}
    issues = []
    for c in courses:
        code = str(c.get("code","")).upper().strip()
        if not re.fullmatch(r"[A-Z]{1,5}\d{2,5}[A-Z]*", code):
            continue
        c["code"] = code
        c["prereq"] = [p for p in dict.fromkeys(str(x).upper().strip() for x in c.get("prereq", [])) if p != code]
        if code in merged:
            old = merged[code]
            if len(str(c.get("name",""))) > len(str(old.get("name",""))):
                merged[code] = c
        else:
            merged[code] = c

    codes = set(merged)
    for c in merged.values():
        unknown = [p for p in c.get("prereq", []) if p not in codes]
        if unknown:
            issues.append({"course": c["code"], "issue": f"Tiên quyết chưa có trong danh sách trích xuất: {', '.join(unknown)}"})
    return list(merged.values()), issues

def _infer_semester_from_line(line):
    m = re.search(r"\b(I{1,3}|IV|V|VI|VII|VIII|IX)\b", str(line).upper())
    return m.group(1) if m else ""


# -----------------------------------------------------------------------------
# TABLE OCR ENGINE - coordinate-aware extraction for scanned CTĐT tables
# -----------------------------------------------------------------------------
def _cluster_positions(values, tolerance=8):
    values = sorted(int(v) for v in values)
    groups = []
    for v in values:
        if not groups or v - groups[-1][-1] > tolerance:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [int(sum(g) / len(g)) for g in groups]

def _detect_table_columns(image_bgr):
    try:
        import cv2
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
        h, w = bw.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 20)))
        vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch >= h * 0.18 and cw <= max(12, w * 0.025):
                xs.extend([x, x + cw])
        xs = _cluster_positions(xs, tolerance=max(5, w // 180))
        if len(xs) >= 8:
            return xs
    except Exception:
        pass
    return []

def _table_boundaries_from_columns(xs, width):
    if len(xs) >= 8:
        return sorted(xs)
    return [int(width * r) for r in (0.045, 0.095, 0.145, 0.475, 0.52, 0.565, 0.61, 0.655, 0.705, 0.82, 0.885, 0.94)]

def _assign_column(x, boundaries):
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= x < boundaries[i + 1]:
            return i
    return -1

def _normalize_course_code(code):
    code = re.sub(r"[^A-Z0-9]", "", str(code or "").upper())
    code = re.sub(r"^(CT|TC|TN|XH|QP|ML|FL|KN|KL|MI|CS|TC|CT)", lambda m: m.group(1), code)
    return code if re.fullmatch(r"[A-Z]{1,4}\d{2,4}[A-Z]?", code or "") else ""

def _extract_table_courses_from_pdf(file_bytes, filename):
    pytesseract_module = _configure_tesseract()
    if pytesseract_module is None:
        return [], ["Table OCR dependencies unavailable: pytesseract is not configured"]

    courses, warnings = [], []
    current_group = "Chưa phân nhóm"
    current_elective_group = ""
    pages, backend, render_warnings = _render_pdf_pages(file_bytes, scale=1.5)
    warnings.extend(render_warnings)
    if not pages:
        return [], warnings + ["Không thể render PDF để Table OCR"]

    for page_no, image in enumerate(pages, start=1):
        try:
            detected_columns = _detect_table_columns(image)
            boundaries = _table_boundaries_from_columns(detected_columns, image.shape[1])
            data = pytesseract_module.image_to_data(
                image, lang="vie+eng", config="--oem 1 --psm 6",
                output_type=pytesseract_module.Output.DICT, timeout=40
            )
            tokens = []
            for i, txt in enumerate(data.get("text", [])):
                txt = _clean_ocr_line(txt)
                try:
                    conf = float(data.get("conf", ["-1"])[i])
                except Exception:
                    conf = -1
                code_like = bool(re.search(r"[A-Za-z]{1,4}\s*\d{2,5}", txt))
                if txt and (conf >= 10 or code_like):
                    x, y = int(data["left"][i]), int(data["top"][i])
                    w, h = int(data["width"][i]), int(data["height"][i])
                    tokens.append({"text": txt, "x": x, "y": y, "w": w, "h": h})
            if not tokens:
                warnings.append(f"Table OCR page {page_no}: không nhận được token")
                continue

            row_tol = max(12, int(image.shape[0] / 180))
            rows = []
            for t in sorted(tokens, key=lambda z: (z["y"] + z["h"] // 2, z["x"])):
                cy = t["y"] + t["h"] // 2
                target = None
                for r in reversed(rows[-8:]):
                    if abs(cy - r["cy"]) <= row_tol:
                        target = r
                        break
                if target is None:
                    target = {"cy": cy, "tokens": []}
                    rows.append(target)
                target["tokens"].append(t)
                target["cy"] = int(sum(z["y"] + z["h"] // 2 for z in target["tokens"]) / len(target["tokens"]))

            for row in rows:
                toks = sorted(row["tokens"], key=lambda z: z["x"])
                full = " ".join(t["text"] for t in toks)
                low = full.lower()
                if (any(k in low for k in ("khối kiến thức", "giáo dục đại cương", "cơ sở ngành", "chuyên ngành", "tự chọn"))
                        and not re.search(r"\b[A-Z]{1,5}\d{2,5}[A-Z]?\b", full.upper())):
                    if len(full) >= 6:
                        current_group = full
                        if "tự chọn" in low:
                            current_elective_group = full
                    continue

                cols = {}
                for t in toks:
                    c = _assign_column(t["x"] + t["w"] // 2, boundaries)
                    if c >= 0:
                        cols.setdefault(c, []).append(t["text"])
                col = {k: " ".join(v) for k, v in cols.items()}

                m = re.search(r"\b([A-Z]{1,5}\s*[-.]?\s*\d{2,5}[A-Z]?)\b", full.upper())
                code = _normalize_course_code(m.group(1)) if m else ""
                if not code:
                    continue

                name = col.get(2, "").strip()
                if not name:
                    after = full[m.end():].strip() if m else full
                    cut = re.search(r"\s+(?:[1-9]|1[0-2])(?:[.,]0)?(?:\s|$)", after)
                    name = after[:cut.start()].strip(" |-:;") if cut else after
                name = re.sub(r"^\d{1,3}\s+", "", name).strip()
                if len(name) < 2 or name.lower() in {"tên học phần", "học phần"}:
                    continue

                credit_text = col.get(3, "")
                mcred = re.search(r"\b([1-9]|1[0-2])(?:[.,]0)?\b", credit_text)
                if not mcred:
                    after = full[m.end():] if m else full
                    nums = re.findall(r"\b([1-9]|1[0-2])\b", after)
                    credit_value = float(nums[-6]) if len(nums) >= 6 else (float(nums[0]) if nums else 0.0)
                else:
                    credit_value = float(mcred.group(1))
                if not (0 < credit_value <= 12):
                    continue

                mandatory = bool(re.search(r"\d", col.get(4, "")))
                elective_txt = col.get(5, "")
                if re.search(r"N\s*[1-9]", elective_txt, re.I):
                    current_elective_group = elective_txt.strip()
                elective = bool(re.search(r"\d|N\s*[1-9]", elective_txt, re.I)) and not mandatory
                prereq_text = col.get(8, "")
                coreq_text = col.get(9, "")
                semester = col.get(10, "")
                prereq = _parse_prereq_from_text(prereq_text, code)
                coreq = _parse_prereq_from_text(coreq_text, code)
                min_tc = 0
                mtc = re.search(r"[≥>=]+\s*(\d{1,3})\s*TC", full.upper())
                if mtc:
                    min_tc = int(mtc.group(1))
                ctype = "Tự chọn" if elective else ("Bắt buộc" if mandatory else _course_type_from_context(name, current_group, full))
                courses.append({
                    "code": code, "name": name, "credits": credit_value, "type": ctype,
                    "group": current_group, "elective_group": current_elective_group if elective else "",
                    "prereq": prereq, "coreq": coreq, "min_credits": min_tc,
                    "semester_hint": semester or _infer_semester_from_line(full),
                    "source": filename, "evidence": full, "source_page": page_no,
                    "extraction_method": f"TABLE_OCR_{backend}"
                })
        except Exception as pe:
            warnings.append(f"Table OCR page {page_no} failed/skipped: {pe}")

    return courses, warnings

def _merge_course_sources(table_courses, text_courses):
    merged = {}
    for c in table_courses + text_courses:
        code = c.get("code")
        if not code: continue
        old = merged.get(code)
        if old is None:
            merged[code] = c; continue
        for k, v in c.items():
            if (not old.get(k)) and v:
                old[k] = v
        if c.get("extraction_method") == "TABLE_OCR":
            old["extraction_method"] = "TABLE_OCR+TEXT"
    return list(merged.values())

def _validate_curriculum_totals(courses, totals):
    issues = []
    if not courses: return issues
    extracted = round(sum(float(c.get("credits") or 0) for c in courses), 1)
    if totals.get("total_credits") and abs(extracted - float(totals["total_credits"])) > 3:
        issues.append({"course": "__TOTAL__", "issue": f"Tổng TC trích xuất {extracted:g} khác tổng CTĐT {totals['total_credits']} TC; cần rà soát OCR/table parser."})
    return issues

def parse_curriculum_pdf(file_bytes, filename):
    if not HAS_PYPDF:
        raise RuntimeError("Chưa cài pypdf. Hãy chạy: pip install pypdf")

    raw_text, page_count = _extract_pdf_text(file_bytes)
    parser_mode = "TEXT"
    warnings = []
    min_text_threshold = max(300, page_count * 120)
    if len(raw_text.strip()) < min_text_threshold:
        ocr_text, ocr_warnings = _ocr_pdf_bytes(file_bytes, page_numbers=None)
        warnings.extend(ocr_warnings)
        if ocr_text.strip():
            if raw_text.strip():
                raw_text = raw_text + "\n\n--- OCR SUPPLEMENT ---\n" + ocr_text
                parser_mode = "HYBRID_TEXT_OCR"
            else:
                raw_text = ocr_text
                parser_mode = "OCR"
        else:
            warnings.append(
                "PDF gần như không có text và OCR chưa chạy được. "
                "Kiểm tra cấu hình Tesseract PATH hoặc biến TESSERACT_CMD."
            )

    decision, decision_date, decision_year, candidates = _extract_decision_metadata(raw_text)
    sample = f"{filename}\n{raw_text[:100000]}"
    major = _normalize_major_key(sample)
    batch_direct = _batch_to_bcode(sample)
    batch = batch_direct if batch_direct else (candidates[0] if len(candidates) == 1 else "")
    variant = "CLC" if "chất lượng cao" in sample.lower() or "chat luong cao" in sample.lower() else "STANDARD"

    table_courses, table_warnings = _extract_table_courses_from_pdf(file_bytes, filename)
    warnings.extend(table_warnings)
    text_courses = _extract_courses_from_text(raw_text, filename)
    courses = _merge_course_sources(table_courses, text_courses) if table_courses else text_courses
    courses, validation_issues = _dedupe_and_validate_courses(courses)

    total_match = re.search(
        r"Tổng\s+cộng\s*[:\-]?\s*(\d+)\s*TC.*?Bắt\s+buộc\s*[:\-]?\s*(\d+)\s*TC.*?Tự\s+chọn\s*[:\-]?\s*(\d+)\s*TC",
        raw_text, re.I | re.S
    )
    totals = {}
    if total_match:
        totals = {"total_credits": int(total_match.group(1)), "mandatory_credits": int(total_match.group(2)), "elective_credits": int(total_match.group(3))}

    validation_issues.extend(_validate_curriculum_totals(courses, totals))
    if len(table_courses) >= 60:
        confidence = "HIGH"
    elif parser_mode == "TEXT" and len(courses) >= 30:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM" if len(courses) >= 20 else "LOW"
    curriculum_text = "\n".join(
        f"{c['code']} | {c['name']} | {c['credits']} TC | {c['type']} | "
        f"Nhóm: {c['group']} | TQ: {', '.join(c['prereq']) or '-'} | "
        f"Điều kiện TC: {c.get('min_credits',0) or '-'}"
        for c in courses
    )
    return {
        "schema_version": 2,
        "curriculum_id": hashlib.sha1(file_bytes).hexdigest()[:16],
        "major": major, "batch": batch, "batch_candidates": candidates,
        "variant": variant, "filename": filename, "uploaded_at": datetime.now().isoformat(),
        "decision_number": decision, "decision_date": decision_date, "decision_year": decision_year,
        "page_count": page_count,
        "parser_mode": ("TABLE_OCR+" + parser_mode) if table_courses else parser_mode,
        "parse_confidence": confidence,
        "course_count": len(courses), "table_course_count": len(table_courses),
        "text_course_count": len(text_courses), "courses": courses, "validation_issues": validation_issues,
        "totals": totals, "active": False,
        "parser_note": "OCR/PDF extraction is evidence-based. Admin must review low-confidence rows and activate only after validation.",
        "parser_warnings": warnings,
        "curriculum_text": curriculum_text,
        "raw_text": raw_text[:160000],
        "pdf_base64": base64.b64encode(file_bytes).decode("ascii")
    }

def resolve_student_batch(mssv="", class_code="", major_text=""):
    evidence = []
    for label, value in [("MSSV", mssv), ("Mã lớp", class_code), ("Ngành", major_text)]:
        b = _batch_to_bcode(value)
        if b:
            evidence.append((b, label))
    if evidence:
        return evidence[0][0], {"status": "EXACT", "evidence": evidence}
    return "", {"status": "UNKNOWN", "evidence": evidence}

def get_admin_curriculum(major_key, batch_code, variant=None):
    registry = load_curriculum_registry()
    exact = []
    for item in registry.values():
        if not isinstance(item, dict) or not item.get("active"):
            continue
        if item.get("major") == major_key and item.get("batch") == batch_code:
            if variant is None or item.get("variant", "STANDARD") == variant:
                exact.append(item)
    if exact:
        exact.sort(key=lambda x: x.get("uploaded_at",""), reverse=True)
        return exact[0]
    return None

def get_admin_program_courses(major_key, batch_code):
    item = get_admin_curriculum(major_key, batch_code)
    return item.get("courses") if item and item.get("courses") else None

def get_curriculum_mapping_status(major_key, batch_code):
    item = get_admin_curriculum(major_key, batch_code)
    if item:
        return {"status": "EXACT", "item": item, "message": f"Đang dùng CTĐT {item.get('filename')}"}
    registry = load_curriculum_registry()
    same_major = [x for x in registry.values() if isinstance(x, dict) and x.get("active") and x.get("major")==major_key]
    if same_major:
        return {"status": "MISSING_EXACT", "item": None, "message": f"Không có CTĐT active khớp khóa {batch_code}; tuyệt đối không dùng CTĐT khóa khác."}
    return {"status": "NO_CURRICULUM", "item": None, "message": "Chưa có CTĐT PDF active cho ngành này."}

def build_curriculum_graph_ai_context(major_key, batch_code, passed_courses, failed_courses, earned_credits, max_chars=9000):
    courses = get_program_courses(major_key, batch_code)
    graph = CurriculumGraphEngine(courses)
    summary = graph.graph_summary(passed_courses, failed_courses)
    lines = [
        "--- CURRICULUM GRAPH ENGINE ---",
        f"NODES: {summary['nodes']} | PREREQUISITE EDGES: {summary['edges']} | UNKNOWN PREREQS: {summary['unknown_prereq_count']}",
        f"EARNED_CREDITS: {earned_credits}",
    ]
    if summary['failed_impact']:
        lines.append("FAILED COURSE IMPACT:")
        for item in summary['failed_impact'][:10]:
            blocked = ", ".join(c['code'] for c in item['blocked_courses'][:12]) or "Không có môn phụ thuộc trực tiếp/gián tiếp được xác nhận"
            lines.append(f"- {item['code']}: ảnh hưởng tối đa {item['blocked_count']} môn phía sau: {blocked}")
    lines.append("HIGH-IMPACT UNCOMPLETED COURSES:")
    for c in summary['critical_courses'][:10]:
        lines.append(f"- {c['code']} | {c.get('name','')} | downstream_impact={c['downstream_impact']}")
    text='\n'.join(lines)
    return text[:max_chars]

def build_curriculum_ai_context(major_key, batch_code, max_chars=26000):
    mapping = get_curriculum_mapping_status(major_key, batch_code)
    item = mapping.get("item")
    if not item:
        return (
            f"TRẠNG THÁI GẮN CTĐT: {mapping['status']}. {mapping['message']}\n"
            "AI không được khẳng định các quy định CTĐT đặc thù của khóa này nếu chưa có PDF khớp chính xác."
        )

    meta = (
        f"CURRICULUM_ID: {item.get('curriculum_id')}\n"
        f"FILE: {item.get('filename')}\n"
        f"NGÀNH: {item.get('major')} | KHÓA: {item.get('batch')} | BIẾN THỂ: {item.get('variant')}\n"
        f"QUYẾT ĐỊNH: {item.get('decision_number') or '-'} | NGÀY: {item.get('decision_date') or '-'}\n"
        f"PARSER: {item.get('parser_mode')} | CONFIDENCE: {item.get('parse_confidence')}\n"
        f"TỔNG QUAN: {item.get('totals') or '-'}\n"
    )
    structured = item.get("curriculum_text") or ""
    body = structured[:max_chars]
    return meta + "\nDANH MỤC HỌC PHẦN ĐÃ CHUẨN HÓA:\n" + body

try:
    from src.models.transformer_recsys import StudentTransformerRecSys
except ImportError:
    try:
        from models.transformer_recsys import StudentTransformerRecSys
    except ImportError:
        StudentTransformerRecSys = None

# -----------------------------------------------------------------------------
# 1B. TÍCH HỢP PERSONALIZED RECOMMENDATION + XAI
# -----------------------------------------------------------------------------
try:
    from src.recommendation.schemas import Course, StudentState
    from src.recommendation.personalized_pathway import PersonalizedPathway
    from src.recommendation.transformer_recommender import TransformerRanker
    from src.recommendation.run_personalized_recommendation import explain_recommendations
    HAS_PERSONALIZED_RECSYS = True
except ImportError:
    Course = StudentState = PersonalizedPathway = TransformerRanker = None
    explain_recommendations = None
    HAS_PERSONALIZED_RECSYS = False

try:
    from src.recommendation.vector_recommendation_service import VectorRecommendationService
    HAS_VECTOR_RECSYS = True
except ImportError:
    VectorRecommendationService = None
    HAS_VECTOR_RECSYS = False


# -----------------------------------------------------------------------------
# 2. CẤU HÌNH TRANG STREAMLIT & BỘ NHẬN DIỆN THƯƠNG HIỆU CTU
# -----------------------------------------------------------------------------

def _course_semester_value(course):
    raw = str(course.get("semester_hint", "") or "").strip()
    nums = re.findall(r"\d+", raw)
    return int(nums[0]) if nums else 999


def _normalize_course_codes(values):
    return {str(x).strip().upper() for x in (values or []) if str(x).strip()}


def _normalize_text_for_matching(value):
    """Chuẩn hóa chữ có/không dấu để nhận diện nhóm học phần đặc biệt ổn định."""
    s = unicodedata.normalize("NFD", str(value or "")).lower()
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _special_pathway_kind(course):
    """
    Nhận diện các học phần cần tách khỏi nhau trong mô phỏng:
    - Niên luận cơ sở
    - Niên luận ngành
    - Thực tập
    - Luận văn
    """
    name = _normalize_text_for_matching(course.get("name", ""))
    ctype = _normalize_text_for_matching(course.get("type", ""))
    combined = f"{name} {ctype}"

    if "luan van" in combined:
        return "LUAN_VAN"
    if "thuc tap" in combined:
        return "THUC_TAP"
    if "nien luan co so" in combined:
        return "NIEN_LUAN_CO_SO"
    if "nien luan nganh" in combined:
        return "NIEN_LUAN_NGANH"

    # Một số CTĐT ghi ngắn là "Niên luận" / "Đồ án - Niên luận".
    if "nien luan" in combined:
        return "NIEN_LUAN"

    return ""


def _is_special_pathway_course(course):
    return bool(_special_pathway_kind(course))


def build_optimal_learning_path(major_key, batch_code, passed_courses, failed_courses,
                                earned_credits, max_credits_per_term=18,
                                min_credits_per_term=12, target_code=None,
                                max_terms=8, include_electives=True):
    """
    Lập lộ trình theo CTĐT + prerequisite + giới hạn tín chỉ.

    Quy tắc đặc biệt:
    1. Niên luận cơ sở, Niên luận ngành, Thực tập và Luận văn không được
       xếp cùng một học kỳ.
    2. Nếu một học phần thuộc nhóm đặc biệt còn tồn sau số học kỳ mô phỏng
       yêu cầu, hệ thống tự mở thêm các học kỳ kế tiếp để xếp học phần đó.
    3. Học phần đặc biệt có thể vượt "Tối đa TC / học kỳ" khi cần thiết,
       ví dụ Luận văn 15 TC nhưng max = 14 TC. Trường hợp này được đánh dấu
       "ĐẶC BIỆT" và chỉ dùng cho học phần thuộc nhóm trên.
    """
    courses = get_program_courses(major_key, batch_code) or []
    graph = CurriculumGraphEngine(courses)
    by_code = graph.courses
    passed = _normalize_course_codes(passed_courses)
    failed = _normalize_course_codes(failed_courses)
    earned = float(earned_credits or 0)

    target = str(target_code or "").strip().upper()
    target_chain = set()
    if target and target in by_code:
        target_chain = graph.ancestors(target) | {target}

    remaining = {c for c in by_code if c not in passed}
    terms = []
    simulated_passed = set(passed)
    simulated_credits = earned
    diagnostics = []

    def get_eligible_candidates():
        candidates = []
        for code in sorted(remaining):
            c = by_code[code]
            if code in failed:
                priority_kind = 0
            elif target_chain and code in target_chain:
                priority_kind = 1
            elif c.get("type") == "Bắt buộc":
                priority_kind = 2
            elif include_electives and c.get("type") == "Tự chọn":
                priority_kind = 4
            else:
                priority_kind = 3

            check = graph.eligibility(c, simulated_passed, simulated_credits)
            if not check["eligible"]:
                continue

            impact = len(graph.descendants(code))
            sem = _course_semester_value(c)
            special_kind = _special_pathway_kind(c)
            candidates.append((priority_kind, -impact, sem, code, c, check, special_kind))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return candidates

    # ------------------------------------------------------------------
    # Giai đoạn 1: xếp đúng số học kỳ mà người dùng yêu cầu.
    # ------------------------------------------------------------------
    for term_no in range(1, max(1, int(max_terms)) + 1):
        candidates = get_eligible_candidates()

        chosen, used = [], 0.0
        special_selected = False

        for _, _, _, code, c, _, special_kind in candidates:
            cr = float(c.get("credits", 0) or 0)
            if cr <= 0:
                continue

            # Nếu kỳ này đã có một học phần đặc biệt thì không được thêm
            # Niên luận/Thực tập/Luận văn thứ hai vào cùng học kỳ.
            if special_selected and special_kind:
                continue

            # Học phần đặc biệt vẫn phải tuân theo max TC khi có thể.
            # Nếu không đủ chỗ, ta để lại cho giai đoạn overflow.
            if used + cr <= float(max_credits_per_term):
                chosen.append(dict(c))
                used += cr
                if special_kind:
                    special_selected = True

        if not chosen:
            blocked = []
            for code in sorted(remaining):
                c = by_code[code]
                chk = graph.eligibility(c, simulated_passed, simulated_credits)
                if not chk["eligible"]:
                    blocked.append({
                        "code": code,
                        "name": c.get("name", ""),
                        "reason": chk.get("reason", "")
                    })

            diagnostics.append({
                "term": term_no,
                "status": "BLOCKED",
                "blocked": blocked[:30]
            })
            break

        term_status = "OK" if used >= float(min_credits_per_term) else "LIGHT"
        terms.append({
            "term": term_no,
            "credits": round(used, 1),
            "status": term_status,
            "courses": chosen,
            "target_progress": bool(target and target in {x.get("code") for x in chosen}),
        })

        new_codes = {str(c.get("code", "")).upper() for c in chosen}
        simulated_passed |= new_codes
        simulated_credits += used
        remaining -= new_codes
        failed -= new_codes

        if target and target in simulated_passed:
            diagnostics.append({"term": term_no, "status": "TARGET_REACHED", "target": target})

        if not remaining:
            break

    # ------------------------------------------------------------------
    # Giai đoạn 2: FORCE OVERFLOW cho học phần đặc biệt.
    #
    # Ví dụ max_terms = 3, max_TC = 14 và Luận văn = 15 TC:
    # -> nếu còn Luận văn đủ điều kiện sau HK3, tự tạo HK4 và xếp Luận văn.
    # Mỗi HK overflow chỉ chứa tối đa 1 học phần đặc biệt.
    # ------------------------------------------------------------------
    overflow_start_term = len(terms) + 1
    overflow_guard = 0
    max_overflow_terms = max(8, len(remaining) + 2)

    while remaining and overflow_guard < max_overflow_terms:
        overflow_guard += 1

        special_candidates = []
        for _, _, _, code, c, check, special_kind in get_eligible_candidates():
            if not special_kind:
                continue
            cr = float(c.get("credits", 0) or 0)
            if cr <= 0:
                continue
            special_candidates.append((code, c, check, special_kind, cr))

        if not special_candidates:
            break

        # Ưu tiên học phần đặc biệt có ảnh hưởng downstream lớn / đã rớt /
        # xuất hiện sớm trong CTĐT.
        special_candidates.sort(
            key=lambda x: (
                0 if x[0] in failed else 1,
                -len(graph.descendants(x[0])),
                _course_semester_value(x[1]),
                x[0],
            )
        )

        code, c, check, special_kind, cr = special_candidates[0]

        # Mỗi kỳ overflow chỉ xếp đúng 1 học phần đặc biệt.
        # Cho phép vượt max_credits_per_term nếu bản thân học phần vượt trần.
        overflow_status = "SPECIAL_OVERFLOW"
        if cr <= float(max_credits_per_term):
            note = (
                f"Được chuyển sang HK {overflow_start_term} để không xếp chung "
                "với học phần Niên luận/Thực tập/Luận văn khác."
            )
        else:
            note = (
                f"Học phần {code} có {cr:g} TC vượt giới hạn {float(max_credits_per_term):g} TC/HK; "
                "hệ thống vẫn xếp riêng ở học kỳ tiếp theo theo quy tắc học phần đặc biệt."
            )
            diagnostics.append({
                "term": overflow_start_term,
                "status": "SPECIAL_CREDIT_OVERRIDE",
                "course": code,
                "credits": cr,
                "max_credits_per_term": max_credits_per_term,
                "reason": "Học phần đặc biệt được phép vượt trần TC mô phỏng để không bị bỏ khỏi lộ trình."
            })

        terms.append({
            "term": overflow_start_term,
            "credits": round(cr, 1),
            "status": overflow_status,
            "courses": [dict(c)],
            "target_progress": bool(target and target == code),
            "note": note,
        })

        simulated_passed.add(code)
        simulated_credits += cr
        remaining.remove(code)
        failed.discard(code)

        diagnostics.append({
            "term": overflow_start_term,
            "status": "SPECIAL_OVERFLOW_SCHEDULED",
            "course": code,
            "kind": special_kind,
            "credits": cr,
            "note": note,
        })

        overflow_start_term += 1

    # Các học phần còn lại sau overflow vẫn giữ trạng thái chưa xếp nếu
    # bị chặn bởi prerequisite/điều kiện dữ liệu.
    unscheduled = [dict(by_code[c]) for c in sorted(remaining) if c in by_code]

    return {
        "status": "OK" if terms else "BLOCKED",
        "curriculum_major": major_key,
        "curriculum_batch": batch_code,
        "target": target or None,
        "start_credits": earned,
        "end_credits_simulated": round(simulated_credits, 1),
        "max_credits_per_term": max_credits_per_term,
        "terms": terms,
        "unscheduled": unscheduled,
        "diagnostics": diagnostics,
        "graph_nodes": len(by_code),
        "graph_edges": sum(len(v) for v in graph.forward.values()),
    }



def optimal_path_to_dataframe(plan):
    rows = []
    for term in plan.get("terms", []):
        for c in term.get("courses", []):
            rows.append({
                "Học kỳ mô phỏng": term.get("term"),
                "Trạng thái kỳ": term.get("status"),
                "Mã HP": c.get("code"),
                "Tên học phần": c.get("name"),
                "TC": c.get("credits"),
                "Loại": c.get("type"),
                "Nhóm": c.get("group", ""),
                "Tiên quyết": ", ".join(c.get("prereq") or []),
                "Điều kiện TC": c.get("min_credits", 0),
            })
    return pd.DataFrame(rows)


def build_optimal_path_ai_context(major_key, batch_code, passed_courses, failed_courses,
                                  earned_credits, target_code=None, max_chars=6000):
    plan = build_optimal_learning_path(
        major_key, batch_code, passed_courses, failed_courses, earned_credits,
        max_credits_per_term=18, min_credits_per_term=12,
        target_code=target_code, max_terms=6,
    )
    lines = [
        "BỘ MÁY MÔ PHỎNG LỘ TRÌNH HỌC TẬP TỐI ƯU (xác định, nhận thức CTĐT)",
        f"CTĐT: {major_key} | {batch_code}",
        f"Target: {plan.get('target') or 'Tối ưu tiến độ chung'}",
        f"Start credits: {plan.get('start_credits')}",
    ]
    for term in plan.get("terms", []):
        courses = term.get("courses", [])
        items = "; ".join(f"{c.get('code')} ({c.get('credits')}TC)" for c in courses)
        lines.append(f"TERM {term.get('term')} [{term.get('credits')} TC, {term.get('status')}]: {items}")
    if plan.get("unscheduled"):
        lines.append("UNSCHEDULED: " + ", ".join(c.get("code", "") for c in plan["unscheduled"][:20]))
    if plan.get("diagnostics"):
        lines.append("DIAGNOSTICS: " + json.dumps(plan["diagnostics"][:5], ensure_ascii=False))
    return "\n".join(lines)[:max_chars]



def get_curriculum_credit_target(major_key, batch_code):
    item = get_admin_curriculum(major_key, batch_code)
    totals = (item or {}).get("totals") or {}
    target = totals.get("total_credits")
    try:
        return float(target) if target is not None else None, item
    except (TypeError, ValueError):
        return None, item


def build_graduation_progress(major_key, batch_code, passed_courses, failed_courses, earned_credits):
    courses = get_program_courses(major_key, batch_code) or []
    passed = _normalize_course_codes(passed_courses)
    failed = _normalize_course_codes(failed_courses)
    target, item = get_curriculum_credit_target(major_key, batch_code)
    by_code = {str(c.get("code", "")).upper(): c for c in courses if c.get("code")}
    remaining = [c for code, c in by_code.items() if code not in passed]
    mandatory_remaining = [c for c in remaining if str(c.get("type", "")) == "Bắt buộc"]
    elective_remaining = [c for c in remaining if str(c.get("type", "")) == "Tự chọn"]
    remaining_credits = sum(float(c.get("credits", 0) or 0) for c in remaining)
    mandatory_credits = sum(float(c.get("credits", 0) or 0) for c in mandatory_remaining)
    target_gap = max(0.0, float(target) - float(earned_credits)) if target is not None else None
    progress = (min(100.0, 100.0 * float(earned_credits) / float(target)) if target else None)
    return {
        "curriculum_found": bool(item), "target_credits": target,
        "earned_credits": float(earned_credits or 0), "credit_gap": target_gap,
        "progress_percent": progress, "course_count": len(by_code),
        "passed_count": len([x for x in passed if x in by_code]),
        "failed_count": len([x for x in failed if x in by_code]),
        "remaining_course_count": len(remaining),
        "remaining_listed_credits": round(remaining_credits, 1),
        "mandatory_remaining_credits": round(mandatory_credits, 1),
        "elective_remaining_course_count": len(elective_remaining),
        "note": ("Tiến độ tín chỉ dựa trên tổng TC được trích xuất từ PDF active. Điều kiện tốt nghiệp khác "
                 "(nhóm tự chọn, chuẩn đầu ra, chứng chỉ...) chỉ được kết luận khi có dữ liệu nguồn tương ứng.")
    }


def simulate_learning_scenarios(major_key, batch_code, passed_courses, failed_courses,
                                earned_credits, max_terms=8, target_code=None,
                                load_options=(15, 18, 21), shock_failed_course=None):
    scenarios = []
    base_progress = build_graduation_progress(major_key, batch_code, passed_courses, failed_courses, earned_credits)
    for load in load_options:
        plan = build_optimal_learning_path(
            major_key, batch_code, passed_courses, failed_courses, earned_credits,
            max_credits_per_term=int(load), min_credits_per_term=0,
            target_code=target_code, max_terms=max_terms
        )
        scheduled = sum(len(t.get("courses", [])) for t in plan.get("terms", []))
        scenarios.append({
            "scenario": f"{load} TC/HK", "max_credits_per_term": int(load),
            "terms_used": len(plan.get("terms", [])),
            "scheduled_courses": scheduled,
            "end_credits": plan.get("end_credits_simulated", earned_credits),
            "remaining_courses": len(plan.get("unscheduled", [])),
            "status": plan.get("status"), "plan": plan,
        })

    shock = str(shock_failed_course or "").strip().upper()
    if shock:
        shock_failed = set(_normalize_course_codes(failed_courses)) | {shock}
        shock_passed = set(_normalize_course_codes(passed_courses)) - {shock}
        plan = build_optimal_learning_path(
            major_key, batch_code, list(shock_passed), list(shock_failed), earned_credits,
            max_credits_per_term=18, min_credits_per_term=0,
            target_code=target_code, max_terms=max_terms
        )
        graph = CurriculumGraphEngine(get_program_courses(major_key, batch_code) or [])
        impacted = graph.descendants(shock) if shock in graph.courses else set()
        scenarios.append({
            "scenario": f"Kịch bản rớt {shock}", "max_credits_per_term": 18,
            "terms_used": len(plan.get("terms", [])),
            "scheduled_courses": sum(len(t.get("courses", [])) for t in plan.get("terms", [])),
            "end_credits": plan.get("end_credits_simulated", earned_credits),
            "remaining_courses": len(plan.get("unscheduled", [])),
            "status": plan.get("status"), "impact_count": len(impacted),
            "impacted_courses": sorted(impacted), "plan": plan,
        })
    return {"progress": base_progress, "scenarios": scenarios}


def scenario_results_to_dataframe(result):
    rows=[]
    for x in result.get("scenarios", []):
        rows.append({
            "Kịch bản": x.get("scenario"), "Tải tối đa": x.get("max_credits_per_term"),
            "HK mô phỏng": x.get("terms_used"), "HP đã xếp": x.get("scheduled_courses"),
            "TC sau mô phỏng": x.get("end_credits"), "HP chưa xếp": x.get("remaining_courses"),
            "Tác động dây chuyền": x.get("impact_count", 0), "Trạng thái": x.get("status")
        })
    return pd.DataFrame(rows)


def build_graduation_scenario_ai_context(major_key, batch_code, passed_courses, failed_courses,
                                         earned_credits, target_code=None, max_chars=5000):
    result = simulate_learning_scenarios(
        major_key, batch_code, passed_courses, failed_courses, earned_credits,
        max_terms=8, target_code=target_code
    )
    p = result["progress"]
    lines=["--- GRADUATION PROGRESS & SCENARIO ENGINE ---",
           f"CURRICULUM: {major_key} | {batch_code}",
           f"EARNED_CREDITS: {p['earned_credits']} | TARGET_CREDITS: {p['target_credits']}",
           f"PROGRESS_PERCENT: {p['progress_percent']} | CREDIT_GAP: {p['credit_gap']}",
           p['note'], "SCENARIOS:"]
    for x in result["scenarios"]:
        lines.append(f"- {x['scenario']}: terms={x['terms_used']}, end_credits={x['end_credits']}, unscheduled={x['remaining_courses']}, status={x['status']}")
    return "\n".join(lines)[:max_chars]


# -----------------------------------------------------------------------------
# 2E. COURSE OFFERING & SEMESTER REALITY ENGINE
# -----------------------------------------------------------------------------
def _term_key(value):
    return str(value or "").strip().upper()

def load_course_offerings():
    data = _load_json(OFFERING_FILE, {"schema_version": 1, "offerings": []})
    if isinstance(data, list):
        data = {"schema_version": 1, "offerings": data}
    data.setdefault("offerings", [])
    return data

def save_course_offerings(data):
    data = data if isinstance(data, dict) else {"offerings": []}
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_json(OFFERING_FILE, data)

def normalize_offering(item):
    return {
        "major": str(item.get("major", "*") or "*").strip().upper(),
        "batch": str(item.get("batch", "*") or "*").strip().upper(),
        "term": _term_key(item.get("term")),
        "course_code": str(item.get("course_code", "") or "").strip().upper(),
        "status": str(item.get("status", "OPEN") or "OPEN").strip().upper(),
        "source": str(item.get("source", "ADMIN") or "ADMIN").strip(),
        "capacity": item.get("capacity"),
        "note": str(item.get("note", "") or "").strip(),
        "updated_at": item.get("updated_at") or datetime.now().isoformat(timespec="seconds"),
    }

def get_offering_index(major_key, batch_code, term):
    major, batch, term = str(major_key).upper(), str(batch_code).upper(), _term_key(term)
    exact, fallback = {}, {}
    for raw in load_course_offerings().get("offerings", []):
        x = normalize_offering(raw)
        if x["term"] != term or not x["course_code"]:
            continue
        if x["major"] in {"*", major} and x["batch"] in {"*", batch}:
            target = exact if (x["major"] == major and x["batch"] == batch) else fallback
            target[x["course_code"]] = x
    return {**fallback, **exact}

def offering_status(course_code, major_key, batch_code, term):
    code = str(course_code).upper()
    idx = get_offering_index(major_key, batch_code, term)
    x = idx.get(code)
    if x is None:
        return {"status": "UNKNOWN", "source": "NO_REALITY_DATA", "course_code": code}
    return x

def course_is_offered_now(course_code, major_key, batch_code, term, unknown_policy="ALLOW_WITH_WARNING"):
    info = offering_status(course_code, major_key, batch_code, term)
    status = info.get("status", "UNKNOWN")
    if status in {"OPEN", "AVAILABLE", "TENTATIVE"}:
        return True, info
    if status == "UNKNOWN" and unknown_policy == "ALLOW_WITH_WARNING":
        return True, info
    return False, info

def build_semester_reality_plan(major_key, batch_code, passed_courses, failed_courses, earned_credits,
                                term, max_credits_per_term=18, min_credits_per_term=0,
                                target_code=None, unknown_policy="ALLOW_WITH_WARNING"):
    base = build_optimal_learning_path(
        major_key, batch_code, passed_courses, failed_courses, earned_credits,
        max_credits_per_term=max_credits_per_term,
        min_credits_per_term=min_credits_per_term,
        target_code=target_code, max_terms=1
    )
    term_courses = (base.get("terms") or [{}])[0].get("courses", []) if base.get("terms") else []
    feasible, blocked, unknown = [], [], []
    for c in term_courses:
        ok, info = course_is_offered_now(c.get("code"), major_key, batch_code, term, unknown_policy)
        row = dict(c)
        row["offering_status"] = info.get("status")
        row["offering_source"] = info.get("source")
        row["offering_note"] = info.get("note", "")
        if ok:
            feasible.append(row)
            if info.get("status") == "UNKNOWN":
                unknown.append(row)
        else:
            blocked.append(row)
    return {
        "term": _term_key(term), "curriculum_plan": base, "feasible_courses": feasible,
        "blocked_by_offering": blocked, "unknown_reality": unknown,
        "credits": round(sum(float(c.get("credits", 0) or 0) for c in feasible), 1),
        "note": "Dữ liệu mở lớp chỉ có giá trị theo nguồn đã nhập/đồng bộ. UNKNOWN không đồng nghĩa chắc chắn có lớp."
    }

def semester_reality_to_dataframe(result):
    rows=[]
    for c in result.get("feasible_courses", []):
        rows.append({"Mã HP": c.get("code"), "Tên học phần": c.get("name"), "TC": c.get("credits"), "Trạng thái mở lớp": c.get("offering_status"), "Nguồn": c.get("offering_source"), "Ghi chú": c.get("offering_note", "")})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2F. REGISTRATION CONFLICT & SCHEDULE OPTIMIZATION ENGINE
# -----------------------------------------------------------------------------
SECTION_OFFERING_FILE = ADMIN_DATA_DIR / "course_sections.json"

def load_course_sections():
    data = _load_json(SECTION_OFFERING_FILE, {"schema_version": 1, "sections": []})
    if isinstance(data, list): data = {"schema_version": 1, "sections": data}
    data.setdefault("sections", [])
    return data

def save_course_sections(data):
    data = data if isinstance(data, dict) else {"sections": []}
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_json(SECTION_OFFERING_FILE, data)

def _to_minutes(v):
    try:
        h,m=str(v).strip().split(":")[:2]
        return int(h)*60+int(m)
    except Exception: return None

def normalize_section(raw):
    r = dict(raw or {})
    sched = r.get("schedule", [])
    if isinstance(sched, str):
        try: sched = json.loads(sched)
        except Exception: sched=[]
    if not isinstance(sched, list): sched=[]
    clean=[]
    for x in sched:
        if not isinstance(x, dict): continue
        day=x.get("day")
        try: day=int(day)
        except Exception: pass
        start=str(x.get("start","")).strip(); end=str(x.get("end","")).strip()
        if start and end: clean.append({"day":day,"start":start,"end":end,"room":str(x.get("room","")).strip()})
    return {"major":str(r.get("major","*") or "*").upper().strip(),"batch":str(r.get("batch","*") or "*").upper().strip(),
            "term":_term_key(r.get("term")),"course_code":str(r.get("course_code","") or "").upper().strip(),
            "section":str(r.get("section","") or "").strip(),"status":str(r.get("status","OPEN") or "OPEN").upper().strip(),
            "capacity":r.get("capacity"),"enrolled":r.get("enrolled"),"schedule":clean,
            "source":str(r.get("source","ADMIN") or "ADMIN").strip(),"note":str(r.get("note","") or "").strip()}

def get_sections_for_term(major_key,batch_code,term):
    major,batch,term=str(major_key).upper(),str(batch_code).upper(),_term_key(term)
    out=[]
    for raw in load_course_sections().get("sections",[]):
        x=normalize_section(raw)
        if x["term"]==term and x["course_code"] and x["status"] in {"OPEN","AVAILABLE","TENTATIVE"} and x["major"] in {"*",major} and x["batch"] in {"*",batch}: out.append(x)
    return out

def sections_conflict(a,b):
    for x in a.get("schedule",[]):
        for y in b.get("schedule",[]):
            if str(x.get("day"))!=str(y.get("day")): continue
            xs,xe,ys,ye=map(_to_minutes,[x.get("start"),x.get("end"),y.get("start"),y.get("end")])
            if None not in (xs,xe,ys,ye) and max(xs,ys)<min(xe,ye): return True
    return False

def section_priority(section, course_meta, failed_codes, impact):
    code=section.get("course_code",""); c=course_meta.get(code,{})
    score=0
    if code in set(failed_codes): score+=100
    score+=float(impact.get(code,0))*5
    if str(c.get("type","")).lower().startswith("bắt buộc"): score+=30
    cap=section.get("capacity"); enr=section.get("enrolled")
    try:
        if cap and enr and float(enr)/float(cap)>=0.85: score+=10
    except Exception: pass
    return score

def optimize_registration_plans(major_key,batch_code,passed_courses,failed_courses,earned_credits,term,max_credits=18,target_code=None,top_k=3):
    reality=build_semester_reality_plan(major_key,batch_code,passed_courses,failed_courses,earned_credits,term,max_credits_per_term=max_credits,target_code=target_code,unknown_policy="BLOCK_UNKNOWN")
    eligible={str(c.get("code")).upper():c for c in reality.get("feasible_courses",[]) if c.get("code")}
    courses={str(c.get("code")).upper():c for c in get_program_courses(major_key,batch_code) if c.get("code")}
    sections=[x for x in get_sections_for_term(major_key,batch_code,term) if x.get("course_code") in eligible]
    by_course={}
    for x in sections: by_course.setdefault(x["course_code"],[]).append(x)
    graph = build_curriculum_graph(major_key, batch_code)
    impact = {
        code: len(graph.descendants(code)) if code in graph.courses else 0
        for code in courses
    }
    choices=[]
    for code, opts in by_course.items():
        opts=sorted(opts,key=lambda z:section_priority(z,courses,failed_courses,impact),reverse=True)
        choices.append((code,opts[:5]))
    choices=sorted(choices,key=lambda z:max(section_priority(o,courses,failed_courses,impact) for o in z[1]),reverse=True)[:20]
    candidates=[]
    def dfs(i,selected,credits,score):
        if credits>max_credits+1e-9: return
        if i>=len(choices):
            if selected: candidates.append((score,list(selected),credits))
            return
        dfs(i+1,selected,credits,score)
        code,opts=choices[i]; cr=float(courses.get(code,{}).get("credits",eligible.get(code,{}).get("credits",0)) or 0)
        for sec in opts:
            if credits+cr>max_credits+1e-9: continue
            if any(sections_conflict(sec,z) for z in selected): continue
            selected.append(sec); dfs(i+1,selected,credits+cr,score+section_priority(sec,courses,failed_courses,impact)); selected.pop()
    dfs(0,[],0,0)
    candidates.sort(key=lambda x:(x[0],x[2]),reverse=True)
    seen=set(); plans=[]
    for score,sel,cr in candidates:
        sig=tuple(sorted((x["course_code"],x.get("section","")) for x in sel))
        if sig in seen: continue
        seen.add(sig)
        plans.append({"score":round(score,2),"credits":round(cr,1),"sections":sel,"conflicts":0})
        if len(plans)>=top_k: break
    return {"term":_term_key(term),"status":"CONFIRMED" if plans else "NO_FEASIBLE_SCHEDULE","plans":plans,"eligible_courses":list(eligible),"section_count":len(sections),"note":"Kết quả chỉ tối ưu trên dữ liệu lớp, lịch và điều kiện hiện có; không thay thế xác nhận đăng ký chính thức."}

def registration_plans_to_dataframe(result):
    rows=[]
    for i,p in enumerate(result.get("plans",[]),1):
        for s in p.get("sections",[]):
            schedule="; ".join(f"T{s.get('day')} {s.get('start')}-{s.get('end')} {s.get('room','')}" for s in s.get("schedule",[]))
            rows.append({"Phương án":i,"Mã HP":s.get("course_code"),"Lớp":s.get("section"),"Lịch":schedule,"Điểm":p.get("score"),"Tổng TC":p.get("credits")})
    return pd.DataFrame(rows)

def build_registration_ai_context(major_key,batch_code,passed_courses,failed_courses,earned_credits,term,max_chars=2500):
    r=optimize_registration_plans(major_key,batch_code,passed_courses,failed_courses,earned_credits,term)
    lines=["--- REGISTRATION CONFLICT & SCHEDULE OPTIMIZATION ENGINE ---",f"TERM: {r['term']}",f"STATUS: {r['status']}",f"SECTION_COUNT: {r['section_count']}"]
    for i,p in enumerate(r.get("plans",[]),1): lines.append(f"PLAN {i}: credits={p['credits']}, score={p['score']}, conflicts={p['conflicts']}")
    return "\n".join(lines)[:max_chars]

def build_semester_reality_ai_context(major_key, batch_code, passed_courses, failed_courses, earned_credits,
                                      term="CURRENT", max_chars=3500):
    result = build_semester_reality_plan(major_key, batch_code, passed_courses, failed_courses,
                                         earned_credits, term=term)
    lines=["--- COURSE OFFERING & SEMESTER REALITY ENGINE ---",
           f"CURRICULUM: {major_key}|{batch_code}", f"TERM: {result['term']}",
           f"FEASIBLE_CREDITS: {result['credits']}", "FEASIBLE COURSES:"]
    for c in result.get("feasible_courses", [])[:40]:
        lines.append(f"- {c.get('code')} | {c.get('name')} | {c.get('credits')} TC | OFFERING={c.get('offering_status')}")
    if result.get("blocked_by_offering"):
        lines.append("BLOCKED BY OFFERING: " + ", ".join(c.get("code", "") for c in result["blocked_by_offering"][:40]))
    if result.get("unknown_reality"):
        lines.append("UNKNOWN REALITY (must not claim definitely open): " + ", ".join(c.get("code", "") for c in result["unknown_reality"][:40]))
    lines.append(result["note"])
    return "\n".join(lines)[:max_chars]

# Biểu tượng tab trình duyệt: Logo Đại học Cần Thơ
# Ưu tiên đường dẫn Windows trong project, sau đó dùng đường dẫn tương đối.
PAGE_ICON_PATH = Path(r"D:\StudentProgressAI\data\pics\LogoCTU.png")
if not PAGE_ICON_PATH.exists():
    PAGE_ICON_PATH = PROJECT_ROOT / "data" / "pics" / "LogoCTU.png"

st.set_page_config(
    page_title="StudentProgressAI - ĐH Cần Thơ",
    page_icon=str(PAGE_ICON_PATH) if PAGE_ICON_PATH.exists() else "🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS tương thích cả LIGHT MODE lẫn DARK MODE
ctu_adaptive_theme_css = r"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=K2D:wght@400;600;700;800&family=Readex+Pro:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Readex Pro', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'K2D', sans-serif !important;
        font-weight: 700 !important;
    }

    @media (prefers-color-scheme: light) {
        h1, h2, h3, h4, h5, h6 {
            color: #1f5ca9 !important;
        }
        .ctu-card {
            background-color: #f8fafc;
            color: #0f172a;
            border-left: 5px solid #1f5ca9;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
    }

    @media (prefers-color-scheme: dark) {
        h1, h2, h3, h4, h5, h6 {
            color: #38bdf8 !important;
        }
        .ctu-card {
            background-color: #1e293b;
            color: #f8fafc;
            border-left: 5px solid #00afef;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
    }

    .ctu-header {
        background: linear-gradient(135deg, #1f5ca9 0%, #00afef 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(31, 92, 169, 0.25);
    }

    .ctu-header h1 {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        margin: 0;
        font-weight: 800 !important;
        text-transform: uppercase;
    }

    .ctu-header p {
        color: #e0f2fe !important;
        margin-top: 0.3rem;
        margin-bottom: 0;
        font-size: 0.95rem;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'K2D', sans-serif !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }

    .stButton > button {
        background-color: #1f5ca9 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #00afef !important;
        color: white !important;
        transform: translateY(-1px);
    }

    .ctu-topbar{background:#1f5ca9;color:white;padding:.55rem 1rem;margin:-1rem -1rem 1rem;font-size:.84rem;display:flex;align-items:center;justify-content:space-between;border-radius:4px;}
    .ctu-brandbar{display:flex;align-items:center;gap:14px;padding:.8rem 0 1rem;border-bottom:3px solid #00afef;margin-bottom:1rem}
    .ctu-mark{width:52px;height:52px;border:3px solid #1f5ca9;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;color:#1f5ca9}
    .ctu-brand-title{font-size:1.05rem;font-weight:800;color:#1f5ca9;margin:0}
    .ctu-brand-sub{font-size:.76rem;color:#64748b;margin:0}
    /* ===== Split-screen login ===== */
    .login-page{
        width:100%;
        min-height:calc(100vh - 70px);
        padding:18px 0 32px;
        box-sizing:border-box;
    }
    .login-layout{
        width:min(1180px, 100%);
        margin:0 auto;
    }
    .login-visual{
        position:relative;
        min-height:680px;
        height:680px;
        overflow:hidden;
        border-radius:28px;
        background:#0a4b94;
        box-shadow:0 22px 55px rgba(15,23,42,.14);
        border:1px solid rgba(148,163,184,.25);
    }
    .login-visual > div{
        white-space:normal;
    }
    .login-visual-media,
    .login-visual-overlay{
        position:absolute;
        inset:0;
    }
    .login-visual-media{
        overflow:hidden;
        background:#0a4b94;
    }

    /* Ảnh CTU chính: phủ toàn bộ khung, hơi chìm để hòa với nền xanh. */
    .login-hero-image{
        position:absolute;
        inset:0;
        z-index:1;
        width:100%;
        height:100%;
        display:block;
        object-fit:cover;
        object-position:center center;
        opacity:.56;
        filter:saturate(.92) brightness(.82) contrast(1.02);
        transform:scale(1.08);
        transform-origin:center center;
        /* Mép ảnh tan dần; phần trong suốt sẽ lộ lớp blur phía dưới. */
        -webkit-mask-image:radial-gradient(ellipse 94% 92% at 50% 50%,
            #000 55%,
            rgba(0,0,0,.98) 68%,
            rgba(0,0,0,.72) 83%,
            rgba(0,0,0,.28) 94%,
            transparent 100%);
        mask-image:radial-gradient(ellipse 94% 92% at 50% 50%,
            #000 55%,
            rgba(0,0,0,.98) 68%,
            rgba(0,0,0,.72) 83%,
            rgba(0,0,0,.28) 94%,
            transparent 100%);
    }

    /* Bản sao blur làm nền đệm ở viền, tránh xuất hiện mép cắt cứng. */
    .login-hero-image-blur{
        position:absolute;
        inset:-32px;
        z-index:0;
        width:calc(100% + 64px);
        height:calc(100% + 64px);
        display:block;
        object-fit:cover;
        object-position:center center;
        opacity:.72;
        filter:blur(26px) saturate(.82) brightness(.72);
        transform:scale(1.05);
        transform-origin:center center;
    }

    /* Feather màu xanh quanh bốn phía, nhưng không che mất toàn bộ ảnh. */
    .login-visual-media::after{
        content:"";
        position:absolute;
        inset:0;
        z-index:2;
        pointer-events:none;
        background:
            linear-gradient(180deg,
                rgba(10,75,148,.78) 0%,
                rgba(10,75,148,.24) 11%,
                rgba(10,75,148,0) 24%,
                rgba(10,75,148,0) 76%,
                rgba(10,75,148,.26) 89%,
                rgba(10,75,148,.82) 100%),
            linear-gradient(90deg,
                rgba(10,75,148,.76) 0%,
                rgba(10,75,148,.18) 9%,
                rgba(10,75,148,0) 18%,
                rgba(10,75,148,0) 82%,
                rgba(10,75,148,.18) 91%,
                rgba(10,75,148,.78) 100%);
        filter:blur(4px);
        transform:scale(1.015);
    }

    .login-visual-overlay{
        z-index:3;
        background:
            linear-gradient(180deg,
                rgba(3,39,83,.18) 0%,
                rgba(5,52,109,.34) 38%,
                rgba(3,35,75,.74) 100%);
    }
    .login-visual-content{
        position:absolute;
        white-space:normal;
        word-break:normal;
        overflow-wrap:anywhere;
        left:42px;
        right:42px;
        top:38px;
        bottom:70px;
        z-index:4;
        color:#fff;
        display:flex;
        flex-direction:column;
        justify-content:flex-end;
    }
    .login-visual-badge{
        display:inline-flex;
        width:max-content;
        padding:7px 12px;
        border-radius:999px;
        background:rgba(255,255,255,.14);
        border:1px solid rgba(255,255,255,.30);
        backdrop-filter:blur(8px);
        font-size:.72rem;
        font-weight:800;
        letter-spacing:.08em;
        margin-bottom:15px;
    }
    .login-visual-logo-wrap{margin-bottom:12px;display:block}
    .login-mini-logo{
        width:78px;
        height:78px;
        border-radius:18px;
        background:rgba(255,255,255,.94);
        display:flex;
        align-items:center;
        justify-content:center;
        padding:7px;
        box-shadow:0 10px 28px rgba(0,0,0,.18);
        border:1px solid rgba(255,255,255,.70);
    }
    .login-mini-logo img{max-width:100%;max-height:100%;object-fit:contain}
    .login-logo-fallback{
        font-size:1rem;
        font-weight:900;
        color:#0a4b94;
    }
    .login-visual-kicker{
        font-size:.75rem;
        font-weight:800;
        letter-spacing:.12em;
        color:#dbeafe;
        margin-top:4px;
    }
    .login-visual-content h2{
        font-size:2.25rem!important;
        line-height:1.1;
        margin:.25rem 0 .65rem;
        color:#fff!important;
        letter-spacing:-.02em;
    }
    .login-visual-text{
        max-width:560px;
        font-size:.95rem;
        line-height:1.65;
        color:rgba(255,255,255,.91);
        margin:0 0 1.15rem;
    }
    .login-feature-list{
        display:grid;
        gap:.55rem;
        max-width:600px;
    }
    .login-feature{
        display:grid;
        grid-template-columns:42px 1fr;
        column-gap:.65rem;
        row-gap:.05rem;
        padding:.62rem .75rem;
        border-radius:14px;
        background:rgba(255,255,255,.09);
        border:1px solid rgba(255,255,255,.14);
        backdrop-filter:blur(8px);
    }
    .login-feature span{
        grid-row:1 / span 2;
        width:34px;
        height:34px;
        border-radius:10px;
        background:rgba(255,255,255,.16);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:.68rem;
        font-weight:800;
    }
    .login-feature b{font-size:.80rem;color:#fff}
    .login-feature small{font-size:.69rem;line-height:1.35;color:rgba(255,255,255,.75)}
    .login-visual-footer{
        position:absolute;
        left:42px;
        right:42px;
        bottom:24px;
        z-index:5;
        font-size:.67rem;
        color:rgba(255,255,255,.70);
        border-top:1px solid rgba(255,255,255,.17);
        padding-top:12px;
    }
    .login-hero-placeholder{
        position:absolute;
        inset:0;
        z-index:0;
        width:100%;
        height:100%;
        box-sizing:border-box;
        background:
            radial-gradient(circle at 75% 20%, rgba(0,175,239,.18), transparent 34%),
            linear-gradient(145deg,#0a4b94,#0e7490);
    }
    .login-panel-head{
        padding:22px 10px 12px;
        text-align:left;
    }
    .login-panel-head .login-mini-logo{
        width:58px;
        height:58px;
        margin-bottom:18px;
        background:#fff;
        border:1px solid #dbeafe;
        box-shadow:0 8px 22px rgba(10,75,148,.10);
    }
    .login-panel-kicker{
        font-size:.70rem;
        letter-spacing:.12em;
        font-weight:850;
        color:#2563eb;
        text-transform:uppercase;
        margin-bottom:.3rem;
    }
    .login-panel-head h1{
        font-size:2rem!important;
        line-height:1.12;
        margin:.25rem 0 .45rem;
        color:#0f2747!important;
        letter-spacing:-.025em;
    }
    .login-panel-head p{
        font-size:.87rem;
        line-height:1.6;
        color:#64748b;
        margin:0;
        max-width:420px;
    }
    .login-form-note{
        font-size:.70rem;
        line-height:1.45;
        color:#94a3b8;
        margin:.2rem 0 .85rem;
    }
    .login-security{
        display:flex;
        gap:.65rem;
        align-items:flex-start;
        margin:1rem 10px 0;
        padding:.75rem .85rem;
        border:1px solid #e2e8f0;
        border-radius:14px;
        background:#f8fafc;
        color:#475569;
    }
    .login-security > span{font-size:1rem}
    .login-security b{display:block;font-size:.74rem;color:#334155}
    .login-security small{display:block;font-size:.67rem;line-height:1.45;color:#94a3b8;margin-top:.14rem}

    /* Login form only: same visual language, light and spacious. */
    body:has(.login-page-marker) div[data-testid="stForm"]{
        background:#fff;
        border:1px solid #e2e8f0;
        border-radius:20px;
        padding:1.15rem 1.1rem 1.05rem;
        box-shadow:0 16px 38px rgba(15,23,42,.06);
    }
    body:has(.login-page-marker) div[data-testid="stForm"] input{
        background:#f8fafc!important;
        border:1px solid #e2e8f0!important;
        border-radius:12px!important;
        color:#0f172a!important;
        min-height:48px!important;
    }
    body:has(.login-page-marker) div[data-testid="stForm"] input:focus{
        border-color:#60a5fa!important;
        box-shadow:0 0 0 3px #dbeafe!important;
    }
    body:has(.login-page-marker) div[data-testid="stForm"] label{
        color:#334155!important;
        font-weight:700!important;
        font-size:.78rem!important;
    }
    body:has(.login-page-marker) div[data-testid="stForm"] button{
        min-height:48px!important;
        border-radius:12px!important;
        border:1px solid #0a4b94!important;
        background:linear-gradient(135deg,#0a4b94,#1f72b8)!important;
        color:#fff!important;
        font-weight:800!important;
        box-shadow:0 10px 20px rgba(10,75,148,.16);
    }
    body:has(.login-page-marker) div[data-testid="stForm"] button:hover{
        transform:translateY(-1px);
        box-shadow:0 12px 24px rgba(10,75,148,.20);
    }

    @media (max-width: 980px){
        .login-visual{height:610px;min-height:610px}
        .login-visual-content{left:28px;right:28px}
        .login-visual-footer{left:28px;right:28px}
        .login-visual-content h2{font-size:1.9rem!important}
    }
    @media (max-width: 760px){
        .login-page{padding:8px 0 24px}
        .login-visual{height:420px;min-height:420px;border-radius:22px}
        .login-visual-content{top:24px;left:24px;right:24px;bottom:54px}
        .login-feature-list{display:none}
        .login-visual-text{font-size:.82rem;line-height:1.5}
        .login-visual-footer{left:24px;right:24px;bottom:18px}
        .login-panel-head{padding:18px 4px 10px}
        .login-panel-head h1{font-size:1.65rem!important}
        body:has(.login-page-marker) div[data-testid="stForm"]{padding:.95rem .9rem}
    }
    .admin-badge{display:inline-block;padding:3px 9px;border-radius:999px;background:#e0f2fe;color:#075985;font-weight:700;font-size:.75rem}
    .user-badge{display:inline-block;padding:3px 9px;border-radius:999px;background:#f1f5f9;color:#475569;font-weight:700;font-size:.75rem}

    /* ===== Modern academic overview — LIGHT MODE ===== */
    .overview-shell{margin-top:.25rem;margin-bottom:.9rem}
    .overview-eyebrow{display:inline-flex;align-items:center;gap:.4rem;font-size:.72rem;letter-spacing:.11em;text-transform:uppercase;font-weight:800;color:#2563eb;margin-bottom:.28rem}
    .overview-eyebrow:before{content:"";width:8px;height:8px;border-radius:50%;background:#2563eb;box-shadow:0 0 0 4px #dbeafe}
    .overview-title{font-size:1.62rem;font-weight:850;color:#0f2747;margin:0 0 .28rem;line-height:1.18}
    .overview-subtitle{font-size:.88rem;color:#64748b;margin:0;line-height:1.55;max-width:900px}

    .overview-hero{position:relative;overflow:hidden;border:1px solid #bfdbfe;border-radius:22px;padding:1.2rem 1.25rem;background:#ffffff;box-shadow:0 10px 28px rgba(30,64,175,.08);margin-bottom:1rem}
    .overview-hero:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:linear-gradient(180deg,#2563eb,#06b6d4)}
    .overview-hero-inner{display:flex;gap:1rem;align-items:flex-start}
    .overview-status-dot{width:52px;height:52px;border-radius:17px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;flex:0 0 52px;font-weight:900;border:1px solid transparent}
    .overview-status-dot.safe{background:#ecfdf5;color:#047857;border-color:#a7f3d0}.overview-status-dot.moderate{background:#fffbeb;color:#b45309;border-color:#fde68a}.overview-status-dot.warning{background:#fff7ed;color:#c2410c;border-color:#fed7aa}.overview-status-dot.critical{background:#fef2f2;color:#b91c1c;border-color:#fecaca}
    .overview-hero-kicker{font-size:.7rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748b}
    .overview-hero-title{font-size:1.23rem;font-weight:850;color:#0f2747;margin:.12rem 0 .34rem}
    .overview-badge{display:inline-flex;align-items:center;padding:.28rem .62rem;border-radius:999px;font-size:.7rem;font-weight:850;margin-bottom:.45rem;border:1px solid transparent}
    .overview-badge.safe{background:#ecfdf5;color:#047857;border-color:#a7f3d0}.overview-badge.moderate{background:#fffbeb;color:#a16207;border-color:#fde68a}.overview-badge.warning{background:#fff7ed;color:#c2410c;border-color:#fed7aa}.overview-badge.critical{background:#fef2f2;color:#b91c1c;border-color:#fecaca}
    .overview-description{font-size:.88rem;line-height:1.62;color:#475569;margin:0}
    .overview-progress{margin-top:.9rem}
    .overview-progress-row{display:flex;justify-content:space-between;align-items:center;font-size:.74rem;color:#64748b;margin-bottom:.4rem}
    .overview-progress-value{font-weight:850;color:#0f2747}
    .overview-progress-track{height:9px;border-radius:999px;background:#eaf0f7;overflow:hidden;box-shadow:inset 0 1px 2px rgba(15,23,42,.05)}
    .overview-progress-fill{height:100%;border-radius:999px}
    .overview-progress-fill.safe{background:linear-gradient(90deg,#10b981,#34d399)}.overview-progress-fill.moderate{background:linear-gradient(90deg,#eab308,#f59e0b)}.overview-progress-fill.warning{background:linear-gradient(90deg,#f97316,#fb923c)}.overview-progress-fill.critical{background:linear-gradient(90deg,#ef4444,#f87171)}

    .overview-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin-bottom:1rem}
    .overview-metric{position:relative;border:1px solid #e2e8f0;background:#fff;border-radius:17px;padding:.9rem 1rem .95rem;box-shadow:0 5px 18px rgba(15,23,42,.045);overflow:hidden}
    .overview-metric:after{content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:#dbeafe}
    .overview-metric.good:after{background:#34d399}.overview-metric.warn:after{background:#fbbf24}.overview-metric.danger:after{background:#f87171}.overview-metric.info:after{background:#60a5fa}
    .overview-metric-label{font-size:.7rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:.045em}
    .overview-metric-value{font-size:1.42rem;font-weight:850;color:#0f2747;margin-top:.18rem;line-height:1.15}
    .overview-metric-note{font-size:.72rem;color:#94a3b8;margin-top:.24rem;line-height:1.4}
    .overview-metric.good .overview-metric-value{color:#047857}.overview-metric.warn .overview-metric-value{color:#a16207}.overview-metric.danger .overview-metric-value{color:#dc2626}.overview-metric.info .overview-metric-value{color:#1d4ed8}

    .overview-columns{display:grid;grid-template-columns:1.15fr .85fr;gap:.9rem;margin-bottom:.9rem}
    .overview-card{border:1px solid #e2e8f0;background:#fff;border-radius:18px;padding:1rem;box-shadow:0 5px 20px rgba(15,23,42,.045);height:100%}
    .overview-section-title{display:flex;align-items:center;gap:.55rem;font-size:.9rem;font-weight:850;color:#0f2747;margin:0 0 .62rem}
    .overview-section-title:before{content:"";width:4px;height:19px;border-radius:99px;background:#2563eb}
    .overview-factor{display:flex;gap:.65rem;align-items:flex-start;padding:.62rem .68rem;border-radius:13px;margin:.44rem 0;font-size:.82rem;line-height:1.5}
    .overview-factor.positive{background:#f0fdf4;border:1px solid #bbf7d0;color:#166534}.overview-factor.attention{background:#fffbeb;border:1px solid #fde68a;color:#92400e}.overview-factor.risk{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}.overview-factor.neutral{background:#f8fafc;border:1px solid #e2e8f0;color:#475569}
    .overview-factor-icon{font-weight:900;flex:0 0 21px;text-align:center}

    .overview-condition{display:flex;justify-content:space-between;gap:.75rem;align-items:center;padding:.68rem .76rem;border-radius:13px;border:1px solid #e2e8f0;margin:.46rem 0;background:#f8fafc}
    .overview-condition-name{font-size:.81rem;font-weight:800;color:#334155}.overview-condition-status{font-size:.7rem;font-weight:850;padding:.24rem .52rem;border-radius:999px;white-space:nowrap;border:1px solid transparent}
    .overview-condition-status.done{background:#ecfdf5;color:#047857;border-color:#a7f3d0}.overview-condition-status.pending{background:#fffbeb;color:#a16207;border-color:#fde68a}

    .overview-xai{background:#f8fbff;border:1px solid #cfe1ff;border-radius:16px;padding:.92rem 1rem;box-shadow:0 4px 15px rgba(37,99,235,.04)}
    .overview-xai-title{font-size:.78rem;font-weight:850;color:#1d4ed8;text-transform:uppercase;letter-spacing:.05em;display:flex;align-items:center;gap:.45rem}
    .overview-xai-title:before{content:"✦";font-size:.85rem}
    .overview-xai-text{font-size:.8rem;line-height:1.62;color:#475569;margin:.38rem 0 0}

    @media (max-width: 1050px){.overview-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.overview-columns{grid-template-columns:1fr}}
    @media (max-width: 600px){.overview-grid{grid-template-columns:1fr}.overview-hero-inner{flex-direction:column}.overview-title{font-size:1.35rem}}
</style>
"""
st.markdown(ctu_adaptive_theme_css, unsafe_allow_html=True)

if not require_login():
    st.stop()

if not is_admin():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        section.main { width: 100% !important; max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

current_user = st.session_state.get("auth_user", {})
role_badge = "<span class='admin-badge'>ADMIN</span>" if is_admin() else "<span class='user-badge'>USER</span>"

# --- PREPARE LOGO ---
logo_b64 = get_ctu_logo_b64()
logo_img_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="height: 48px; width: auto; object-fit: contain; margin-right: 14px;" alt="Logo CTU">'
    if logo_b64 else '<div style="background:#0a4b94; color:#fff; font-weight:bold; padding:6px 12px; border-radius:6px; margin-right:12px;">CTU</div>'
)

# --- HEADER LAYOUT ---
head_left, head_right = st.columns([7.5, 2.5], vertical_alignment="center")

with head_left:
    st.markdown(f"""
    <div style="display: flex; align-items: center; background: #ffffff; padding: 10px 16px; border-radius: 8px; border: 1px solid #e2e8f0; border-left: 5px solid #0a4b94;">
        {logo_img_html}
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #0a4b94; letter-spacing: 0.5px; text-transform: uppercase;">ĐẠI HỌC CẦN THƠ</div>
            <div style="font-size: 18px; font-weight: 800; color: #0f172a; line-height: 1.2;">STUDENT PROGRESS AI</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Dự báo tiến độ học tập · Tư vấn lộ trình cá nhân hóa với AI ·</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_right:
    user_col, logout_col = st.columns([6, 4], vertical_alignment="center")
    
    with user_col:
        st.markdown(f"""
        <div style="text-align: right; line-height: 1.2;">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 600;">Tài khoản</div>
            <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-top: 2px;">
                {current_user.get('username','')} {role_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with logout_col:
        if st.button("Đăng xuất", key="top_nav_logout", use_container_width=True):
            st.session_state.pop("auth_user", None)
            st.rerun()


# -----------------------------------------------------------------------------
# 3. DANH MỤC CÁC HỌC PHẦN GDTC, NGOẠI NGỮ & GDQP-AN THEO CHUẨN ĐH CẦN THƠ
# -----------------------------------------------------------------------------
GDTC_SPORTS = {
    "Điền kinh": [
        {"code": "TC001", "name": "Điền kinh 1", "credits": 1, "prereq": []},
        {"code": "TC002", "name": "Điền kinh 2", "credits": 1, "prereq": ["TC001"]},
        {"code": "TC024", "name": "Điền kinh 3", "credits": 1, "prereq": ["TC002"]}
    ],
    "Taekwondo": [
        {"code": "TC003", "name": "Taekwondo 1", "credits": 1, "prereq": []},
        {"code": "TC004", "name": "Taekwondo 2", "credits": 1, "prereq": ["TC003"]},
        {"code": "TC019", "name": "Taekwondo 3", "credits": 1, "prereq": ["TC004"]}
    ],
    "Bóng chuyền": [
        {"code": "TC005", "name": "Bóng chuyền 1", "credits": 1, "prereq": []},
        {"code": "TC006", "name": "Bóng chuyền 2", "credits": 1, "prereq": ["TC005"]},
        {"code": "TC020", "name": "Bóng chuyền 3", "credits": 1, "prereq": ["TC006"]}
    ],
    "Bóng đá": [
        {"code": "TC007", "name": "Bóng đá 1", "credits": 1, "prereq": []},
        {"code": "TC008", "name": "Bóng đá 2", "credits": 1, "prereq": ["TC007"]},
        {"code": "TC021", "name": "Bóng đá 3", "credits": 1, "prereq": ["TC008"]}
    ],
    "Bóng bàn": [
        {"code": "TC009", "name": "Bóng bàn 1", "credits": 1, "prereq": []},
        {"code": "TC010", "name": "Bóng bàn 2", "credits": 1, "prereq": ["TC009"]},
        {"code": "TC022", "name": "Bóng bàn 3", "credits": 1, "prereq": ["TC010"]}
    ],
    "Cầu lông": [
        {"code": "TC011", "name": "Cầu lông 1", "credits": 1, "prereq": []},
        {"code": "TC012", "name": "Cầu lông 2", "credits": 1, "prereq": ["TC011"]},
        {"code": "TC023", "name": "Cầu lông 3", "credits": 1, "prereq": ["TC012"]}
    ],
    "Thể dục nhịp điệu": [
        {"code": "TC016", "name": "Thể dục nhịp điệu 1", "credits": 1, "prereq": []},
        {"code": "TC017", "name": "Thể dục nhịp điệu 2", "credits": 1, "prereq": ["TC016"]},
        {"code": "TC018", "name": "Thể dục nhịp điệu 3", "credits": 1, "prereq": ["TC017"]}
    ],
    "Cờ vua": [
        {"code": "TC025", "name": "Cờ vua 1", "credits": 1, "prereq": []},
        {"code": "TC026", "name": "Cờ vua 2", "credits": 1, "prereq": ["TC025"]},
        {"code": "TC027", "name": "Cờ vua 3", "credits": 1, "prereq": ["TC026"]}
    ],
    "Bóng rổ": [
        {"code": "TC028", "name": "Bóng rổ 1", "credits": 1, "prereq": []},
        {"code": "TC029", "name": "Bóng rổ 2", "credits": 1, "prereq": ["TC028"]},
        {"code": "TC030", "name": "Bóng rổ 3", "credits": 1, "prereq": ["TC029"]}
    ],
    "Bơi lội": [
        {"code": "TC013", "name": "Bơi lội", "credits": 1, "prereq": []}
    ]
}

FL_COURSES = [
    {"code": "XH023", "name": "Anh văn căn bản 1", "credits": 4},
    {"code": "XH024", "name": "Anh văn căn bản 2", "credits": 3},
    {"code": "XH025", "name": "Anh văn căn bản 3", "credits": 3},
    {"code": "XH031", "name": "Anh văn tăng cường 1", "credits": 4},
    {"code": "XH032", "name": "Anh văn tăng cường 2", "credits": 3},
    {"code": "XH033", "name": "Anh văn tăng cường 3", "credits": 3},
    {"code": "FL001", "name": "Pháp văn căn bản 1", "credits": 4},
    {"code": "FL002", "name": "Pháp văn căn bản 2", "credits": 3},
    {"code": "FL003", "name": "Pháp văn căn bản 3", "credits": 3},
    {"code": "FL007", "name": "Pháp văn tăng cường 1", "credits": 4},
    {"code": "FL008", "name": "Pháp văn tăng cường 2", "credits": 3},
    {"code": "FL009", "name": "Pháp văn tăng cường 3", "credits": 3},
]

GDQP_COURSES = [
    {"code": "QP010E", "name": "Giáo dục quốc phòng và An ninh 1 (*)", "credits": 2},
    {"code": "QP011E", "name": "Giáo dục quốc phòng và An ninh 2 (*)", "credits": 2},
    {"code": "QP012", "name": "Giáo dục quốc phòng và An ninh 3 (*)", "credits": 2},
    {"code": "QP013", "name": "Giáo dục quốc phòng và An ninh 4 (*)", "credits": 2},
]


# -----------------------------------------------------------------------------
# 4. HÀM XỬ LÝ NẠP BẢNG WIDE MATRIX & GIẢI NÉN THEO HỌC KỲ / MÔN HỌC
# -----------------------------------------------------------------------------
def load_wide_matrix_file(file_path_or_buffer):
    if file_path_or_buffer is None:
        return None
    df = pd.read_csv(file_path_or_buffer, low_memory=False)
    
    rename_dict = {}
    for col in df.columns:
        c_upper = col.strip().upper()
        if c_upper in ["MSSV", "STUDENT_ID", "MA_SV"]:
            rename_dict[col] = "MSSV"
        elif c_upper in ["MALOP", "MA_LOP", "CLASS_CODE", "TENLOP"]:
            rename_dict[col] = "MaLop"
        elif c_upper in ["NGANHHOC", "NGANH_HOC", "MAJOR"]:
            rename_dict[col] = "NganhHoc"
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def gpa_10_to_4(gpa_10):
    if pd.isna(gpa_10) or gpa_10 is None:
        return np.nan
    try:
        val = float(gpa_10)
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


def extract_batch_code(mssv_str):
    if not mssv_str:
        return "B23"
    match = re.search(r"(B\d{2})", str(mssv_str).strip().upper())
    if match:
        batch = match.group(1)
        if batch in ["B21", "B22", "B23", "B24", "B25", "B26"]:
            return batch
        elif batch in ["B18", "B19", "B20"]:
            return "B21"
        elif int(batch[1:]) > 26:
            return "B26"
    return "B23"


def parse_student_semester_history(sv_row):
    sem_list = []
    for k in range(1, 13):
        prefix = f"HK{k}_"
        tc_dk_col = f"{prefix}So_TC_Dang_Ky"
        
        if tc_dk_col in sv_row.index and pd.notna(sv_row[tc_dk_col]):
            tc_dk = float(sv_row[tc_dk_col])
            if tc_dk > 0 or pd.notna(sv_row.get(f"{prefix}DTBCHK_Thang_10")):
                tc_dat = float(sv_row.get(f"{prefix}So_TC_Tich_Luy_Dat", 0))
                tc_no = float(sv_row.get(f"{prefix}So_TC_No_Dong", 0))
                dtb_10 = float(sv_row.get(f"{prefix}DTBCHK_Thang_10", 0.0))
                dtb_4 = float(sv_row.get(f"{prefix}DTBCHK_Thang_4", 0.0))
                cpa_10 = float(sv_row.get(f"{prefix}CPA_Tich_Luy_Thang_10", 0.0))
                cpa_4 = float(sv_row.get(f"{prefix}CPA_Tich_Luy_Thang_4", 0.0))
                
                raw_fail = sv_row.get(f"{prefix}Ty_Le_Rot_Mon", "0.0%")
                if isinstance(raw_fail, str):
                    fail_rate = float(raw_fail.replace("%", "").strip())
                else:
                    fail_rate = float(raw_fail) if pd.notna(raw_fail) else 0.0

                xep_loai = str(sv_row.get(f"{prefix}Xep_Loai_HK", "Chưa rõ"))

                sem_list.append({
                    "Học kỳ": f"Học kỳ {k}",
                    "hk_num": k,
                    "Số TC Đăng ký": int(tc_dk),
                    "Số TC Tích lũy (Đạt)": int(tc_dat),
                    "Số TC Nợ đọng": int(tc_no),
                    "ĐTBCHK (Thang 10)": dtb_10,
                    "ĐTBCHK (Thang 4)": dtb_4,
                    "ĐTBCTL (Thang 10)": cpa_10,
                    "ĐTBCTL (Thang 4)": cpa_4,
                    "Tỷ lệ rớt môn": f"{fail_rate:.1f}%",
                    "fail_rate_num": fail_rate,
                    "Xếp loại HK": xep_loai
                })

    if not sem_list:
        sem_list.append({
            "Học kỳ": "Học kỳ 1",
            "hk_num": 1,
            "Số TC Đăng ký": 16,
            "Số TC Tích lũy (Đạt)": 16,
            "Số TC Nợ đọng": 0,
            "ĐTBCHK (Thang 10)": 7.0,
            "ĐTBCHK (Thang 4)": 3.0,
            "ĐTBCTL (Thang 10)": 7.0,
            "ĐTBCTL (Thang 4)": 3.0,
            "Tỷ lệ rớt môn": "0.0%",
            "fail_rate_num": 0.0,
            "Xếp loại HK": "Khá"
        })

    return pd.DataFrame(sem_list)


def parse_student_course_details(sv_row):
    diem_cols = [c for c in sv_row.index if str(c).startswith("DIEM_")]
    passed_courses = set()
    failed_courses = set()
    low_grade_courses = set()
    exempt_courses = set()
    grade_dict = {}

    for col in diem_cols:
        course_code = col.replace("DIEM_", "").strip().upper()
        val_score = sv_row[col]

        solan_hoc = int(sv_row.get(f"SOLAN_HOC_{course_code}", 0)) if pd.notna(sv_row.get(f"SOLAN_HOC_{course_code}")) else 0
        solan_hoan_thanh = int(sv_row.get(f"SOLAN_HOAN_THANH_{course_code}", 0)) if pd.notna(sv_row.get(f"SOLAN_HOAN_THANH_{course_code}")) else 0

        if pd.notna(val_score):
            s_str = str(val_score).strip().upper()
            if s_str in ["M", "MIEN", "MIỄN"]:
                exempt_courses.add(course_code)
                passed_courses.add(course_code)
                grade_dict[course_code] = "M (Miễn)"
            else:
                try:
                    sc = float(val_score)
                    grade_dict[course_code] = sc
                    if sc >= 4.0 or solan_hoan_thanh > 0:
                        passed_courses.add(course_code)
                        if 4.0 <= sc < 5.5:
                            low_grade_courses.add(course_code)
                    else:
                        failed_courses.add(course_code)
                except (ValueError, TypeError):
                    pass
        elif solan_hoc > 0:
            if solan_hoan_thanh > 0:
                passed_courses.add(course_code)
            else:
                failed_courses.add(course_code)

    failed_courses = failed_courses - passed_courses
    return passed_courses, failed_courses, low_grade_courses, exempt_courses, grade_dict


def auto_detect_major_from_row(sv_row):
    ma_lop = str(sv_row.get("MaLop", sv_row.get("ma_lop", "KXD"))).strip().upper()
    nganh_hoc_str = str(sv_row.get("NganhHoc", sv_row.get("nganh_hoc", ""))).strip()

    if any(k in nganh_hoc_str.upper() for k in ["AN TOÀN", "ATTT", "MAT MA"]):
        return "ATTT", ma_lop
    elif any(k in nganh_hoc_str.upper() for k in ["MẠNG", "MMT", "TRUYỀN THÔNG"]):
        return "MMT", ma_lop

    if any(k in ma_lop for k in ["D2", "ATTT", "7480202"]):
        return "ATTT", ma_lop
    elif any(k in ma_lop for k in ["T9", "MMT", "7480102"]):
        return "MMT", ma_lop

    return "ATTT", ma_lop


def get_certificate_exempt_courses(student_id):
    item = load_certificate_registry().get(str(student_id), {})
    if not isinstance(item, dict): return set()
    if item.get("status") == "DA_DUYET":
        return set(str(x).strip().upper() for x in item.get("exempt_courses", []))
    return set()

def save_student_certificate(student_id, certificate_data):
    registry = load_certificate_registry()
    registry[str(student_id)] = certificate_data
    save_certificate_registry(registry)

def analyze_fl_progress(passed_courses, exempt_courses, grade_dict):
    all_fl = {c["code"]: dict(c) for c in FL_COURSES}

    fl_results = []
    total_fl_credits_completed = 0
    total_fl_courses_completed = 0
    has_m_exemption = False

    for code, info in all_fl.items():
        gr = grade_dict.get(code, None)
        is_in_dataset = (code in passed_courses) or (code in exempt_courses) or (gr is not None)
        is_exempt = (code in exempt_courses) or (gr == "M (Miễn)") or (isinstance(gr, str) and gr.strip().upper() in ["M", "MIEN", "MIỄN"])

        if is_exempt:
            status = "🏅 Đã miễn (Điểm M - Nộp chứng chỉ)"
            score_display = "M"
            is_done = True
            has_m_exemption = True
        elif is_in_dataset:
            if isinstance(gr, (int, float)):
                score_display = f"{gr:.1f}"
                if gr >= 4.0:
                    status = f"✅ Đã qua môn ({gr}/10)"
                    is_done = True
                else:
                    status = f"❌ Chưa đạt ({gr}/10 - Cần học lại)"
                    is_done = False
            elif code in passed_courses:
                score_display = "Đạt"
                status = "✅ Đã qua môn"
                is_done = True
            else:
                score_display = "Chưa đạt"
                status = "❌ Chưa đạt"
                is_done = False
        else:
            score_display = "—"
            status = "⏳ Chưa đăng ký học"
            is_done = False

        if is_done:
            total_fl_credits_completed += info["credits"]
            total_fl_courses_completed += 1

        fl_results.append({
            "code": info["code"],
            "name": info["name"],
            "credits": info["credits"],
            "score": score_display,
            "status": status,
            "is_completed": is_done
        })

    REQUIRED_FL_CREDITS = 10
    is_qualified = (total_fl_credits_completed >= REQUIRED_FL_CREDITS) or has_m_exemption

    return {
        "fl_courses": fl_results,
        "total_credits": total_fl_credits_completed,
        "total_courses": total_fl_courses_completed,
        "has_m_exemption": has_m_exemption,
        "is_qualified": is_qualified,
        "required_credits": REQUIRED_FL_CREDITS
    }


def analyze_gdqp_progress(passed_courses, exempt_courses, grade_dict):
    qp_list = GDQP_COURSES
    qp_in_student = [c for c in grade_dict.keys() if c.startswith("QP")]
    if any(c in qp_in_student for c in ["QP008", "QP009"]):
        qp_list = [
            {"code": "QP008", "name": "Đường lối quốc phòng và an ninh của Đảng CSVN", "credits": 3},
            {"code": "QP009", "name": "Công tác quốc phòng và an ninh", "credits": 2},
            {"code": "QP010", "name": "Quân sự chung", "credits": 2},
            {"code": "QP011", "name": "Kỹ thuật chiến đấu bộ binh và thuật chiến", "credits": 4},
        ]

    gdqp_results = []
    completed_count = 0

    for course in qp_list:
        code = course["code"]
        gr = grade_dict.get(code, None)
        is_exempt = (code in exempt_courses) or (gr == "M (Miễn)") or (isinstance(gr, str) and gr.strip().upper() in ["M", "MIEN", "MIỄN"])

        if is_exempt:
            status = "🏅 Đã miễn (Điểm M)"
            score_display = "M"
            is_done = True
        elif code in passed_courses:
            score_display = f"{gr:.1f}" if isinstance(gr, (int, float)) else "Đạt"
            status = "✅ Đã qua môn"
            is_done = True
        elif gr is not None:
            score_display = f"{gr:.1f}" if isinstance(gr, (int, float)) else "Chưa đạt"
            status = "❌ Chưa đạt (Học lại)"
            is_done = False
        else:
            score_display = "—"
            status = "⏳ Chưa hoàn thành"
            is_done = False

        if is_done:
            completed_count += 1

        gdqp_results.append({
            "code": code,
            "name": course["name"],
            "credits": course["credits"],
            "score": score_display,
            "status": status
        })

    is_qualified = (completed_count >= len(qp_list))

    return {
        "gdqp_courses": gdqp_results,
        "completed_count": completed_count,
        "total_courses": len(qp_list),
        "is_qualified": is_qualified
    }


# -----------------------------------------------------------------------------
# 5. METADATA VÀ CƠ SỞ DỮ LIỆU CTĐT ĐẠI HỌC CẦN THƠ
# -----------------------------------------------------------------------------
PROGRAM_METADATA = {
    "ATTT": {
        "name": "An toàn thông tin",
        "code": "7480202",
        "mapping": {
            "B21": {
                "decision": "Quyết định số 3334/QĐ-ĐHCT (ngày 24/08/2022)",
                "department": "Bộ môn Mạng máy tính và Truyền thông, Khoa CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 121, Tự chọn: 40)",
                "khoa": "Khóa 47 (Nhập học 2021)",
                "note": "Áp dụng khung chương trình 2022"
            },
            "B22": {
                "decision": "Quyết định số 3334/QĐ-ĐHCT (ngày 24/08/2022)",
                "department": "Bộ môn Mạng máy tính và Truyền thông, Khoa CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 121, Tự chọn: 40)",
                "khoa": "Khóa 48 (Nhập học 2022)",
                "note": "Áp dụng khung chương trình 2022"
            },
            "B23": {
                "decision": "Quyết định số 2423/QĐ-ĐHCT (ngày 05/06/2023)",
                "department": "Khoa Mạng máy tính và Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 121, Tự chọn: 40)",
                "khoa": "Khóa 49 (Nhập học 2023)",
                "note": "Áp dụng khung chương trình 2023"
            },
            "B24": {
                "decision": "Quyết định số 2423/QĐ-ĐHCT (ngày 05/06/2023)",
                "department": "Khoa Mạng máy tính và Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 121, Tự chọn: 40)",
                "khoa": "Khóa 50 (Nhập học 2024)",
                "note": "Áp dụng khung chương trình 2023"
            },
            "B25": {
                "decision": "Quyết định số 2423/QĐ-ĐHCT (ngày 05/06/2023)",
                "department": "Khoa Mạng máy tính và Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 121, Tự chọn: 40)",
                "khoa": "Khóa 51 (Nhập học 2025)",
                "note": "Đổi học phần CT200 (Nền tảng CNTT) thành CT005 (Nền tảng công nghệ số)"
            },
            "B26": {
                "decision": "Quyết định số 3922/QĐ-ĐHCT (ngày 15/08/2025)",
                "department": "Khoa Mạng máy tính và Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 121, Tự chọn: 40)",
                "khoa": "Khóa 52 (Nhập học 2026)",
                "note": "Áp dụng khung chương trình cập nhật mới nhất"
            }
        }
    },
    "MMT": {
        "name": "Mạng máy tính và Truyền thông dữ liệu",
        "code": "7480102",
        "mapping": {
            "B21": {
                "decision": "Quyết định số 3334/QĐ-ĐHCT (ngày 24/08/2022)",
                "department": "Bộ môn Mạng máy tính và Truyền thông, Khoa CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 116, Tự chọn: 45)",
                "khoa": "Khóa 47 (Nhập học 2021)",
                "note": "Áp dụng khung chương trình 2022"
            },
            "B22": {
                "decision": "Quyết định số 3334/QĐ-ĐHCT (ngày 24/08/2022)",
                "department": "Bộ môn Mạng máy tính và Truyền thông, Khoa CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 116, Tự chọn: 45)",
                "khoa": "Khóa 48 (Nhập học 2022)",
                "note": "Áp dụng khung chương trình 2022"
            },
            "B23": {
                "decision": "Quyết định số 2423/QĐ-ĐHCT (ngày 05/06/2023)",
                "department": "Khoa Mạng máy tính & Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 116, Tự chọn: 45)",
                "khoa": "Khóa 49 (Nhập học 2023)",
                "note": "Áp dụng khung chương trình 2023"
            },
            "B24": {
                "decision": "Quyết định số 2423/QĐ-ĐHCT (ngày 05/06/2023)",
                "department": "Khoa Mạng máy tính & Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 116, Tự chọn: 45)",
                "khoa": "Khóa 50 (Nhập học 2024)",
                "note": "Áp dụng khung chương trình 2023"
            },
            "B25": {
                "decision": "Quyết định số 3922/QĐ-ĐHCT (ngày 15/08/2025)",
                "department": "Khoa Mạng máy tính và Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 116, Tự chọn: 45)",
                "khoa": "Khóa 51 (Nhập học 2025)",
                "note": "Đổi học phần CT200 (Nền tảng CNTT) thành CT005 (Nền tảng công nghệ số)"
            },
            "B26": {
                "decision": "Quyết định số 3922/QĐ-ĐHCT (ngày 15/08/2025)",
                "department": "Khoa Mạng máy tính và Truyền thông, Trường CNTT&TT",
                "duration": "4.5 năm",
                "credits": "161 Tín chỉ (Bắt buộc: 116, Tự chọn: 45)",
                "khoa": "Khóa 52 (Nhập học 2026)",
                "note": "Áp dụng khung chương trình cập nhật mới nhất"
            }
        }
    }
}


def get_program_courses(major_key, batch_code):
    admin_courses = get_admin_program_courses(major_key, batch_code)
    if admin_courses:
        return admin_courses
    poli_courses = [
        {"code": "ML014", "name": "Triết học Mác - Lênin", "credits": 3, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "ML016", "name": "Kinh tế chính trị Mác - Lênin", "credits": 2, "type": "Bắt buộc", "prereq": ["ML014"], "min_credits": 0},
        {"code": "ML018", "name": "Chủ nghĩa xã hội khoa học", "credits": 2, "type": "Bắt buộc", "prereq": ["ML016"], "min_credits": 0},
        {"code": "ML019", "name": "Lịch sử Đảng Cộng sản Việt Nam", "credits": 2, "type": "Bắt buộc", "prereq": ["ML018"], "min_credits": 0},
        {"code": "ML021", "name": "Tư tưởng Hồ Chí Minh", "credits": 2, "type": "Bắt buộc", "prereq": ["ML019"], "min_credits": 0},
    ]

    general_law_skills = [
        {"code": "KL001E", "name": "Pháp luật đại cương", "credits": 2, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "KN001E", "name": "Kỹ năng mềm", "credits": 2, "type": "Tự chọn", "prereq": [], "min_credits": 0},
        {"code": "KN002E", "name": "Đổi mới sáng tạo và khởi nghiệp", "credits": 2, "type": "Tự chọn", "prereq": [], "min_credits": 0},
        {"code": "ML007", "name": "Logic học đại cương", "credits": 2, "type": "Tự chọn", "prereq": [], "min_credits": 0},
        {"code": "XH028", "name": "Xã hội học đại cương", "credits": 2, "type": "Tự chọn", "prereq": [], "min_credits": 0},
        {"code": "XH011", "name": "Cơ sở văn hóa Việt Nam", "credits": 2, "type": "Tự chọn", "prereq": [], "min_credits": 0},
    ]

    qp_courses = [
        {"code": "QP010E", "name": "Giáo dục quốc phòng và An ninh 1 (*)", "credits": 2, "type": "Điều kiện", "prereq": [], "min_credits": 0},
        {"code": "QP011E", "name": "Giáo dục quốc phòng và An ninh 2 (*)", "credits": 2, "type": "Điều kiện", "prereq": [], "min_credits": 0},
        {"code": "QP012", "name": "Giáo dục quốc phòng và An ninh 3 (*)", "credits": 2, "type": "Điều kiện", "prereq": [], "min_credits": 0},
        {"code": "QP013", "name": "Giáo dục quốc phòng và An ninh 4 (*)", "credits": 2, "type": "Điều kiện", "prereq": [], "min_credits": 0},
    ]

    math_general = [
        {"code": "TN001", "name": "Vi - Tích phân A1", "credits": 3, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "TN002", "name": "Vi - Tích phân A2", "credits": 4, "type": "Bắt buộc", "prereq": ["TN001"], "min_credits": 0},
        {"code": "TN010", "name": "Xác suất thống kê", "credits": 3, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "TN012", "name": "Đại số tuyến tính và hình học", "credits": 4, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT100", "name": "Kỹ năng học đại học", "credits": 2, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
    ]

    intro_course = [
        {"code": "CT005", "name": "Nền tảng công nghệ số", "credits": 4, "type": "Bắt buộc", "prereq": [], "min_credits": 0}
        if batch_code in ["B25", "B26"] else
        {"code": "CT200", "name": "Nền tảng công nghệ thông tin", "credits": 4, "type": "Bắt buộc", "prereq": [], "min_credits": 0}
    ]

    cs_core_courses = [
        {"code": "CT172", "name": "Toán rời rạc", "credits": 4, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT101", "name": "Lập trình căn bản A", "credits": 4, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT177", "name": "Cấu trúc dữ liệu", "credits": 3, "type": "Bắt buộc", "prereq": ["CT101"], "min_credits": 0},
        {"code": "CT175", "name": "Lý thuyết đồ thị", "credits": 3, "type": "Bắt buộc", "prereq": ["CT101"], "min_credits": 0},
        {"code": "CT174", "name": "Phân tích và thiết kế thuật toán", "credits": 3, "type": "Bắt buộc", "prereq": ["CT177"], "min_credits": 0},
        {"code": "CT180", "name": "Cơ sở dữ liệu", "credits": 3, "type": "Bắt buộc", "prereq": ["CT177"], "min_credits": 0},
        {"code": "CT173", "name": "Kiến trúc máy tính", "credits": 3, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT178", "name": "Nguyên lý hệ điều hành", "credits": 3, "type": "Bắt buộc", "prereq": ["CT173"], "min_credits": 0},
        {"code": "CT112", "name": "Mạng máy tính", "credits": 3, "type": "Bắt buộc", "prereq": ["CT178"], "min_credits": 0},
        {"code": "CT176", "name": "Lập trình hướng đối tượng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT101"], "min_credits": 0},
        {"code": "CT296", "name": "Phân tích và thiết kế HTTT", "credits": 3, "type": "Bắt buộc", "prereq": ["CT180"], "min_credits": 0},
        {"code": "CT182", "name": "Ngôn ngữ mô hình hóa", "credits": 3, "type": "Bắt buộc", "prereq": ["CT176"], "min_credits": 0},
        {"code": "CT179", "name": "Quản trị hệ thống", "credits": 3, "type": "Bắt buộc", "prereq": ["CT178"], "min_credits": 0},
        {"code": "CT188", "name": "Nhập môn lập trình Web", "credits": 3, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT190", "name": "Nhập môn trí tuệ nhân tạo", "credits": 2, "type": "Bắt buộc", "prereq": ["CT177"], "min_credits": 0},
    ]

    attt_specific = [
        {"code": "CT197", "name": "Cơ sở lý thuyết mật mã", "credits": 3, "type": "Bắt buộc", "prereq": ["CT172"], "min_credits": 0},
        {"code": "CT211", "name": "An ninh mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT485", "name": "Các kỹ thuật tấn công mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT489", "name": "Luật an ninh mạng & Đạo đức CNTT", "credits": 2, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT490", "name": "An ninh Web", "credits": 3, "type": "Bắt buộc", "prereq": ["CT188"], "min_credits": 0},
        {"code": "CT335", "name": "Thiết kế và cài đặt mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT428", "name": "Lập trình Web", "credits": 3, "type": "Bắt buộc", "prereq": ["CT180", "CT188"], "min_credits": 0},
        {"code": "CT294", "name": "Máy học ứng dụng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT190"], "min_credits": 0},
        {"code": "CT297", "name": "Pháp y máy tính", "credits": 3, "type": "Tự chọn", "prereq": [], "min_credits": 0},
        {"code": "CT486", "name": "Phát hiện và phân tích mã độc", "credits": 3, "type": "Tự chọn", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT491E", "name": "Niên luận cơ sở an toàn thông tin", "credits": 3, "type": "Đồ án / Niên luận", "prereq": [], "min_credits": 90},
        {"code": "CT492E", "name": "Niên luận an toàn thông tin", "credits": 3, "type": "Đồ án / Niên luận", "prereq": [], "min_credits": 110},
        {"code": "CT493E", "name": "Thực tập doanh nghiệp - ATTT", "credits": 5, "type": "Thực tập", "prereq": ["CT211", "CT335", "CT490"], "min_credits": 125},
        {"code": "CT556E", "name": "Luận văn tốt nghiệp - ATTT", "credits": 15, "type": "Tốt nghiệp", "prereq": [], "min_credits": 125},
    ]

    mmt_specific = [
        {"code": "CT335", "name": "Thiết kế và cài đặt mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT212", "name": "Quản trị mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT428", "name": "Lập trình Web", "credits": 3, "type": "Bắt buộc", "prereq": ["CT180", "CT188"], "min_credits": 0},
        {"code": "CT221", "name": "Lập trình mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT211", "name": "An ninh mạng", "credits": 3, "type": "Bắt buộc", "prereq": ["CT112"], "min_credits": 0},
        {"code": "CT233", "name": "Điện toán đám mây", "credits": 3, "type": "Bắt buộc", "prereq": [], "min_credits": 0},
        {"code": "CT126", "name": "Lý thuyết xếp hàng", "credits": 2, "type": "Tự chọn", "prereq": ["TN010"], "min_credits": 0},
        {"code": "CT227", "name": "Kỹ thuật phát hiện tấn công mạng", "credits": 3, "type": "Tự chọn", "prereq": ["CT211"], "min_credits": 0},
        {"code": "CT226E", "name": "Niên luận cơ sở mạng máy tính", "credits": 3, "type": "Đồ án / Niên luận", "prereq": [], "min_credits": 90},
        {"code": "CT439E", "name": "Niên luận mạng máy tính", "credits": 3, "type": "Đồ án / Niên luận", "prereq": [], "min_credits": 110},
        {"code": "CT517E", "name": "Thực tập doanh nghiệp - MMT", "credits": 5, "type": "Thực tập", "prereq": ["CT112"], "min_credits": 125},
        {"code": "CT555E", "name": "Luận văn tốt nghiệp - MMT", "credits": 15, "type": "Tốt nghiệp", "prereq": [], "min_credits": 125},
    ]

    base_common = poli_courses + general_law_skills + qp_courses + math_general + intro_course + cs_core_courses

    if major_key == "MMT":
        return base_common + mmt_specific
    else:
        return base_common + attt_specific


class CurriculumGraphEngine:
    def __init__(self, courses):
        self.courses = {str(c.get("code", "")).strip().upper(): dict(c) for c in courses if c.get("code")}
        self.forward = {code: set() for code in self.courses}
        self.reverse = {code: set() for code in self.courses}
        self.unknown_prereqs = {}
        for code, c in self.courses.items():
            prereqs = [str(x).strip().upper() for x in (c.get("prereq") or []) if str(x).strip()]
            for pre in prereqs:
                if pre in self.courses:
                    self.forward.setdefault(pre, set()).add(code)
                    self.reverse.setdefault(code, set()).add(pre)
                else:
                    self.unknown_prereqs.setdefault(code, []).append(pre)

    def _normalize_set(self, codes):
        return {str(x).strip().upper() for x in (codes or []) if str(x).strip()}

    def descendants(self, code, max_nodes=500):
        code = str(code).strip().upper()
        seen, queue = set(), list(self.forward.get(code, set()))
        while queue and len(seen) < max_nodes:
            cur = queue.pop(0)
            if cur in seen: continue
            seen.add(cur)
            queue.extend(self.forward.get(cur, set()) - seen)
        return seen

    def ancestors(self, code, max_nodes=500):
        code = str(code).strip().upper()
        seen, queue = set(), list(self.reverse.get(code, set()))
        while queue and len(seen) < max_nodes:
            cur = queue.pop(0)
            if cur in seen: continue
            seen.add(cur)
            queue.extend(self.reverse.get(cur, set()) - seen)
        return seen

    def missing_chain(self, target_code, passed_courses):
        passed = self._normalize_set(passed_courses)
        target = str(target_code).strip().upper()
        missing = self.ancestors(target) - passed
        return [self.courses[x] for x in sorted(missing) if x in self.courses]

    def direct_impact(self, failed_codes):
        passed = set()
        out = []
        for code in self._normalize_set(failed_codes):
            blocked = self.descendants(code)
            out.append({
                "code": code,
                "blocked_count": len(blocked),
                "blocked_courses": [self.courses[x] for x in sorted(blocked) if x in self.courses],
            })
        return out

    def eligibility(self, course, passed_courses, earned_credits):
        passed = self._normalize_set(passed_courses)
        code = str(course.get("code", "")).strip().upper()
        prereqs = [str(x).strip().upper() for x in (course.get("prereq") or [])]
        missing = [x for x in prereqs if x and x not in passed]
        unknown = self.unknown_prereqs.get(code, [])
        credit_need = int(course.get("min_credits") or 0)
        credit_ok = float(earned_credits or 0) >= credit_need
        eligible = not missing and not unknown and credit_ok
        reasons = []
        if missing: reasons.append("Chưa đạt tiên quyết: " + ", ".join(missing))
        if unknown: reasons.append("Tiên quyết chưa được xác thực trong CTĐT: " + ", ".join(unknown))
        if not credit_ok: reasons.append(f"Cần tích lũy ≥ {credit_need} TC (hiện có {earned_credits} TC)")
        return {"eligible": eligible, "missing_prereqs": missing, "unknown_prereqs": unknown,
                "credit_ok": credit_ok, "credit_need": credit_need, "reason": " | ".join(reasons)}

    def critical_courses(self, passed_courses):
        passed = self._normalize_set(passed_courses)
        rows = []
        for code, course in self.courses.items():
            if code in passed: continue
            impact = len(self.descendants(code))
            if impact:
                rows.append({**course, "downstream_impact": impact})
        return sorted(rows, key=lambda x: (-x["downstream_impact"], x.get("code", "")))

    def graph_summary(self, passed_courses, failed_courses):
        failed = self._normalize_set(failed_courses)
        return {
            "nodes": len(self.courses),
            "edges": sum(len(v) for v in self.forward.values()),
            "failed_impact": self.direct_impact(failed),
            "critical_courses": self.critical_courses(passed_courses)[:10],
            "unknown_prereq_count": sum(len(v) for v in self.unknown_prereqs.values()),
        }


def build_curriculum_graph(major_key, batch_code):
    return CurriculumGraphEngine(get_program_courses(major_key, batch_code))


def analyze_ctu_recommendations(major_key, batch_code, passed_courses, failed_courses, low_grade_courses, earned_credits):
    courses = get_program_courses(major_key, batch_code)
    graph = CurriculumGraphEngine(courses)
    eligible_mandatory, eligible_elective, eligible_projects = [], [], []
    retake_and_improve, locked_courses, condition_tracking = [], [], []

    for course in courses:
        code = course["code"]
        if course.get("type") == "Điều kiện":
            condition_tracking.append({**course, "status": "Đã hoàn thành (*)" if code in passed_courses else "Chưa tích lũy (*)"})
            continue
        if code in failed_courses:
            impact = len(graph.descendants(code))
            retake_and_improve.append({**course, "status_type": "HỌC LẠI BẮT BUỘC",
                "reason": "Đã bị rớt học phần này (Điểm F / < 4.0)", "downstream_impact": impact})
            continue
        if code in low_grade_courses:
            retake_and_improve.append({**course, "status_type": "CẢI THIỆN GPA",
                "reason": "Điểm hiện tại ở mức thấp (D/D+)", "downstream_impact": len(graph.descendants(code))})
        if code in passed_courses and code not in low_grade_courses:
            continue
        check = graph.eligibility(course, passed_courses, earned_credits)
        if check["eligible"]:
            if course.get("type") == "Bắt buộc": eligible_mandatory.append({**course, "priority_score": len(graph.descendants(code))})
            elif course.get("type") == "Tự chọn": eligible_elective.append(course)
            else: eligible_projects.append(course)
        else:
            locked_courses.append({**course, "reason": check["reason"],
                                   "missing_chain": ", ".join(x["code"] for x in graph.missing_chain(code, passed_courses)),
                                   "downstream_impact": len(graph.descendants(code))})

    eligible_mandatory.sort(key=lambda x: (-x.get("priority_score", 0), x.get("semester_hint", ""), x["code"]))
    retake_and_improve.sort(key=lambda x: (-x.get("downstream_impact", 0), x["code"]))

    gdtc_summary = []
    active_sport_found = False
    for sport_name, sport_courses in GDTC_SPORTS.items():
        completed_in_sport = [c for c in sport_courses if c["code"] in passed_courses]
        if len(completed_in_sport) == len(sport_courses):
            gdtc_summary.append({"sport": sport_name, "status": "Hoàn thành"})
        elif completed_in_sport:
            active_sport_found = True
            gdtc_summary.append({"sport": sport_name, "status": f"Đang học ({len(completed_in_sport)}/{len(sport_courses)})"})
    if not active_sport_found and not gdtc_summary:
        gdtc_summary.append({"sport": "Chưa xác định", "status": "Chưa hoàn thành chuỗi GDTC"})

    return {
        "mandatory": eligible_mandatory, "elective": eligible_elective, "projects": eligible_projects,
        "improve": retake_and_improve, "locked": locked_courses, "conditions": condition_tracking,
        "gdtc_summary": gdtc_summary, "graph": graph.graph_summary(passed_courses, failed_courses),
        "critical_path_candidates": graph.critical_courses(passed_courses)[:10],
    }

def _build_recommendation_courses(major_key, batch_code):
    if not HAS_PERSONALIZED_RECSYS:
        return []
    raw_courses = get_program_courses(major_key, batch_code)
    return [
        Course(
            course_id=str(c.get("code", "")).strip().upper(),
            course_name=str(c.get("name", "")).strip(),
            credits=float(c.get("credits", 0) or 0),
            prerequisites=[str(x).strip().upper() for x in c.get("prereq", [])],
            semester_hint=None,
            required=(str(c.get("type", "")).strip() == "Bắt buộc"),
            category=str(c.get("type", "general")),
        )
        for c in raw_courses
        if str(c.get("code", "")).strip()
    ]


def _build_student_state(
    selected_mssv, sem_df, passed_courses, failed_courses,
    tc_earned, cpa_4, dtbchk_4, fail_rate, risk_prob
):
    current_semester = int(sem_df.iloc[-1].get("hk_num", len(sem_df)))

    if len(sem_df) >= 2:
        previous_gpa = float(sem_df.iloc[-2]["ĐTBCHK (Thang 4)"])
        gpa_trend = dtbchk_4 - previous_gpa
    else:
        gpa_trend = 0.0

    recent_credits = sem_df["Số TC Tích lũy (Đạt)"].tail(3).astype(float)
    credit_velocity = float(recent_credits.diff().dropna().mean()) if len(recent_credits) >= 2 else float(recent_credits.iloc[-1] if len(recent_credits) else 0.0)

    total_program_credits = 161.0
    remaining_credits = max(0.0, total_program_credits - float(tc_earned))
    expected_credits = min(total_program_credits, current_semester * 18.0)
    credit_deficit = max(0.0, expected_credits - float(tc_earned))

    return StudentState(
        student_id=str(selected_mssv),
        current_semester=current_semester,
        passed_courses=sorted(str(x).strip().upper() for x in passed_courses),
        failed_courses=sorted(str(x).strip().upper() for x in failed_courses),
        current_courses=[],
        remaining_credits=remaining_credits,
        credit_deficit=credit_deficit,
        cumulative_gpa=float(cpa_4),
        latest_semester_gpa=float(dtbchk_4),
        gpa_trend=float(gpa_trend),
        credit_velocity=float(max(0.0, credit_velocity)),
        fail_rate=float(max(0.0, min(1.0, fail_rate / 100.0))),
        prerequisite_block_count=0,
        risk_probability=float(max(0.0, min(1.0, risk_prob))),
        history=sorted(str(x).strip().upper() for x in passed_courses),
    )


@st.cache_resource(show_spinner=False)
def load_personalized_recommendation_engine(major_key, batch_code):
    if not HAS_PERSONALIZED_RECSYS:
        return None, None, "PACKAGE_NOT_AVAILABLE"

    courses = _build_recommendation_courses(major_key, batch_code)
    if not courses:
        return None, {}, "NO_COURSES"

    ranker = None
    checkpoint = PROJECT_ROOT / "models" / "transformer_recsys_student.pth"
    try:
        course_to_idx = {"__PAD__": 0, **{course.course_id: idx + 1 for idx, course in enumerate(courses)}}
        ranker = TransformerRanker(
            checkpoint=checkpoint,
            course_to_idx=course_to_idx,
            device="cpu",
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_len=32,
        ) if checkpoint.exists() else None
    except Exception:
        ranker = None

    engine = PersonalizedPathway(
        courses=courses,
        ranker=ranker,
        transformer_weight=0.70,
        feature_weight=0.30,
    )
    courses_by_id = {course.course_id: course for course in courses}
    status = "TRANSFORMER+RULE+XAI" if ranker is not None else "RULE+XAI_FALLBACK"
    return engine, courses_by_id, status


def generate_personalized_recommendations(
    selected_mssv, major_key, batch_code, sem_df, passed_courses, failed_courses,
    tc_earned, cpa_4, dtbchk_4, fail_rate, risk_prob, top_k=8
):
    engine, courses_by_id, engine_status = load_personalized_recommendation_engine(major_key, batch_code)
    if engine is None:
        return [], engine_status

    student = _build_student_state(
        selected_mssv, sem_df, passed_courses, failed_courses,
        tc_earned, cpa_4, dtbchk_4, fail_rate, risk_prob
    )

    recommendations = engine.recommend(student, top_k=top_k, candidate_limit=80)
    if recommendations and explain_recommendations is not None:
        recommendations = explain_recommendations(student, recommendations, courses_by_id)

    return recommendations, engine_status


# -----------------------------------------------------------------------------
# 5C. VECTOR SIMILARITY RECOMMENDATION + XAI
# -----------------------------------------------------------------------------
def generate_vector_personalized_recommendations(
    selected_mssv, df_students_source, major_key, batch_code, top_k=8, group=None
):
    if not HAS_VECTOR_RECSYS or df_students_source is None or df_students_source.empty:
        return [], {}, "VECTOR_PACKAGE_NOT_AVAILABLE"

    try:
        raw_courses = get_program_courses(major_key, batch_code)
        courses = []
        for c in raw_courses:
            cc = dict(c)
            cc["code"] = str(cc.get("code", "")).strip().upper()
            cc["name"] = str(cc.get("name", cc["code"])).strip()
            cc["prereq"] = [str(x).strip().upper() for x in cc.get("prereq", [])]
            courses.append(cc)

        service = VectorRecommendationService(
            df_students=df_students_source,
            courses=courses,
            top_neighbors=10,
            min_common_courses=3,
        )
        recommendations, diagnostics = service.recommend(
            student_id=str(selected_mssv),
            top_k=top_k,
            group=group,
        )
        return recommendations, diagnostics, "VECTOR_KNN+XAI"
    except Exception as exc:
        return [], {"error": str(exc)}, "VECTOR_ERROR"


# -----------------------------------------------------------------------------
# 5D. KNN CÁ NHÂN HÓA TRỰC TIẾP: VECTOR ĐIỂM TOÀN BỘ HỌC PHẦN + LỘ TRÌNH
# -----------------------------------------------------------------------------
PATHWAY_KEYWORDS = {
    "An ninh mạng": ["AN NINH", "MẬT MÃ", "TẤN CÔNG", "MÃ ĐỘC", "PHÁP Y", "BẢO MẬT", "CT211", "CT485", "CT486", "CT490", "CT491", "CT492"],
    "Mạng & Hạ tầng": ["MẠNG", "QUẢN TRỊ MẠNG", "LẬP TRÌNH MẠNG", "ĐÁM MÂY", "CT112", "CT212", "CT221", "CT233", "CT335", "CT226", "CT439"],
    "Phát triển phần mềm & Web": ["LẬP TRÌNH", "WEB", "CƠ SỞ DỮ LIỆU", "HỆ THỐNG", "CT101", "CT176", "CT177", "CT180", "CT188", "CT428"],
    "AI & Dữ liệu": ["TRÍ TUỆ NHÂN TẠO", "MÁY HỌC", "DỮ LIỆU", "CT190", "CT294"],
}

def _score_value(value):
    if pd.isna(value):
        return np.nan
    try:
        x = float(value)
        return x if 0 <= x <= 10 else np.nan
    except (TypeError, ValueError):
        return np.nan

def _course_pathway(course):
    hay = f"{course.get('code','')} {course.get('name','')}".upper()
    scores = {name: sum(k in hay for k in keys) for name, keys in PATHWAY_KEYWORDS.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Nền tảng & liên ngành"

def _cosine_common(a, b, common):
    if len(common) == 0: return 0.0
    x = np.asarray([a[c] for c in common], dtype=float)
    y = np.asarray([b[c] for c in common], dtype=float)
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 1e-12 else 0.0

def generate_cross_cohort_recommendations(
    selected_mssv, df_students_source, major_key, batch_code, passed_courses,
    failed_courses, earned_credits, top_k=8, top_neighbors=12,
    min_common_courses=3, min_confidence=0.35
):
    if df_students_source is None or df_students_source.empty:
        return [], {"error": "Dữ liệu sinh viên trống."}, "VECTOR_ERROR"
    data=df_students_source.copy(); data["MSSV"]=data["MSSV"].astype(str)
    target_rows=data[data["MSSV"]==str(selected_mssv)]
    if target_rows.empty: return [], {"error":"Không tìm thấy sinh viên được chọn."}, "VECTOR_ERROR"
    courses=get_program_courses(major_key,batch_code); course_map={str(c["code"]).upper():c for c in courses}
    score_cols={c:f"DIEM_{c}" for c in course_map if f"DIEM_{c}" in data.columns}
    if not score_cols: return [], {"error":"Không tìm thấy cột DIEM_<MÃ_MÔN>."}, "VECTOR_ERROR"
    target=target_rows.iloc[0]; tv={c:_score_value(target[col]) for c,col in score_cols.items()}; learned={c for c,v in tv.items() if not pd.isna(v)}
    target_batch=extract_batch_code(str(selected_mssv)); target_major=auto_detect_major_from_row(target)[0]
    pool=[]
    for _,row in data[data["MSSV"]!=str(selected_mssv)].iterrows():
        vec={c:_score_value(row[col]) for c,col in score_cols.items()}; common=[c for c in learned if not pd.isna(vec[c])]
        if len(common)<min_common_courses: continue
        sim=_cosine_common(tv,vec,common)
        if not np.isfinite(sim) or sim<=0: continue
        nb=extract_batch_code(str(row.get("MSSV",""))); nm=auto_detect_major_from_row(row)[0]
        same_major=(nm==target_major); batch_distance=abs(int(nb[1:])-int(target_batch[1:]))
        overlap=len(common)/max(1,len(learned)); confidence=0.55*max(0,sim)+0.30*min(1,len(common)/8)+0.15*overlap
        if same_major: confidence=min(1,confidence+0.08)
        if confidence<min_confidence: continue
        pool.append({"student_id":str(row["MSSV"]),"batch":nb,"major":nm,"same_major":same_major,"similarity":float(sim),"confidence":float(confidence),"common_courses":common,"vector":vec,"batch_distance":batch_distance})
    pool.sort(key=lambda n:(n["same_major"],n["confidence"],n["similarity"],-n["batch_distance"]),reverse=True)
    
    neighbors=[]; per_batch={}
    for n in pool:
        if len(neighbors)>=top_neighbors: break
        if per_batch.get(n["batch"],0)>=2: continue
        neighbors.append(n); per_batch[n["batch"]]=per_batch.get(n["batch"],0)+1
    for n in pool:
        if len(neighbors)>=top_neighbors: break
        if n not in neighbors: neighbors.append(n)
    if not neighbors: return [], {"target_batch":target_batch,"error":"Không đủ hàng xóm đạt minimum common courses và confidence."}, "VECTOR_ERROR"
    candidates=[]
    for code,course in course_map.items():
        if code in learned or code in failed_courses or course.get("type")=="Điều kiện": continue
        if any(p not in passed_courses for p in course.get("prereq",[])) or earned_credits<course.get("min_credits",0): continue
        evidence=[(n,float(n["vector"][code])) for n in neighbors if not pd.isna(n["vector"].get(code))]
        if len(evidence)<2: continue
        weights=np.array([max(.001,n["confidence"]*(1.08 if n["same_major"] else .85)) for n,g in evidence]); grades=np.array([g for n,g in evidence])
        predicted=float(np.average(grades,weights=weights)); support=len(evidence)/len(neighbors); conf=float(np.average([n["confidence"] for n,g in evidence],weights=weights))
        pathway=_course_pathway(course); priority=.35 if course.get("type")=="Bắt buộc" else .15
        score=.58*(predicted/10)+.22*conf+.12*support+.08*priority
        candidates.append({"course_id":code,"course_name":course["name"],"credits":course["credits"],"predicted_score":predicted,"confidence":conf,"support_count":len(evidence),"support_ratio":support,"pathway":pathway,"recommendation_score":score,"supporting_neighbors":[{"student_id":n["student_id"],"batch":n["batch"],"similarity":n["similarity"],"confidence":n["confidence"],"common_courses":n["common_courses"],"grade":g} for n,g in evidence],"reasons":[f"Dự đoán {predicted:.2f}/10 từ {len(evidence)} hàng xóm chéo khóa",f"Confidence {conf:.2f}; hỗ trợ {support:.0%}",f"Ưu tiên cùng ngành {target_major}, đa dạng B21–B26"]})
    summary={}
    for c in candidates:
        d=summary.setdefault(c["pathway"],[]); d.append(c)
    pathways=[]
    for name,cs in summary.items():
        ps=float(.65*np.mean([c["predicted_score"] for c in cs])/10+.20*np.mean([c["confidence"] for c in cs])+.15*np.mean([c["support_ratio"] for c in cs]))
        pathways.append({"pathway":name,"pathway_score":ps,"avg_predicted_score":float(np.mean([c["predicted_score"] for c in cs])),"course_count":len(cs)})
    pathways.sort(key=lambda x:x["pathway_score"],reverse=True); ranks={p["pathway"]:i+1 for i,p in enumerate(pathways)}
    for c in candidates: c["pathway_rank"]=ranks[c["pathway"]]; c["final_score"]=c["recommendation_score"]+.12/c["pathway_rank"]
    candidates.sort(key=lambda x:x["final_score"],reverse=True)
    for i,c in enumerate(candidates,1): c["rank"]=i
    diagnostics={"target_batch":target_batch,"target_major":target_major,"target_vector":{c:float(v) for c,v in tv.items() if not pd.isna(v)},"neighbors":neighbors,"pathways":pathways,"learned_course_count":len(learned),"candidate_count":len(candidates),"neighbor_batches":sorted(set(n["batch"] for n in neighbors))}
    return candidates[:top_k],diagnostics,"CROSS_COHORT_KNN_PATHWAY"

# -----------------------------------------------------------------------------
# 5E. PHÂN TÍCH THEO KHÓA / COHORT ANALYTICS CHO ADMIN & CỐ VẤN
# -----------------------------------------------------------------------------
BATCH_LABELS = {f"B{i}": f"B{i} — Khóa {i+26}" for i in range(21, 31)}


def _batch_label(batch):
    b = str(batch or "").strip().upper()
    return BATCH_LABELS.get(b, b or "Chưa xác định")


@st.cache_data(show_spinner=False)
def prepare_cohort_dataframe(df_students_source):
    """Chuẩn hóa mỗi sinh viên thành một bản ghi phục vụ phân tích theo khóa."""
    if df_students_source is None or df_students_source.empty:
        return pd.DataFrame()
    rows = []
    registry = load_curriculum_registry()
    active_credit_targets = {}
    for item in registry.values():
        if not isinstance(item, dict) or not item.get("active"):
            continue
        key = (str(item.get("major", "")).upper(), str(item.get("batch", "")).upper())
        total = (item.get("totals") or {}).get("total_credits")
        try:
            if total is not None:
                active_credit_targets[key] = float(total)
        except (TypeError, ValueError):
            pass

    mssv_col = "MSSV" if "MSSV" in df_students_source.columns else df_students_source.columns[0]
    for _, row in df_students_source.iterrows():
        mssv = str(row.get(mssv_col, "")).strip()
        major_key, class_code = auto_detect_major_from_row(row)
        batch, resolution = resolve_student_batch(mssv, class_code, row.get("NganhHoc", ""))
        if not batch:
            batch = extract_batch_code(mssv)
        sem_df = parse_student_semester_history(row)
        latest = sem_df.iloc[-1] if not sem_df.empty else {}
        cpa4 = float(latest.get("ĐTBCTL (Thang 4)", 0.0) or 0.0)
        cpa10 = float(latest.get("ĐTBCTL (Thang 10)", 0.0) or 0.0)
        earned = float(latest.get("Số TC Tích lũy (Đạt)", 0.0) or 0.0)
        debt = float(latest.get("Số TC Nợ đọng", 0.0) or 0.0)
        fail_rate = float(latest.get("fail_rate_num", 0.0) or 0.0)
        semester_no = int(latest.get("hk_num", len(sem_df) or 0) or 0)
        target_credits = active_credit_targets.get((major_key, batch), 161.0)
        progress_pct = max(0.0, min(100.0, earned / target_credits * 100.0)) if target_credits else np.nan
        expected_credits = min(target_credits, semester_no * 18.0) if semester_no else 0.0
        progress_gap = max(0.0, expected_credits - earned)

        # Quy tắc sàng lọc nội bộ, dùng để hỗ trợ cố vấn; không phải kết luận chính thức của CTU.
        if cpa4 < 2.0 or debt >= 6 or progress_gap >= 18:
            support_level = "CẦN CAN THIỆP"
        elif cpa4 < 2.5 or debt > 0 or progress_gap >= 9:
            support_level = "CẦN THEO DÕI"
        else:
            support_level = "ỔN ĐỊNH"

        rows.append({
            "MSSV": mssv,
            "MaLop": class_code,
            "Nganh": PROGRAM_METADATA.get(major_key, {}).get("name", major_key),
            "MajorKey": major_key,
            "Khóa": batch,
            "Tên khóa": _batch_label(batch),
            "HK gần nhất": semester_no,
            "ĐTBCTL 4": cpa4,
            "ĐTBCTL 10": cpa10,
            "TC tích lũy": earned,
            "TC nợ": debt,
            "Tỷ lệ rớt môn (%)": fail_rate,
            "CTĐT mục tiêu TC": target_credits,
            "Tiến độ CTĐT (%)": progress_pct,
            "Khoảng cách tiến độ (TC)": progress_gap,
            "Mức hỗ trợ": support_level,
            "Batch resolution": resolution.get("status", "UNKNOWN") if isinstance(resolution, dict) else "UNKNOWN",
        })
    return pd.DataFrame(rows)


def _cohort_semester_dataframe(df_students_source, selected_batch="Tất cả các khóa", selected_major="Tất cả ngành"):
    if df_students_source is None or df_students_source.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df_students_source.iterrows():
        mssv = str(row.get("MSSV", "")).strip()
        major_key, class_code = auto_detect_major_from_row(row)
        batch, _ = resolve_student_batch(mssv, class_code, row.get("NganhHoc", ""))
        if not batch:
            batch = extract_batch_code(mssv)
        if selected_batch != "Tất cả các khóa" and batch != selected_batch:
            continue
        major_name = PROGRAM_METADATA.get(major_key, {}).get("name", major_key)
        if selected_major != "Tất cả ngành" and major_name != selected_major:
            continue
        sem_df = parse_student_semester_history(row)
        for _, sem in sem_df.iterrows():
            rows.append({
                "Khóa": _batch_label(batch),
                "Major": major_name,
                "HK": int(sem.get("hk_num", 0) or 0),
                "ĐTBHK 4": float(sem.get("ĐTBCHK (Thang 4)", 0.0) or 0.0),
                "ĐTBCTL 4": float(sem.get("ĐTBCTL (Thang 4)", 0.0) or 0.0),
                "TC đạt HK": float(sem.get("Số TC Tích lũy (Đạt)", 0.0) or 0.0),
                "TC nợ": float(sem.get("Số TC Nợ đọng", 0.0) or 0.0),
            })
    return pd.DataFrame(rows)


def render_cohort_analytics(df_students_source):
    st.title("📊 Phân tích tiến độ học tập theo khóa")
    st.caption("Dashboard dành cho Admin/Cố vấn học tập. Các cờ hỗ trợ là quy tắc sàng lọc nội bộ để ưu tiên theo dõi, không phải kết luận học vụ chính thức.")

    prepared = prepare_cohort_dataframe(df_students_source)
    if prepared.empty:
        st.warning("Chưa có dữ liệu sinh viên để phân tích.")
        return

    batches = sorted([x for x in prepared["Khóa"].dropna().unique() if x], key=lambda x: int(x[1:]) if str(x)[1:].isdigit() else 999)
    majors = sorted(prepared["Nganh"].dropna().unique().tolist())
    f1, f2 = st.columns([1, 1])
    with f1:
        batch_label = st.selectbox("Chọn khóa", ["Tất cả các khóa"] + [_batch_label(x) for x in batches], key="cohort_batch_filter")
    with f2:
        selected_major = st.selectbox("Chọn ngành", ["Tất cả ngành"] + majors, key="cohort_major_filter")
    selected_batch = "Tất cả các khóa" if batch_label == "Tất cả các khóa" else next((b for b in batches if _batch_label(b) == batch_label), batch_label)

    view = prepared.copy()
    if selected_batch != "Tất cả các khóa":
        view = view[view["Khóa"] == selected_batch]
    if selected_major != "Tất cả ngành":
        view = view[view["Nganh"] == selected_major]

    if view.empty:
        st.info("Không có sinh viên khớp bộ lọc hiện tại.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sinh viên", f"{len(view):,}")
    k2.metric("ĐTBCTL 4 trung bình", f"{view['ĐTBCTL 4'].mean():.2f}")
    k3.metric("TC tích lũy TB", f"{view['TC tích lũy'].mean():.1f}")
    k4.metric("TC nợ TB", f"{view['TC nợ'].mean():.1f}")
    k5.metric("Cần can thiệp", f"{(view['Mức hỗ trợ'] == 'CẦN CAN THIỆP').sum():,}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    if selected_batch == "Tất cả các khóa":
        st.markdown("#### 🏫 So sánh các khóa")
        batch_summary = (
            view.groupby("Tên khóa", as_index=False)
            .agg(
                **{
                    "Sinh viên": ("MSSV", "count"),
                    "ĐTBCTL 4 TB": ("ĐTBCTL 4", "mean"),
                    "TC tích lũy TB": ("TC tích lũy", "mean"),
                    "TC nợ TB": ("TC nợ", "mean"),
                }
            )
        )
        batch_summary["Tỷ lệ cần can thiệp (%)"] = view.groupby("Tên khóa")["Mức hỗ trợ"].apply(
            lambda x: (x.eq("CẦN CAN THIỆP").mean() * 100.0)
        ).reindex(batch_summary["Tên khóa"]).to_numpy()
        batch_summary["ĐTBCTL 4 TB"] = batch_summary["ĐTBCTL 4 TB"].round(2)
        batch_summary["TC tích lũy TB"] = batch_summary["TC tích lũy TB"].round(1)
        batch_summary["TC nợ TB"] = batch_summary["TC nợ TB"].round(1)
        batch_summary["Tỷ lệ cần can thiệp (%)"] = batch_summary["Tỷ lệ cần can thiệp (%)"].round(1)
        st.dataframe(batch_summary, use_container_width=True, hide_index=True)

    with c1:
        st.markdown("#### 📈 Phân bố ĐTBCTL")
        fig = px.histogram(view, x="ĐTBCTL 4", nbins=12, marginal="box")
        fig.update_layout(xaxis_title="ĐTBCTL (thang 4)", yaxis_title="Số sinh viên", bargap=0.08)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 🚦 Phân loại mức hỗ trợ")
        counts = view["Mức hỗ trợ"].value_counts().reindex(["ỔN ĐỊNH", "CẦN THEO DÕI", "CẦN CAN THIỆP"]).fillna(0).reset_index()
        counts.columns = ["Mức hỗ trợ", "Sinh viên"]
        fig = px.bar(counts, x="Mức hỗ trợ", y="Sinh viên")
        fig.update_layout(xaxis_title="", yaxis_title="Số sinh viên")
        st.plotly_chart(fig, use_container_width=True)

    sem_view = _cohort_semester_dataframe(df_students_source, selected_batch=selected_batch, selected_major=selected_major)
    if not sem_view.empty:
        st.markdown("#### 📚 Xu hướng theo học kỳ")
        trend = sem_view.groupby("HK", as_index=False).agg({"ĐTBHK 4": "mean", "ĐTBCTL 4": "mean", "TC nợ": "mean"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["HK"], y=trend["ĐTBHK 4"], mode="lines+markers", name="ĐTBHK TB"))
        fig.add_trace(go.Scatter(x=trend["HK"], y=trend["ĐTBCTL 4"], mode="lines+markers", name="ĐTBCTL TB"))
        fig.update_layout(xaxis_title="Học kỳ", yaxis_title="Điểm (thang 4)", yaxis_range=[0, 4])
        st.plotly_chart(fig, use_container_width=True)
        debt_trend = sem_view.groupby("HK", as_index=False)["TC nợ"].mean()
        fig_debt = px.line(debt_trend, x="HK", y="TC nợ", markers=True, title="TC nợ trung bình theo học kỳ")
        fig_debt.update_layout(xaxis_title="Học kỳ", yaxis_title="TC nợ TB")
        st.plotly_chart(fig_debt, use_container_width=True)

    st.markdown("#### 🚨 Danh sách sinh viên cần ưu tiên hỗ trợ")
    support_view = view[view["Mức hỗ trợ"] != "ỔN ĐỊNH"].copy()
    support_view = support_view.sort_values(["Mức hỗ trợ", "TC nợ", "ĐTBCTL 4"], ascending=[True, False, True])
    display_support = support_view[["MSSV", "MaLop", "Nganh", "Khóa", "HK gần nhất", "ĐTBCTL 4", "TC tích lũy", "TC nợ", "Tiến độ CTĐT (%)", "Khoảng cách tiến độ (TC)", "Mức hỗ trợ"]]
    st.dataframe(display_support, use_container_width=True, hide_index=True)

    st.markdown("#### 📄 Dữ liệu phục vụ báo cáo")
    report = view.sort_values(["Mức hỗ trợ", "ĐTBCTL 4"], ascending=[True, True]).copy()
    st.download_button(
        "⬇️ Tải CSV thống kê sinh viên",
        data=report.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"bao_cao_{selected_batch if selected_batch != 'Tất cả các khóa' else 'tat_ca_khoa'}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Cột 'Mức hỗ trợ' và 'Khoảng cách tiến độ' được tạo từ quy tắc sàng lọc nội bộ của hệ thống, giúp cố vấn ưu tiên xem xét hồ sơ.")

# -----------------------------------------------------------------------------
# 5E. ĐÁNH GIÁ TIẾN ĐỘ TỐT NGHIỆP + XẾP LOẠI HỌC TẬP CTU + TƯ VẤN XAI
# -----------------------------------------------------------------------------
def evaluate_graduation_progress(on_time_prob):
    pct = float(max(0.0, min(1.0, on_time_prob)) * 100.0)
    if pct < 40:
        return {
            "title": "ĐÁNH GIÁ TIẾN ĐỘ TỐT NGHIỆP",
            "value": "NGUY CƠ RẤT CAO",
            "description": "Các chỉ báo hiện tại cho thấy nguy cơ chậm tiến độ ở mức cao; cần rà soát môn nợ, tiên quyết và tải học tập.",
            "level": "critical",
            "bg": "background-color: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444;",
        }
    if pct < 50:
        return {
            "title": "ĐÁNH GIÁ TIẾN ĐỘ TỐT NGHIỆP",
            "value": "CHƯA AN TOÀN",
            "description": "Cảnh báo nguy cơ không tốt nghiệp đúng tiến độ. Cần ưu tiên xử lý môn nợ và các học phần tiên quyết.",
            "level": "warning",
            "bg": "background-color: rgba(249, 115, 22, 0.15); border-left: 5px solid #f97316;",
        }
    if pct <= 60:
        return {
            "title": "ĐÁNH GIÁ TIẾN ĐỘ TỐT NGHIỆP",
            "value": "ỔN ĐỊNH, CẦN THEO DÕI",
            "description": "Tiến độ hiện có thể duy trì, nhưng vẫn cần kiểm soát nợ môn, tín chỉ tích lũy và các điều kiện tốt nghiệp.",
            "level": "moderate",
            "bg": "background-color: rgba(234, 179, 8, 0.15); border-left: 5px solid #eab308;",
        }
    return {
        "title": "ĐÁNH GIÁ TIẾN ĐỘ TỐT NGHIỆP",
        "value": "🟢 AN TOÀN, TỐT NGHIỆP ĐÚNG TIẾN ĐỘ",
        "description": "Bạn đang ở vùng an toàn. Hãy duy trì kết quả học tập và hoàn thành đầy đủ các điều kiện tốt nghiệp.",
        "level": "safe",
        "bg": "background-color: rgba(34, 197, 94, 0.15); border-left: 5px solid #22c55e;",
    }


def classify_ctu_academic_performance(gpa_4):
    gpa = float(max(0.0, min(4.0, gpa_4)))
    if gpa >= 3.6:
        label, icon, desc = "XUẤT SẮC", "🏆", "Kết quả học tập xuất sắc"
    elif gpa >= 3.2:
        label, icon, desc = "GIỎI", "🌟", "Kết quả học tập giỏi"
    elif gpa >= 2.5:
        label, icon, desc = "KHÁ", "•", "Kết quả học tập khá"
    elif gpa >= 2.0:
        label, icon, desc = "TRUNG BÌNH", "•", "Kết quả học tập trung bình"
    elif gpa >= 1.0:
        label, icon, desc = "YẾU", "•", "Kết quả học tập yếu, cần cải thiện"
    else:
        label, icon, desc = "KÉM", "•", "Kết quả học tập kém, cần can thiệp sớm"
    return {"label": label, "icon": icon, "description": desc, "gpa": gpa}


def build_xai_consultation(
    cpa_4, dtbchk_4, tc_earned, tc_no_dong, fail_rate,
    progress_assessment, academic_classification,
    recommendations, diagnostics, rec_results, fl_data, gdqp_data
):
    lines = []
    lines.append("### TƯ VẤN TỔNG QUAN")
    lines.append(
        f"**Xếp loại học tập hiện tại:** {academic_classification['icon']} "
        f"**{academic_classification['label']}** (ĐTBCTL {cpa_4:.2f}/4.0; ĐTBCHK gần nhất {dtbchk_4:.2f}/4.0)."
    )
    lines.append(
        f"**Tiến độ tốt nghiệp:** {progress_assessment['value']}. "
        f"{progress_assessment['description']}"
    )

    factors = []
    if tc_no_dong > 0:
        factors.append(f"{tc_no_dong} TC nợ đọng đang làm tăng nguy cơ chậm tiến độ")
    if fail_rate > 0:
        factors.append(f"tỷ lệ rớt môn gần nhất {fail_rate:.1f}%")
    if cpa_4 < 2.5:
        factors.append(f"GPA {cpa_4:.2f}/4.0 chưa ở mức Khá")
    if not factors:
        factors.append("chưa có chỉ báo học vụ tiêu cực nổi bật từ dữ liệu hiện tại")
    lines.append("**Các yếu tố ảnh hưởng chính:** " + "; ".join(factors) + ".")

    neighbors = diagnostics.get("neighbors", []) if isinstance(diagnostics, dict) else []
    batches = diagnostics.get("neighbor_batches", []) if isinstance(diagnostics, dict) else []
    improve = rec_results.get("improve", []) if isinstance(rec_results, dict) else []
    if improve:
        lines.append("**Can thiệp trước tiên:** ưu tiên hoàn thành/học lại các môn đang nợ hoặc cải thiện trước khi mở rộng tải tín chỉ mới.")
    elif progress_assessment["level"] in {"critical", "warning"}:
        lines.append("**Can thiệp trước tiên:** giảm rủi ro bằng cách xử lý học phần có nguy cơ, kiểm tra tiên quyết và xây dựng tải học phần khả thi.")

    if not fl_data.get("is_qualified", False):
        lines.append("**Điều kiện tốt nghiệp:** Ngoại ngữ chưa hoàn tất, cần đưa vào kế hoạch học tập.")
    if not gdqp_data.get("is_qualified", False):
        lines.append("**Điều kiện tốt nghiệp:** GDQP-AN chưa hoàn tất, cần theo dõi tiến độ song song.")

    return "\n\n".join(lines)


# -----------------------------------------------------------------------------
# 6. HÀM GỌI GROQ API (LLAMA 3.3 70B MIỄN PHÍ) ĐỂ TƯ VẤN HỌC TẬP TỰ ĐỘNG
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_chatgpt_advice(
    mssv,
    class_code,
    major_info,
    major_key,
    batch_code,
    meta,
    cpa_10,
    cpa_4,
    dtbchk_4,
    tc_earned,
    tc_no_dong,
    passed_courses_list,
    failed_courses_list,
    low_grade_courses_list,
    rec_mandatory,
    rec_elective,
    rec_projects,
    gdtc_summary_list,
    fl_data,
    gdqp_data,
    api_key,
    user_question="",
    prompt_text="",
    prompt_version=""
):
    active_key = str(api_key or "").strip() or load_groq_api_key()
    active_prompt = str(prompt_text or load_prompt_template()).strip()
    active_prompt_version = str(prompt_version or get_prompt_version())

    if not active_key:
        return "⚠️ Chưa cấu hình Groq API Key trong hệ thống."

    failed_str = ", ".join(failed_courses_list) if failed_courses_list else "Không có môn rớt"
    low_grade_str = ", ".join(low_grade_courses_list) if low_grade_courses_list else "Không có môn điểm thấp (D/D+)"
    
    mandatory_next = [f"{c['code']} - {c['name']} ({c['credits']} TC)" for c in rec_mandatory]
    mandatory_str = ", ".join(mandatory_next[:6]) if mandatory_next else "Đã hoàn thành hầu hết các môn bắt buộc khả thi"

    elective_next = [f"{c['code']} - {c['name']} ({c['credits']} TC)" for c in rec_elective]
    elective_str = ", ".join(elective_next[:5]) if elective_next else "Không có gợi ý tự chọn ưu tiên"

    projects_next = [f"{c['code']} - {c['name']} ({c['credits']} TC)" for c in rec_projects]
    projects_str = ", ".join(projects_next) if projects_next else "Chưa đủ điều kiện đăng ký Niên luận/Thực tập/Luận văn"

    gdtc_str = " | ".join([f"{g['sport']}: {g['status']}" for g in gdtc_summary_list])
    
    fl_qual = (
        "Đã đủ điều kiện tốt nghiệp (≥10 TC Ngoại ngữ hoặc Miễn điểm M)" 
        if fl_data.get("is_qualified") else f"Chưa đủ (Hiện có {fl_data.get('total_credits')}/10 TC Ngoại ngữ)"
    )

    gdqp_qual = (
        f"Đã đạt chuẩn ({gdqp_data.get('completed_count')}/{gdqp_data.get('total_courses')} học phần)"
        if gdqp_data.get("is_qualified") else f"Chưa đạt (Mới hoàn thành {gdqp_data.get('completed_count')}/{gdqp_data.get('total_courses')} học phần)"
    )

    prompt_content = f"""
{active_prompt}

[Prompt-Version: {active_prompt_version}]

Dựa vào dữ liệu chính thức dưới đây, hãy đưa ra phân tích, chẩn đoán và lời tư vấn chuyên sâu cho sinh viên:

--- CƠ SỞ TƯ VẤN (KNOWLEDGE BASE) ---
Vui lòng tham chiếu các quy định và thông tin từ các nguồn chính thức sau của CTU để đưa ra lời khuyên:
1. Quy chế đào tạo học vụ CTU: https://daa.ctu.edu.vn/quy-che-quy-dinh/quy-che-dao-tao (Chú ý các quy định về cảnh báo học vụ, giới hạn tín chỉ, điều kiện tốt nghiệp).
2. Thông tin CTĐT Ngành {major_info['name']}: [ĐIỀN LINK CHƯƠNG TRÌNH ĐÀO TẠO CỦA NGÀNH TẠI ĐÂY]
3. Cổng thông tin tín chỉ sinh viên: https://htql.ctu.edu.vn/

--- THÔNG TIN HỌC TẬP SINH VIÊN ---
- MSSV: {mssv} | Mã Lớp: {class_code} | Ngành: {major_info['name']} ({major_info['code']})
- Khóa học: Khóa {batch_code} ({meta['khoa']}) - {meta['department']}
- GPA Tích lũy (Thang 4): {cpa_4:.2f} / 4.0 (Tương đương Thang 10: {cpa_10:.2f})
- ĐTBCHK gần nhất: {dtbchk_4:.2f} / 4.0
- Tổng Tín chỉ Đạt tích lũy toàn khóa: {tc_earned} TC
- Số Tín chỉ Nợ đọng: {tc_no_dong} TC
- Danh sách môn rớt/chưa đạt: {failed_str}
- Danh sách môn điểm thấp (D/D+) cần cải thiện: {low_grade_str}
- Tiến độ Ngoại ngữ chuẩn ra trường (10 TC): {fl_qual}
- Tiến độ Giáo dục Quốc phòng - An ninh (GDQP-AN): {gdqp_qual}
- Tiến độ Giáo dục thể chất (GDTC): {gdtc_str}
- Môn bắt buộc mở khóa có thể đăng ký HK tới: {mandatory_str}
- Môn tự chọn mở khóa: {elective_str}
- Đồ án / Niên luận / Thực tập khả thi: {projects_str}

--- CÂU HỎI CỦA SINH VIÊN ---
{user_question.strip() if user_question and user_question.strip() else "Hãy chủ động tư vấn tổng quan và kế hoạch học tập phù hợp nhất cho sinh viên."}

--- NGUỒN CTĐT PDF ĐANG GẮN CHO SINH VIÊN ---
{build_curriculum_ai_context(major_key, batch_code)}

--- PHÂN TÍCH ĐỒ THỊ TIÊN QUYẾT ---
{build_curriculum_graph_ai_context(major_key, batch_code, passed_courses_list, failed_courses_list, tc_earned)}

--- LỘ TRÌNH HỌC TỐI ƯU (MÔ PHỎNG TỪ GRAPH ENGINE) ---
{build_optimal_path_ai_context(major_key, batch_code, passed_courses_list, failed_courses_list, tc_earned)}

--- TIẾN ĐỘ TỐT NGHIỆP & KỊCH BẢN ---
{build_graduation_scenario_ai_context(major_key, batch_code, passed_courses_list, failed_courses_list, tc_earned)}

{build_registration_ai_context(major_key, batch_code, passed_courses_list, failed_courses_list, tc_earned, term="CURRENT")}

--- COURSE OFFERING & SEMESTER REALITY ---
{build_semester_reality_ai_context(major_key, batch_code, passed_courses_list, failed_courses_list, tc_earned, term="CURRENT")}

--- CÂU LỆNH YÊU CẦU ---
1. Trả lời trực tiếp câu hỏi của sinh viên (nếu có)
2. ĐÁNH GIÁ TỔNG QUAN & CHẨN ĐOÁN (Giọng điệu: Thấu cảm, Khách quan, Sắc bén)
- Khen ngợi điểm mạnh (nếu có) dựa trên ĐTBCHK hoặc TC tích lũy.
- Phân tích thẳng thắn điểm yếu và rủi ro: Chỉ ra rõ nguy cơ cảnh báo học vụ hoặc trễ tiến độ tốt nghiệp dựa trên 'Số TC Nợ đọng' và 'ĐTBCTL'. (Trích dẫn quy chế đào tạo tín chỉ của CTU về ngưỡng điểm tối thiểu nếu cần).

3. LỘ TRÌNH ĐĂNG KÝ HỌC PHẦN HỌC KỲ TIẾP THEO (Giọng điệu: Logic, Chiến lược)
Dựa trên nguyên tắc ưu tiên bắt buộc của CTU, hãy đề xuất danh sách môn học cụ thể cho học kỳ tới.
- Mức tín chỉ khuyến nghị: Gợi ý tổng số tín chỉ đăng ký hợp lý nhất (từ 15-18 TC để đảm bảo sức học, hoặc ít hơn nếu sinh viên đang có ĐTBCHK rất thấp).
- Thứ tự xếp học phần: 
  + Rút từ danh sách "Môn rớt/chưa đạt" (Giải quyết nợ đọng, gỡ nút thắt tiên quyết).
  + Rút từ danh sách "Môn cải thiện điểm thấp" (Kéo nhanh CPA).
  + Rút từ danh sách "Môn bắt buộc mở khóa khả thi".
  + Rút từ danh sách "Môn tự chọn" & Các môn Điều kiện (Ngoại ngữ, GDQP, GDTC).
*Lưu ý: Nêu rõ Tên môn, Mã môn và lý do chọn môn đó. quy định không ghi nhưng thường sinh viên không nên đăng ký Niên luận cơ cở, niên luận ngành, thực tập hay luận văn cùng 1 học kỳ, nếu có ý định muốn đăng ký thì nên khuyên hỏi ý kiến giáo viên cố vấn

4. TƯ VẤN ĐIỀU KIỆN TỐT NGHIỆP & CHUYÊN NGÀNH (Giọng điệu: Định hướng tương lai)
- Nhắc nhở tiến độ các môn điều kiện (Ngoại ngữ, GDQP, GDTC). 
- Phân tích ngắn gọn tính khả thi của việc đăng ký Đồ án/Niên luận/Thực tập (nếu có trong danh sách).

5. LỜI KHUYÊN HÀNH ĐỘNG CÁ NHÂN HÓA 
- Đưa ra 2-3 lời khuyên thực tế về phương pháp học tập, quản lý thời gian, hoặc việc liên hệ Giảng viên cố vấn học tập tại {meta['department']}.

--- YÊU CẦU RÀNG BUỘC (QUAN TRỌNG) ---
- Tuyệt đối KHÔNG bịa đặt mã môn học, tên môn học, hoặc các quy định không có ở Đại học Cần Thơ.
- Chỉ tư vấn dựa trên dữ liệu các biến được cung cấp.
- Trình bày dạng Markdown mạch lạc, in đậm các ý chính, sử dụng emoji phù hợp để tạo thiện cảm.
- Câu trả lời phải nhất quán không quá sai lệch thông tin giữa những lần trả lời
"""

    if not (HAS_OPENAI and hasattr(openai, "OpenAI")):
        return "🚨 Thư viện `openai` chưa được cài đặt."

    try:
        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=active_key
        )

        available_model_ids = set()
        try:
            models_page = client.models.list()
            available_model_ids = {m.id for m in models_page.data}
        except Exception:
            pass

        priority_models = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "groq/compound-mini",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ]

        if available_model_ids:
            candidates = [m for m in priority_models if m in available_model_ids]
            if not candidates:
                candidates = priority_models
        else:
            candidates = priority_models

        last_exception = None
        used_model = None
        response_content = None

        for model_name in candidates:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": active_prompt},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.7,
                )
                used_model = model_name
                response_content = response.choices[0].message.content
                break
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                if "404" in err_str or "model_not_found" in err_str or "not_found" in err_str:
                    continue
                else:
                    raise e

        if response_content is None:
            raise last_exception if last_exception else Exception("Không có model khả dụng.")

        debug_header = f"🤖 *[Model đang dùng: `{used_model}`]*\n\n"
        return debug_header + response_content

    except Exception as e:
        return f"🚨 **Lỗi khi gọi Groq API**: {str(e)}"


# -----------------------------------------------------------------------------
# 7. LOAD MODEL & DATASET CACHE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = PROJECT_ROOT / "models" / "transformer_recsys_student.pth"
    model = None
    is_loaded = False

    if StudentTransformerRecSys is not None and HAS_TORCH:
        try:
            model = StudentTransformerRecSys(feature_dim=11, num_courses=50)
            if model_path.exists():
                model.load_state_dict(
                    torch.load(model_path, map_location=torch.device("cpu"))
                )
                model.eval()
                is_loaded = True
        except Exception as e:
            pass

    return model, is_loaded


model, is_model_loaded = load_trained_model()

# -----------------------------------------------------------------------------
# 8. KHU VỰC ADMIN DÀNH CHO QUẢN TRỊ VIÊN
# -----------------------------------------------------------------------------
admin_page = "Dashboard"
if is_admin():
    st.sidebar.markdown("### 🔑 **CẤU HÌNH GROQ AI API**")
    persisted_groq_key = load_persisted_groq_api_key()
    effective_groq_key = persisted_groq_key or HARDCODED_GROQ_API_KEY
    api_key_input = st.sidebar.text_input(
        "Groq API Key:",
        value=effective_groq_key,
        type="password",
        key="admin_groq_api_key_input",
        help="Key được lưu bền vững trong data/admin, không mất khi logout hoặc Streamlit restart."
    )
    key_save_col, key_test_col = st.sidebar.columns(2)
    with key_save_col:
        if st.button("💾 Lưu key", key="save_groq_key", use_container_width=True):
            if save_groq_api_key(api_key_input):
                st.sidebar.success("Đã lưu key.")
                st.rerun()
            else:
                st.sidebar.error("API key trống.")
    with key_test_col:
        if st.button("🧪 Kiểm tra", key="test_groq_key", use_container_width=True):
            ok, msg = test_groq_api_key(api_key_input or effective_groq_key)
            (st.sidebar.success if ok else st.sidebar.error)(msg)
    if persisted_groq_key:
        if st.sidebar.button("🗑️ Xóa key đã lưu", key="delete_groq_key", use_container_width=True):
            delete_groq_api_key()
            st.sidebar.success("Đã xóa key lưu cục bộ.")
            st.rerun()
        st.sidebar.caption("✅ Groq key đang được lưu bền vững.")
    elif HARDCODED_GROQ_API_KEY:
        st.sidebar.caption("ℹ️ Đang dùng fallback từ biến môi trường GROQ_API_KEY.")
    else:
        st.sidebar.caption("⚠️ Chưa có Groq API key.")

    st.sidebar.success("Quyền quản trị đang bật")
    admin_page = st.sidebar.radio(
        "Khu vực Quản trị",
        ["Dashboard", "📊 Phân tích theo khóa", "⚙️ Quản trị hệ thống"],
        key="admin_area"
    )

if is_admin() and admin_page == "⚙️ Quản trị hệ thống":
    st.title("⚙️ Quản trị hệ thống StudentProgressAI")
    st.caption("Khu vực quản lý toàn bộ Tài khoản, Chương trình đào tạo và Duyệt chứng chỉ ngoại ngữ.")
    
    a1, a2, a3, a4 = st.tabs(["🤖 Prompt Groq AI", "📚 Chương trình đào tạo", "👥 Quản lý tài khoản", "📜 Duyệt chứng chỉ ngoại ngữ"])
    
    # ---------------- TAB 1: PROMPT GROQ ----------------
    with a1:
        st.subheader("Prompt tư vấn Groq AI")
        prompt_cfg = load_prompt_config()
        st.caption(f"Phiên bản prompt hiện tại: `{prompt_cfg['version']}`" + (f" · Cập nhật: {prompt_cfg['updated_at']}" if prompt_cfg.get('updated_at') else ""))
        edited = st.text_area("Prompt hệ thống", value=prompt_cfg["prompt"], height=320, key="admin_prompt")
        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("💾 Lưu & áp dụng prompt", key="save_prompt", use_container_width=True):
                save_prompt_template(edited)
                st.session_state.pop("admin_prompt", None)
                st.success("Đã lưu. Prompt mới được áp dụng ngay cho lượt gọi AI tiếp theo và không dùng cache prompt cũ.")
                st.rerun()
        with pc2:
            if st.button("↩️ Khôi phục prompt mặc định", key="reset_prompt", use_container_width=True):
                save_prompt_template(DEFAULT_GROQ_PROMPT)
                st.session_state.pop("admin_prompt", None)
                st.success("Đã khôi phục prompt mặc định.")
                st.rerun()
        st.info("Prompt được lưu kèm version. Mỗi lần thay đổi version sẽ làm mất hiệu lực cache tư vấn trước đó, nên AI thực sự chạy theo prompt mới.")

    # ---------------- TAB 2: CHƯƠNG TRÌNH ĐÀO TẠO (THÊM / SỬA / XÓA) ----------------
    with a2:
        st.subheader("Quản lý Chương trình đào tạo (CTĐT)")
        registry = load_curriculum_registry()
        
        # 1. THÊM CTĐT
        st.markdown("#### ➕ Thêm / Nạp CTĐT mới")
        up = st.file_uploader("Chọn một hoặc nhiều file PDF chương trình đào tạo", type=["pdf"], accept_multiple_files=True, key="curriculum_pdf")
        if up and HAS_PYPDF and st.button("🔎 Phân tích các PDF", key="parse_curriculum"):
            try:
                parsed_list = [parse_curriculum_pdf(f.getvalue(), f.name) for f in up]
                st.session_state["parsed_curricula"] = parsed_list
                st.success(f"Đã phân tích {len(parsed_list)} file PDF. Chọn từng CTĐT để kiểm tra và lưu.")
            except Exception as e:
                st.error(f"Lỗi khi phân tích PDF: {e}")
                
        parsed_list = st.session_state.get("parsed_curricula", [])
        parsed = None
        if parsed_list:
            parsed_names = [f"{x.get('filename')} ({x.get('major')} - {x.get('batch') or 'chưa xác định'})" for x in parsed_list]
            parsed = parsed_list[st.selectbox("Chọn CTĐT đã phân tích", range(len(parsed_list)), format_func=lambda i: parsed_names[i])]
        if parsed:
            majors = {"MMT": "Mạng máy tính và Truyền thông dữ liệu", "ATTT": "An toàn thông tin"}
            st.markdown("#### 🔍 Kết quả phân tích & xác nhận")
            cmeta1, cmeta2, cmeta3, cmeta4 = st.columns(4)
            cmeta1.metric("Phương thức", parsed.get("parser_mode", "TEXT"))
            cmeta2.metric("Độ tin cậy", parsed.get("parse_confidence", "LOW"))
            cmeta3.metric("Học phần", parsed.get("course_count", 0))
            cmeta4.metric("Số trang", parsed.get("page_count", 0))

            st.info(
                f"Quyết định: **{parsed.get('decision_number') or 'chưa nhận diện'}** | "
                f"Ngày ban hành: **{parsed.get('decision_date') or 'chưa nhận diện'}** | "
                f"Khóa ứng viên: **{', '.join(parsed.get('batch_candidates', [])) or 'chưa xác định'}**"
            )
            if parsed.get("parser_warnings"):
                st.warning("\n".join(str(x) for x in parsed.get("parser_warnings", [])))
            if parsed.get("validation_issues"):
                st.warning(f"Phát hiện {len(parsed['validation_issues'])} vấn đề cần kiểm tra về quan hệ học phần.")

            maj = st.selectbox("Ngành", list(majors), index=list(majors).index(parsed["major"]) if parsed.get("major") in majors else 0)
            batches = [f"B{i}" for i in range(21, 27)]
            default_batch = parsed.get("batch") if parsed.get("batch") in batches else (parsed.get("batch_candidates") or ["B21"])[0]
            bat = st.selectbox("Khóa áp dụng chính thức", batches, index=batches.index(default_batch) if default_batch in batches else 0)
            var = st.selectbox("Biến thể", ["STANDARD", "CLC"], index=1 if parsed.get("variant") == "CLC" else 0)

            st.markdown("**Chỉnh sửa dữ liệu học phần trước khi lưu/kích hoạt**")
            editable_cols = ["code", "name", "credits", "type", "group", "elective_group", "prereq", "coreq", "min_credits", "semester_hint"]
            edit_df = pd.DataFrame(parsed.get("courses", []))
            for col in editable_cols:
                if col not in edit_df.columns:
                    edit_df[col] = [] if col in ("prereq","coreq") else ""
            edited_df = st.data_editor(
                edit_df[editable_cols],
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key=f"curriculum_editor_{parsed.get('curriculum_id')}"
            )

            total_credits = float(pd.to_numeric(edited_df["credits"], errors="coerce").fillna(0).sum()) if not edited_df.empty else 0
            expected_total = (parsed.get("totals") or {}).get("total_credits")
            st.caption(f"Tổng tín chỉ trích xuất hiện tại: {total_credits:g} TC" + (f" | PDF ghi: {expected_total} TC" if expected_total else ""))

            if st.button("💾 Lưu CTĐT vào kho (chưa kích hoạt)", key="save_curriculum_draft"):
                records = edited_df.to_dict("records")
                for r in records:
                    for list_col in ("prereq", "coreq"):
                        if isinstance(r.get(list_col), str):
                            r[list_col] = [x.strip().upper() for x in re.split(r"[,;]", r[list_col]) if x.strip()]
                        elif not isinstance(r.get(list_col), list):
                            r[list_col] = []
                    r["code"] = str(r.get("code","")).upper().strip()
                    r["name"] = str(r.get("name","")).strip()
                    r["credits"] = float(r.get("credits") or 0)
                    r["min_credits"] = int(float(r.get("min_credits") or 0))
                    r["source"] = parsed.get("filename")
                records, issues = _dedupe_and_validate_courses(records)
                parsed.update({
                    "major": maj, "batch": bat, "variant": var, "courses": records,
                    "course_count": len(records), "validation_issues": issues,
                    "curriculum_text": "\n".join(
                        f"{c['code']} | {c['name']} | {c['credits']} TC | {c['type']} | Nhóm: {c.get('group','')} | TQ: {', '.join(c.get('prereq',[])) or '-'}"
                        for c in records
                    ),
                    "active": False
                })
                registry[parsed["curriculum_id"]] = parsed
                save_curriculum_registry(registry)
                st.success(f"Đã lưu CTĐT vào kho với ID {parsed['curriculum_id']}. Hãy kiểm tra và kích hoạt khi sẵn sàng.")
                st.rerun()

            if st.button("✅ Lưu & Kích hoạt CTĐT này", key="activate_curriculum"):
                records = edited_df.to_dict("records")
                for r in records:
                    for list_col in ("prereq", "coreq"):
                        if isinstance(r.get(list_col), str):
                            r[list_col] = [x.strip().upper() for x in re.split(r"[,;]", r[list_col]) if x.strip()]
                        elif not isinstance(r.get(list_col), list):
                            r[list_col] = []
                    r["code"] = str(r.get("code","")).upper().strip()
                    r["name"] = str(r.get("name","")).strip()
                    r["credits"] = float(r.get("credits") or 0)
                    r["min_credits"] = int(float(r.get("min_credits") or 0))
                    r["source"] = parsed.get("filename")
                records, issues = _dedupe_and_validate_courses(records)
                parsed.update({
                    "major": maj, "batch": bat, "variant": var, "courses": records,
                    "course_count": len(records), "validation_issues": issues,
                    "active": True,
                    "activated_at": datetime.now().isoformat(),
                    "curriculum_text": "\n".join(
                        f"{c['code']} | {c['name']} | {c['credits']} TC | {c['type']} | Nhóm: {c.get('group','')} | TQ: {', '.join(c.get('prereq',[])) or '-'}"
                        for c in records
                    )
                })
                for old in registry.values():
                    if isinstance(old, dict) and old.get("curriculum_id") != parsed["curriculum_id"]:
                        if old.get("major")==maj and old.get("batch")==bat and old.get("variant","STANDARD")==var:
                            old["active"] = False
                registry[parsed["curriculum_id"]] = parsed
                save_curriculum_registry(registry)
                st.success(f"Đã lưu và kích hoạt CTĐT {maj} - {bat} - {var}. Các bản cùng mapping khác đã được tắt Active.")
                st.session_state.pop("parsed_curricula", None)
                st.rerun()

        st.markdown("---")
        # 2. DANH SÁCH / SỬA / XÓA CTĐT
        st.markdown("#### 📋 Danh sách & Chỉnh sửa / Xóa CTĐT")
        if registry:
            selected_ctdt = st.selectbox("Chọn CTĐT cần thao tác:", list(registry.keys()))
            if selected_ctdt:
                item = registry[selected_ctdt]
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    is_active = st.checkbox("Trạng thái Kích hoạt (Active)", value=item.get("active", False), key=f"act_{selected_ctdt}")
                    if st.button("💾 Cập nhật trạng thái Active", key=f"save_act_{selected_ctdt}"):
                        registry[selected_ctdt]["active"] = is_active
                        save_curriculum_registry(registry)
                        st.success("Đã cập nhật trạng thái CTĐT.")
                        st.rerun()
                with col_c2:
                    if st.button("🗑️ Xóa CTĐT này", key=f"del_ctdt_{selected_ctdt}"):
                        del registry[selected_ctdt]
                        save_curriculum_registry(registry)
                        st.success(f"Đã xóa CTĐT {selected_ctdt}.")
                        st.rerun()

                st.markdown("**Danh sách học phần thuộc CTĐT:**")
                df_courses = pd.DataFrame(item.get("courses", []))
                st.dataframe(df_courses, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có CTĐT tùy chỉnh nào được nạp trong registry.")

    # ---------------- TAB 3: QUẢN LÝ TÀI KHOẢN (THÊM / SỬA / XÓA) ----------------
    with a3:
        st.subheader("Quản lý Tài khoản Hệ thống")
        users = load_users()
        
        df_users = pd.DataFrame([
            {"Username": u, "Role": d.get("role"), "MSSV": d.get("mssv", "—"), "Trạng thái": "Hoạt động" if d.get("active", True) else "Khóa"} 
            for u, d in users.items()
        ])
        st.dataframe(df_users, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        m_tab1, m_tab2, m_tab3 = st.tabs(["➕ Thêm tài khoản", "✏️ Sửa tài khoản", "🗑️ Xóa tài khoản"])
        
        with m_tab1:
            with st.form("form_add_user"):
                u_add = st.text_input("Tên tài khoản")
                p_add = st.text_input("Mật khẩu", type="password")
                r_add = st.selectbox("Quyền", ["user", "admin"])
                m_add = st.text_input("MSSV liên kết (bắt buộc cho USER)", placeholder="Ví dụ: B2103882")
                if st.form_submit_button("Thêm tài khoản mới"):
                    if u_add.strip() and p_add:
                        if u_add.strip() in users:
                            st.error("Tên tài khoản đã tồn tại.")
                        elif r_add == "user" and not m_add.strip():
                            st.error("Tài khoản USER phải liên kết với MSSV.")
                        else:
                            users[u_add.strip()] = {
                                "password_hash": _hash_password(p_add),
                                "role": r_add,
                                "active": True,
                                "mssv": m_add.strip() if m_add.strip() else None
                            }
                            _save_json(AUTH_FILE, users)
                            st.success(f"Đã thêm tài khoản {u_add.strip()} thành công.")
                            st.rerun()
                    else:
                        st.error("Vui lòng điền đầy đủ Tên tài khoản và Mật khẩu.")

        with m_tab2:
            u_edit = st.selectbox("Chọn tài khoản cần sửa:", list(users.keys()), key="select_user_edit")
            if u_edit:
                curr_item = users[u_edit]
                with st.form("form_edit_user"):
                    p_edit = st.text_input("Mật khẩu mới (Để trống nếu không muốn đổi)", type="password")
                    r_edit = st.selectbox("Quyền", ["user", "admin"], index=0 if curr_item.get("role") == "user" else 1)
                    m_edit = st.text_input("MSSV liên kết", value=curr_item.get("mssv") or "")
                    act_edit = st.checkbox("Trạng thái Hoạt động (Active)", value=curr_item.get("active", True))
                    if st.form_submit_button("Cập nhật thông tin"):
                        if p_edit.strip():
                            curr_item["password_hash"] = _hash_password(p_edit.strip())
                        curr_item["role"] = r_edit
                        curr_item["mssv"] = m_edit.strip() if m_edit.strip() else None
                        curr_item["active"] = act_edit
                        users[u_edit] = curr_item
                        _save_json(AUTH_FILE, users)
                        st.success(f"Đã cập nhật thông tin tài khoản {u_edit}.")
                        st.rerun()

        with m_tab3:
            u_del = st.selectbox("Chọn tài khoản cần xóa:", [u for u in users.keys() if u != "admin"], key="select_user_del")
            if u_del:
                st.warning(f"Bạn có chắc chắn muốn xóa tài khoản **{u_del}**?")
                if st.button(f"🔥 Xóa vĩnh viễn {u_del}", key="btn_confirm_del"):
                    del users[u_del]
                    _save_json(AUTH_FILE, users)
                    st.success(f"Đã xóa tài khoản {u_del}.")
                    st.rerun()

    # ---------------- TAB 4: DUYỆT CHỨNG CHỈ NGOẠI NGỮ ----------------
    with a4:
        st.subheader("Duyệt kê khai Chứng chỉ Ngoại ngữ của Sinh viên")
        cert_registry = load_certificate_registry()
        
        pending_certs = {k: v for k, v in cert_registry.items() if isinstance(v, dict) and v.get("status") == "CHO_DUYET"}
        
        if pending_certs:
            selected_cert_mssv = st.selectbox("Chọn MSSV cần duyệt chứng chỉ:", list(pending_certs.keys()))
            cert_item = pending_certs[selected_cert_mssv]
            
            c_info1, c_info2 = st.columns([1, 1])
            with c_info1:
                st.write(f"**MSSV:** {selected_cert_mssv}")
                st.write(f"**Loại chứng chỉ:** {cert_item.get('cert_type')}")
                st.write(f"**Điểm số / Trình độ:** {cert_item.get('score')}")
                st.write(f"**Ngày kê khai:** {cert_item.get('submitted_at')}")
                
                fl_codes = [c["code"] for c in FL_COURSES]
                exempt_selected = st.multiselect(
                    "Chọn các học phần được MIỄN (Điểm M):",
                    options=fl_codes,
                    default=cert_item.get("exempt_courses", ["XH023", "XH024", "XH025"])
                )
                
                b_approve, b_reject = st.columns(2)
                with b_approve:
                    if st.button("✅ DUYỆT CHỨNG CHỈ", key="approve_cert"):
                        cert_registry[selected_cert_mssv]["status"] = "DA_DUYET"
                        cert_registry[selected_cert_mssv]["exempt_courses"] = exempt_selected
                        cert_registry[selected_cert_mssv]["reviewed_at"] = datetime.now().isoformat()
                        save_certificate_registry(cert_registry)
                        st.success(f"Đã duyệt chứng chỉ cho sinh viên {selected_cert_mssv}.")
                        st.rerun()
                with b_reject:
                    if st.button("❌ TỪ CHỐI", key="reject_cert"):
                        cert_registry[selected_cert_mssv]["status"] = "TU_CHOI"
                        cert_registry[selected_cert_mssv]["reviewed_at"] = datetime.now().isoformat()
                        save_certificate_registry(cert_registry)
                        st.error(f"Đã từ chối chứng chỉ của sinh viên {selected_cert_mssv}.")
                        st.rerun()

            with c_info2:
                img_b64 = cert_item.get("image_b64")
                if img_b64:
                    st.write("**Ảnh minh chứng:**")
                    st.markdown(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:8px; border:1px solid #ccc;">', unsafe_allow_html=True)
                else:
                    st.info("Không có hình ảnh minh chứng đính kèm.")
        else:
            st.info("Hiện không có kê khai chứng chỉ nào đang chờ duyệt.")
            
        st.markdown("---")
        st.markdown("#### Tất cả chứng chỉ đã xử lý")
        df_certs = pd.DataFrame([
            {"MSSV": k, "Loại chứng chỉ": v.get("cert_type"), "Điểm": v.get("score"), "Trạng thái": v.get("status"), "Môn miễn": ", ".join(v.get("exempt_courses", []))}
            for k, v in cert_registry.items() if isinstance(v, dict)
        ])
        if not df_certs.empty:
            st.dataframe(df_certs, use_container_width=True, hide_index=True)

    st.stop()


# -----------------------------------------------------------------------------
# 9. DỮ LIỆU & GIAO DIỆN MÀN HÌNH CHÍNH (DÀNH CHO CẢ USER VÀ ADMIN DASHBOARD)
# -----------------------------------------------------------------------------
processed_dir = PROJECT_ROOT / "data" / "processed"
candidate_files = [
    processed_dir / "full_student_features_wide.csv",
    processed_dir / "student_full_matrix_wide.csv",
    PROJECT_ROOT / "data" / "student_full_matrix_wide.csv",
    Path("full_student_features_wide.csv"),
]
found_file = next((p for p in candidate_files if p.exists()), None)
df_students = load_wide_matrix_file(found_file) if found_file else None

if is_admin() and admin_page == "📊 Phân tích theo khóa":
    if df_students is not None and not df_students.empty:
        render_cohort_analytics(df_students)
    else:
        st.warning("Không tìm thấy dữ liệu sinh viên để phân tích theo khóa.")
    st.stop()

if df_students is not None and not df_students.empty:
    mssv_col = "MSSV" if "MSSV" in df_students.columns else df_students.columns[0]
    list_mssv = sorted(df_students[mssv_col].astype(str).unique())
    
    if is_admin():
        st.sidebar.markdown("### 🎯 **CHỌN SINH VIÊN**")
        selection_mode = st.sidebar.radio("Cách chọn", ["Theo MSSV", "Theo khóa + MSSV"], key="student_selection_mode")
        if selection_mode == "Theo khóa + MSSV":
            cohort_df_sidebar = prepare_cohort_dataframe(df_students)
            cohort_options = sorted(cohort_df_sidebar["Khóa"].dropna().unique().tolist(), key=lambda x: int(x[1:]) if str(x)[1:].isdigit() else 999)
            selected_sidebar_batch = st.sidebar.selectbox("Khóa:", ["Tất cả các khóa"] + [_batch_label(x) for x in cohort_options], key="sidebar_batch")
            batch_code_sidebar = "Tất cả các khóa" if selected_sidebar_batch == "Tất cả các khóa" else next((b for b in cohort_options if _batch_label(b) == selected_sidebar_batch), selected_sidebar_batch)
            filtered_mssv = list_mssv
            if batch_code_sidebar != "Tất cả các khóa":
                filtered_mssv = cohort_df_sidebar.loc[cohort_df_sidebar["Khóa"] == batch_code_sidebar, "MSSV"].astype(str).tolist()
            selected_mssv = st.sidebar.selectbox("MSSV trong khóa:", sorted(filtered_mssv), key="sidebar_mssv_by_batch")
        else:
            selected_mssv = st.sidebar.selectbox("MSSV:", list_mssv, key="sidebar_mssv")
    else:
        selected_mssv = str(current_user.get("mssv") or "").strip()
        if not selected_mssv and list_mssv:
            selected_mssv = list_mssv[0]

    if selected_mssv and selected_mssv in list_mssv:
        sv_row = df_students[df_students[mssv_col].astype(str) == selected_mssv].iloc[0]
        
        major_key, ma_lop = auto_detect_major_from_row(sv_row)
        batch_code, batch_resolution = resolve_student_batch(
            selected_mssv,
            ma_lop,
            sv_row.get("NganhHoc", "")
        )
        major_info = PROGRAM_METADATA.get(major_key, PROGRAM_METADATA["ATTT"])
        metadata_batch = batch_code if batch_code in major_info["mapping"] else extract_batch_code(selected_mssv)
        meta = major_info["mapping"].get(metadata_batch, list(major_info["mapping"].values())[-1])
        
        curriculum_mapping = get_curriculum_mapping_status(major_key, batch_code)
        if curriculum_mapping["status"] != "EXACT":
            st.warning(f"⚠️ CTĐT PDF: {curriculum_mapping['message']}")
        else:
            item_map = curriculum_mapping["item"]
            st.caption(
                f"📚 CTĐT đang áp dụng: {item_map.get('filename')} | "
                f"{item_map.get('decision_number') or 'Không rõ quyết định'} | "
                f"{item_map.get('decision_date') or 'Không rõ ngày'}"
            )

        sem_df = parse_student_semester_history(sv_row)
        passed_courses, failed_courses, low_grade_courses, dataset_exempt, grade_dict = parse_student_course_details(sv_row)
        
        cert_exempt = get_certificate_exempt_courses(selected_mssv)
        exempt_courses = dataset_exempt | cert_exempt
        passed_courses = passed_courses | exempt_courses

        all_program_courses = get_program_courses(major_key, batch_code)
        course_credit_lookup = {c["code"]: c["credits"] for c in all_program_courses}
        for c in FL_COURSES:
            course_credit_lookup[c["code"]] = c["credits"]
        for c in GDQP_COURSES:
            course_credit_lookup[c["code"]] = c["credits"]
        for sport in GDTC_SPORTS.values():
            for c in sport:
                course_credit_lookup[c["code"]] = c["credits"]

        tc_earned_total = 0
        for code in (passed_courses | exempt_courses):
            tc_earned_total += int(course_credit_lookup.get(code, 2))

        latest_sem = sem_df.iloc[-1]
        cpa_10 = float(latest_sem.get("ĐTBCTL (Thang 10)", 0.0))
        cpa_4 = float(latest_sem.get("ĐTBCTL (Thang 4)", 0.0))
        dtbchk_4 = float(latest_sem.get("ĐTBCHK (Thang 4)", 0.0))
        tc_no_dong = int(latest_sem.get("Số TC Nợ đọng", 0))
        fail_rate = float(latest_sem.get("fail_rate_num", 0.0))

        rec_results = analyze_ctu_recommendations(major_key, batch_code, passed_courses, failed_courses, low_grade_courses, tc_earned_total)
        fl_data = analyze_fl_progress(passed_courses, exempt_courses, grade_dict)
        gdqp_data = analyze_gdqp_progress(passed_courses, exempt_courses, grade_dict)
        
        on_time_prob = max(0.2, min(0.98, (cpa_4 / 4.0) * 0.6 + (1 - min(tc_no_dong, 20)/20.0) * 0.4))
        risk_prob = 1.0 - on_time_prob

        progress_assessment = evaluate_graduation_progress(on_time_prob)
        academic_classification = classify_ctu_academic_performance(cpa_4)

        vector_recs, vector_diag, vec_status = generate_cross_cohort_recommendations(
            selected_mssv, df_students, major_key, batch_code, passed_courses, failed_courses, tc_earned_total
        )

        xai_consultation_text = build_xai_consultation(
            cpa_4, dtbchk_4, tc_earned_total, tc_no_dong, fail_rate,
            progress_assessment, academic_classification,
            vector_recs, vector_diag, rec_results, fl_data, gdqp_data
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("MSSV - Lớp", f"{selected_mssv}", f"{ma_lop} ({batch_code})")
        with m2:
            st.metric("ĐTBCTL (Thang 4)", f"{cpa_4:.2f} / 4.0", f"Thang 10: {cpa_10:.2f}")
        with m3:
            st.metric("TC Tích lũy (Toàn khóa)", f"{tc_earned_total} TC", f"Nợ đọng: {tc_no_dong} TC", delta_color="inverse")
        with m4:
            st.metric("Xếp loại Học tập", f"{academic_classification['icon']} {academic_classification['label']}", f"ĐTBHK: {dtbchk_4:.2f}")
        with m5:
            st.metric("Dự báo Đúng tiến độ", f"{on_time_prob*100:.1f}%", f"Rủi ro: {risk_prob*100:.1f}%", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        tab_names = [
            "📊 Đánh giá tổng quan",
            "🎯 Lộ trình đăng ký học phần",
            "📜 Tiến độ chuẩn đầu ra",
            "📈 Lịch sử Học tập & Biểu đồ",
            "🤖 Cố vấn Học tập AI"
        ]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

        with tab1:
            # -----------------------------------------------------------------
            # MODERN OVERVIEW / ACADEMIC ASSESSMENT
            # -----------------------------------------------------------------
            status_level = progress_assessment.get("level", "safe")
            status_icon = {"safe": "✓", "moderate": "•", "warning": "!", "critical": "×"}.get(status_level, "•")
            status_percent = int(round(on_time_prob * 100))
            academic_gpa_class = str(academic_classification.get("label", ""))

            st.markdown(
                f"""
                <div class='overview-shell'>
                    <div class='overview-eyebrow'>HỒ SƠ HỌC TẬP • PHÂN TÍCH AI</div>
                    <div class='overview-title'>Đánh giá tổng quan kết quả học tập</div>
                    <div class='overview-subtitle'>Tóm tắt năng lực học tập, tiến độ tốt nghiệp và các chỉ báo cần ưu tiên xử lý.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class='overview-hero'>
                    <div class='overview-hero-inner'>
                        <div class='overview-status-dot {status_level}'>{status_icon}</div>
                        <div style='flex:1;'>
                            <div class='overview-hero-kicker'>Đánh giá tiến độ tốt nghiệp</div>
                            <div class='overview-hero-title'>{progress_assessment['value']}</div>
                            <div class='overview-badge {status_level}'>Mức trạng thái: {status_level.upper()}</div>
                            <p class='overview-description'>{progress_assessment['description']}</p>
                            <div class='overview-progress'>
                                <div class='overview-progress-row'>
                                    <span>Mức độ dự báo đúng tiến độ</span>
                                    <span class='overview-progress-value'>{status_percent}%</span>
                                </div>
                                <div class='overview-progress-track'>
                                    <div class='overview-progress-fill {status_level}' style='width:{status_percent}%'></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # KPI cards: dùng màu theo ý nghĩa, không phải màu trang trí.
            gpa_cls = "good" if cpa_4 >= 3.2 else ("warn" if cpa_4 >= 2.0 else "danger")
            latest_cls = "good" if dtbchk_4 >= 3.2 else ("warn" if dtbchk_4 >= 2.0 else "danger")
            debt_cls = "good" if tc_no_dong == 0 else ("warn" if tc_no_dong <= 6 else "danger")
            fail_cls = "good" if fail_rate == 0 else ("warn" if fail_rate < 20 else "danger")
            st.markdown(
                f"""
                <div class='overview-grid'>
                    <div class='overview-metric {gpa_cls}'>
                        <div class='overview-metric-label'>ĐTBCTL · Thang 4</div>
                        <div class='overview-metric-value'>{cpa_4:.2f}</div>
                        <div class='overview-metric-note'>Tương đương thang 10: {cpa_10:.2f} · {academic_gpa_class}</div>
                    </div>
                    <div class='overview-metric {latest_cls}'>
                        <div class='overview-metric-label'>ĐTB học kỳ gần nhất</div>
                        <div class='overview-metric-value'>{dtbchk_4:.2f}</div>
                        <div class='overview-metric-note'>So với mức 4.0 tối đa</div>
                    </div>
                    <div class='overview-metric {debt_cls}'>
                        <div class='overview-metric-label'>Tín chỉ nợ đọng</div>
                        <div class='overview-metric-value'>{tc_no_dong} TC</div>
                        <div class='overview-metric-note'>0 TC là trạng thái tốt</div>
                    </div>
                    <div class='overview-metric {fail_cls}'>
                        <div class='overview-metric-label'>Tỷ lệ rớt môn gần nhất</div>
                        <div class='overview-metric-value'>{fail_rate:.1f}%</div>
                        <div class='overview-metric-note'>Chỉ báo cần theo dõi xu hướng</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.15, .85], gap="large")
            with left:
                st.markdown(
                    f"""
                    <div class='overview-card'>
                        <div class='overview-section-title'>Các yếu tố ảnh hưởng chính</div>
                    """,
                    unsafe_allow_html=True,
                )
                factors = []
                if cpa_4 >= 3.2:
                    factors.append(("positive", "✓", f"ĐTBCTL {cpa_4:.2f}/4.0 cho thấy nền tảng học tập hiện tại ở mức {academic_gpa_class.lower()}."))
                elif cpa_4 >= 2.5:
                    factors.append(("neutral", "•", f"ĐTBCTL {cpa_4:.2f}/4.0 đang ở mức {academic_gpa_class.lower()}, cần duy trì ổn định."))
                else:
                    factors.append(("risk", "!", f"ĐTBCTL {cpa_4:.2f}/4.0 đang thấp; đây là yếu tố cần ưu tiên cải thiện."))
                if tc_no_dong > 0:
                    factors.append(("risk" if tc_no_dong > 6 else "attention", "!", f"Đang có {tc_no_dong} tín chỉ nợ đọng; cần xử lý các học phần có thể chặn chuỗi tiên quyết."))
                else:
                    factors.append(("positive", "✓", "Không có tín chỉ nợ đọng ở học kỳ gần nhất."))
                if fail_rate > 0:
                    factors.append(("risk" if fail_rate >= 20 else "attention", "!", f"Tỷ lệ rớt môn gần nhất là {fail_rate:.1f}%; nên theo dõi thêm xu hướng qua các học kỳ."))
                else:
                    factors.append(("positive", "✓", "Không ghi nhận tỷ lệ rớt môn ở học kỳ gần nhất."))
                for kind, icon, text in factors:
                    st.markdown(f"<div class='overview-factor {kind}'><div class='overview-factor-icon'>{icon}</div><div>{text}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown(
                    f"""
                    <div class='overview-card'>
                        <div class='overview-section-title'>Điều kiện tốt nghiệp cần theo dõi</div>
                    """,
                    unsafe_allow_html=True,
                )
                fl_done = bool(fl_data.get("is_qualified", False))
                gdqp_done = bool(gdqp_data.get("is_qualified", False))
                fl_text = "Đã hoàn tất" if fl_done else f"Chưa hoàn tất · {fl_data.get('total_credits', 0)}/10 TC"
                gdqp_text = "Đã hoàn tất" if gdqp_done else f"Chưa hoàn tất · {gdqp_data.get('completed_count', 0)}/{gdqp_data.get('total_courses', 0)} HP"
                fl_status = "done" if fl_done else "pending"
                gdqp_status = "done" if gdqp_done else "pending"
                st.markdown(f"<div class='overview-condition'><div class='overview-condition-name'>Ngoại ngữ</div><div class='overview-condition-status {fl_status}'>{fl_text}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='overview-condition'><div class='overview-condition-name'>GDQP-AN</div><div class='overview-condition-status {gdqp_status}'>{gdqp_text}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='overview-condition'><div class='overview-condition-name'>Tín chỉ đã tích lũy</div><div class='overview-condition-status done'>{tc_earned_total} TC</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

            # XAI: hiển thị bản tóm tắt có cấu trúc, phần diễn giải dài đưa vào expander.
            neighbors = vector_diag.get("neighbors", []) if isinstance(vector_diag, dict) else []
            avg_sim = float(np.mean([float(n.get("similarity", 0) or 0) for n in neighbors])) if neighbors else 0.0
            pathways = vector_diag.get("pathways", []) if isinstance(vector_diag, dict) else []
            top_path = pathways[0].get("pathway", "Chưa xác định") if pathways else "Chưa xác định"
            st.markdown(
                f"""
                <div class='overview-xai'>
                    <div class='overview-xai-title'>Giải thích XAI & bằng chứng recommendation</div>
                    <div class='overview-xai-text'>
                        Hồ sơ hiện tại được đối chiếu với <b>{len(neighbors)}</b> sinh viên tương đồng;
                        độ tương đồng trung bình <b>{avg_sim:.3f}</b>. Hướng lộ trình nổi bật: <b>{top_path}</b>.
                        Recommendation nên được đọc cùng điều kiện CTĐT/prerequisite thay vì xem như quyết định tự động.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Xem giải thích chi tiết từ hệ thống XAI", expanded=False):
                st.markdown(xai_consultation_text)

            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

            c_fail1, c_fail2 = st.columns(2, gap="large")
            with c_fail1:
                st.markdown(
                    f"""<div class='overview-card'><div class='overview-section-title'>Môn chưa đạt / nợ đọng</div>""",
                    unsafe_allow_html=True,
                )
                if failed_courses:
                    for fc in sorted(failed_courses):
                        st.markdown(f"<div class='overview-factor risk'><div class='overview-factor-icon'>×</div><div><b>{fc}</b><br><span style='font-size:.75rem'>Nên ưu tiên học lại để tránh ảnh hưởng các học phần phía sau.</span></div></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='overview-factor positive'><div class='overview-factor-icon'>✓</div><div>Không có học phần chưa đạt.</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c_fail2:
                st.markdown(
                    f"""<div class='overview-card'><div class='overview-section-title'>Môn điểm thấp nên cân nhắc cải thiện</div>""",
                    unsafe_allow_html=True,
                )
                if low_grade_courses:
                    for lgc in sorted(low_grade_courses):
                        score = grade_dict.get(lgc, "D")
                        st.markdown(f"<div class='overview-factor attention'><div class='overview-factor-icon'>!</div><div><b>{lgc}</b> · Điểm {score}<br><span style='font-size:.75rem'>Cân nhắc cải thiện nếu phù hợp mục tiêu GPA.</span></div></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='overview-factor positive'><div class='overview-factor-icon'>✓</div><div>Không có học phần điểm thấp cần ưu tiên cải thiện.</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("### Gợi ý Lộ trình Đăng ký học phần ở học kỳ Tiếp theo")
            
            st.markdown("#### 🤖 Gợi ý bằng Mô hình Vector Similarity & Cross-Cohort kNN (XAI)")
            if vector_recs:
                df_vec = pd.DataFrame([
                    {
                        "Thứ tự": r.get("rank"),
                        "Mã HP": r.get("course_id"),
                        "Tên HP": r.get("course_name"),
                        "Số TC": r.get("credits"),
                        "Điểm dự đoán": f"{r.get('predicted_score', 0):.2f}/10",
                        "Độ tin cậy": f"{r.get('confidence', 0):.2f}",
                        "Hướng chuyên sâu": r.get("pathway"),
                        "Lý do gợi ý": " | ".join(r.get("reasons", []))
                    }
                    for r in vector_recs
                ])
                st.dataframe(df_vec, use_container_width=True, hide_index=True)
            else:
                st.warning("Chưa tạo được gợi ý kNN vector similarity.")

            st.markdown("---")
            st.markdown("#### 🧭 Bộ máy mô phỏng lộ trình học tập tối ưu")
            path_c1, path_c2, path_c3 = st.columns(3)
            with path_c1:
                path_max_tc = st.slider("Tối đa TC / học kỳ", 12, 24, 18, key="path_max_tc")
            with path_c2:
                path_min_tc = st.slider("Mục tiêu tối thiểu TC / học kỳ", 0, 18, 12, key="path_min_tc")
            with path_c3:
                path_terms = st.slider("Số học kỳ mô phỏng", 1, 8, 4, key="path_terms")

            all_program_codes = [c.get("code") for c in get_program_courses(major_key, batch_code) if c.get("code")]
            target_options = ["Không chọn mục tiêu cụ thể"] + all_program_codes
            target_selected = st.selectbox("🎯 Học phần mục tiêu (tùy chọn)", target_options, key="path_target")
            target_code = None if target_selected == "Không chọn mục tiêu cụ thể" else target_selected

            path_plan = build_optimal_learning_path(
                major_key, batch_code, list(passed_courses), list(failed_courses), tc_earned_total,
                max_credits_per_term=path_max_tc,
                min_credits_per_term=path_min_tc,
                target_code=target_code,
                max_terms=path_terms,
            )
            if path_plan.get("terms"):
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Học kỳ mô phỏng", len(path_plan.get("terms", [])))
                pm2.metric("TC sau mô phỏng", path_plan.get("end_credits_simulated", 0))
                pm3.metric("Học phần chưa xếp", len(path_plan.get("unscheduled", [])))
                path_df = optimal_path_to_dataframe(path_plan)
                if not path_df.empty:
                    st.dataframe(path_df, use_container_width=True, hide_index=True)
                for term in path_plan.get("terms", []):
                    status_icon = "🟢" if term.get("status") == "OK" else "🟡"
                    codes = ", ".join(f"{c.get('code')} - {c.get('name')} ({c.get('credits')} TC)" for c in term.get("courses", []))
                    status_label = " — Kỳ bổ sung đặc biệt" if term.get("status") == "SPECIAL_OVERFLOW" else ""
                    st.markdown(f"{status_icon} **HK mô phỏng {term.get('term')}{status_label} — {term.get('credits')} TC:** {codes}")
                if path_plan.get("unscheduled"):
                    st.warning(
                        "Một số học phần vẫn chưa thể xếp do điều kiện CTĐT/prerequisite. "
                        "Các học phần Niên luận, Thực tập và Luận văn đủ điều kiện đã được tự động "
                        "đẩy sang học kỳ bổ sung nếu vượt số kỳ mô phỏng ban đầu."
                    )
            else:
                st.warning("Chưa tạo được lộ trình khả thi. Kiểm tra các tiên quyết chưa xác thực hoặc dữ liệu CTĐT.")

            st.markdown("---")
            st.markdown("#### 🎓 Graduation Progress & Scenario Simulation")
            grad_progress = build_graduation_progress(
                major_key, batch_code, list(passed_courses), list(failed_courses), tc_earned_total
            )
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("TC đã tích lũy", f"{grad_progress['earned_credits']:.0f}")
            gc2.metric("Mục tiêu CTĐT", f"{grad_progress['target_credits']:.0f}" if grad_progress['target_credits'] is not None else "Chưa xác định")
            gc3.metric("TC còn thiếu", f"{grad_progress['credit_gap']:.0f}" if grad_progress['credit_gap'] is not None else "—")
            gc4.metric("Tiến độ", f"{grad_progress['progress_percent']:.1f}%" if grad_progress['progress_percent'] is not None else "—")
            st.caption(grad_progress['note'])

            scenario_shock_options = ["Không mô phỏng môn rớt mới"] + all_program_codes
            scenario_shock = st.selectbox("⚠️ Mô phỏng nếu rớt/thất bại một học phần", scenario_shock_options, key="scenario_shock")
            shock_code = None if scenario_shock == "Không mô phỏng môn rớt mới" else scenario_shock
            scenario_result = simulate_learning_scenarios(
                major_key, batch_code, list(passed_courses), list(failed_courses), tc_earned_total,
                max_terms=8, target_code=target_code, shock_failed_course=shock_code
            )
            scenario_df = scenario_results_to_dataframe(scenario_result)
            if not scenario_df.empty:
                st.dataframe(scenario_df, use_container_width=True, hide_index=True)
            if shock_code:
                shock_row = next((x for x in scenario_result['scenarios'] if x.get('scenario') == f'Kịch bản rớt {shock_code}'), None)
                if shock_row is not None:
                    impacted = shock_row.get('impacted_courses', [])
                    st.warning(f"Nếu {shock_code} không hoàn thành, Graph Engine xác nhận tối đa {len(impacted)} học phần phía sau có thể bị ảnh hưởng bởi quan hệ tiên quyết đã trích xuất.")
                    if impacted:
                        st.caption("Học phần bị ảnh hưởng: " + ", ".join(impacted[:30]))

            st.markdown("---")
            st.markdown("#### 📌 Phân loại Học phần Khả thi theo Khung CTĐT")
            
            r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
                "🔴 Cần học lại / Cải thiện", 
                "📘 Bắt buộc khả thi", 
                "📗 Tự chọn khả thi", 
                "🔒 HP chưa đủ điều kiện"
            ])
            
            with r_tab1:
                if rec_results["improve"]:
                    st.dataframe(pd.DataFrame(rec_results["improve"]), use_container_width=True, hide_index=True)
                else:
                    st.success("Không có môn nào cần học lại hoặc cải thiện.")

            with r_tab2:
                if rec_results["mandatory"]:
                    st.dataframe(pd.DataFrame(rec_results["mandatory"]), use_container_width=True, hide_index=True)
                else:
                    st.info("Đã hoàn thành các môn bắt buộc khả thi hiện tại.")

            with r_tab3:
                if rec_results["elective"]:
                    st.dataframe(pd.DataFrame(rec_results["elective"]), use_container_width=True, hide_index=True)
                else:
                    st.info("Không có môn tự chọn khả thi.")

            with r_tab4:
                if rec_results["locked"]:
                    st.dataframe(pd.DataFrame(rec_results["locked"]), use_container_width=True, hide_index=True)

                st.markdown("#### 🧠 Phân tích đồ thị tiên quyết & tác động")
                graph_info = rec_results.get("graph", {})
                g1, g2, g3 = st.columns(3)
                g1.metric("Học phần trong đồ thị", graph_info.get("nodes", 0))
                g2.metric("Quan hệ tiên quyết", graph_info.get("edges", 0))
                g3.metric("Tiên quyết chưa xác thực", graph_info.get("unknown_prereq_count", 0))
                if rec_results.get("critical_path_candidates"):
                    st.caption("Các học phần chưa hoàn thành có tác động mở khóa lớn nhất.")
                    st.dataframe(pd.DataFrame(rec_results["critical_path_candidates"])[[c for c in ["code","name","credits","downstream_impact"] if c in pd.DataFrame(rec_results["critical_path_candidates"]).columns]], use_container_width=True, hide_index=True)
                if graph_info.get("failed_impact"):
                    impact_rows=[]
                    for item in graph_info["failed_impact"]:
                        impact_rows.append({"Môn đang rớt": item.get("code"), "Số môn có thể bị ảnh hưởng": item.get("blocked_count"), "Một số môn phía sau": ", ".join(c.get("code","") for c in item.get("blocked_courses", [])[:8])})
                    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
                else:
                    st.success("Tất cả các môn đều đã đủ điều kiện!")

        with tab3:
            st.markdown("### 📜 Tiến độ Chuẩn đầu ra: Ngoại ngữ, GDQP-AN & GDTC")

            with st.expander("📝 **KÊ KHAI CHỨNG CHỈ NGOẠI NGỮ (Dành cho Sinh viên)**", expanded=True):
                cert_reg = load_certificate_registry()
                current_cert = cert_reg.get(selected_mssv, {})
                
                if current_cert.get("status") == "DA_DUYET":
                    st.success(f"🎉 Chứng chỉ **{current_cert.get('cert_type')}** (Điểm/Trình độ: {current_cert.get('score')}) đã **ĐƯỢC DUYỆT**. Các môn miễn: {', '.join(current_cert.get('exempt_courses', []))}.")
                elif current_cert.get("status") == "CHO_DUYET":
                    st.warning(f"⏳ Chứng chỉ **{current_cert.get('cert_type')}** (Kê khai ngày {current_cert.get('submitted_at')}) đang trong trạng thái **CHỜ CỐ VẤN / ADMIN DUYỆT**.")
                else:
                    st.write("Sinh viên nộp chứng chỉ quốc tế (IELTS, TOEIC, VSTEP,...) để kê khai xét miễn 10 TC Ngoại ngữ:")
                    with st.form("form_submit_cert"):
                        cert_type = st.selectbox("Loại chứng chỉ Ngoại ngữ:", ["TOEIC", "IELTS", "VSTEP", "TOEFL iBT", "Chứng chỉ Tiếng Pháp (DELF/DALF)", "Khác"])
                        score_str = st.text_input("Điểm số / Trình độ đạt được:", placeholder="Ví dụ: 550, 6.0, B2,...")
                        uploaded_img = st.file_uploader("Tải ảnh minh chứng (PNG, JPG):", type=["png", "jpg", "jpeg"])
                        
                        if st.form_submit_button("📤 Gửi Kê khai Chứng chỉ"):
                            if score_str.strip():
                                img_b64 = ""
                                if uploaded_img:
                                    img_b64 = base64.b64encode(uploaded_img.getvalue()).decode("utf-8")
                                
                                new_cert_data = {
                                    "mssv": selected_mssv,
                                    "cert_type": cert_type,
                                    "score": score_str.strip(),
                                    "image_b64": img_b64,
                                    "status": "CHO_DUYET",
                                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "exempt_courses": ["XH023", "XH024", "XH025"]
                                }
                                save_student_certificate(selected_mssv, new_cert_data)
                                st.success("Đã gửi kê khai chứng chỉ thành công. Vui lòng chờ Cố vấn/Admin duyệt.")
                                st.rerun()
                            else:
                                st.error("Vui lòng nhập điểm số hoặc trình độ chứng chỉ.")

            st.markdown("---")

            st.markdown("#### 1. 🇬🇧/🇫🇷 Tiến độ Học phần Ngoại ngữ (Yêu cầu 10 TC)")
            if fl_data["is_qualified"]:
                st.success(f"✅ **Đã đạt chuẩn Ngoại ngữ tốt nghiệp!** Tổng TC hoàn thành: {fl_data['total_credits']}/10 TC" + (" (Có bằng chứng miễn điểm M)" if fl_data['has_m_exemption'] else ""))
            else:
                st.warning(f"⚠️ **Chưa đủ điều kiện Ngoại ngữ!** Đã tích lũy: {fl_data['total_credits']}/10 TC. Cần đăng ký thêm học phần hoặc nộp chứng chỉ.")

            df_fl = pd.DataFrame(fl_data["fl_courses"])
            st.dataframe(
                df_fl.rename(columns={"code": "Mã HP", "name": "Tên học phần", "credits": "Số TC", "score": "Điểm", "status": "Trạng thái"}),
                use_container_width=True, hide_index=True
            )

            st.markdown("---")

            st.markdown("#### 2. 🛡️ Tiến độ Giáo dục Quốc phòng - An ninh (GDQP-AN)")
            if gdqp_data["is_qualified"]:
                st.success(f"✅ **Đã hoàn thành toàn bộ khối học phần GDQP-AN!** ({gdqp_data['completed_count']}/{gdqp_data['total_courses']} môn)")
            else:
                st.warning(f"⚠️ **Chưa hoàn thành đủ GDQP-AN!** Đã đạt {gdqp_data['completed_count']}/{gdqp_data['total_courses']} môn.")

            df_qp = pd.DataFrame(gdqp_data["gdqp_courses"])
            st.dataframe(
                df_qp.rename(columns={"code": "Mã HP", "name": "Tên học phần", "credits": "Số TC", "score": "Điểm", "status": "Trạng thái"}),
                use_container_width=True, hide_index=True
            )

            st.markdown("---")

            st.markdown("#### 3. ⚽ Tiến độ Giáo dục Thể chất (GDTC)")
            st.write("Theo quy định CTU, sinh viên phải hoàn thành **1 môn thể thao chọn trước nối tiếp đủ 3 học phần** (Ví dụ: Cầu lông 1 ➔ Cầu lông 2 ➔ Cầu lông 3).")
            
            gdtc_list = rec_results.get("gdtc_summary", [])
            for g_item in gdtc_list:
                st.markdown(f"- **Môn:** `{g_item['sport']}` | **Trạng thái:** {g_item['status']}")
    
            # Sử dụng .get() để tránh KeyError, chỉ hiển thị nếu có dữ liệu
            next_course = g_item.get("next_course")
            if next_course:
                st.caption(f"👉 {next_course}")

            gdtc_table_rows = []
            for sport_name, sport_courses in GDTC_SPORTS.items():
                for sc in sport_courses:
                    code = sc["code"]
                    gr = grade_dict.get(code, None)
                    is_exempt = (code in exempt_courses) or (gr == "M (Miễn)")
                    if is_exempt:
                        stt = "🏅 Đã miễn (M)"
                        sc_str = "M"
                    elif code in passed_courses:
                        stt = "✅ Đã qua môn"
                        sc_str = f"{gr:.1f}" if isinstance(gr, (int, float)) else "Đạt"
                    elif gr is not None:
                        stt = "❌ Chưa đạt"
                        sc_str = f"{gr:.1f}" if isinstance(gr, (int, float)) else "Chưa đạt"
                    else:
                        stt = "⏳ Chưa học"
                        sc_str = "—"

                    gdtc_table_rows.append({
                        "Môn thể thao": sport_name,
                        "Mã HP": code,
                        "Tên học phần": sc["name"],
                        "Số TC": sc["credits"],
                        "Điểm": sc_str,
                        "Trạng thái": stt
                    })

            with st.expander("🔍 **Xem chi tiết toàn bộ danh mục môn Giáo dục thể chất**"):
                st.dataframe(pd.DataFrame(gdtc_table_rows), use_container_width=True, hide_index=True)

        with tab4:
            st.markdown("### 📈 Bảng Lịch sử Học tập & Tiến trình ĐTBCHK/ĐTBCTL qua các Học kỳ")
            st.dataframe(sem_df, use_container_width=True, hide_index=True)

            if len(sem_df) >= 2:
                # Đổi tên cột tạm thời để rút gọn tên trên tooltip và legend
                plot_df = sem_df.rename(columns={
                    "ĐTBCHK (Thang 4)": "ĐTBCHK",
                    "ĐTBCTL (Thang 4)": "ĐTBCTL"
                })
                
                fig = px.line(
                    plot_df, 
                    x="Học kỳ", 
                    y=["ĐTBCHK", "ĐTBCTL"],
                    markers=True,
                    title="Diễn biến ĐTBCHK và ĐTBCTL qua các học kỳ (Thang điểm 4)"
                )
                
                # Định dạng lại tooltip để chỉ hiện giá trị số, bỏ các nhãn 'variable=' và 'value='
                fig.update_traces(hovertemplate="%{y:.2f}")
                
                # Loại bỏ tiêu đề trục hoành trong phần tooltip (chữ Học kỳ=Học kỳ x)
                fig.update_layout(
                    yaxis_range=[0, 4.0], 
                    hovermode="x unified",
                    hoverlabel=dict(namelength=-1)
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.markdown("### 🤖 Cố vấn Học tập AI")
            st.caption("Tự động phân tích toàn bộ hồ sơ học tập và đưa ra lời khuyên cá nhân hóa trực tiếp.")

            user_q = st.text_input("Đặt câu hỏi cho AI Cố vấn (tùy chọn):", placeholder="Ví dụ: Em nên đăng ký bao nhiêu tín chỉ học kỳ tới? Em bị nợ môn CT177 thì nên xử lý thế nào?")

            if st.button("🚀 Yêu cầu AI Tư vấn Lộ trình", key="btn_ask_groq"):
                with st.spinner("Groq AI đang phân tích toàn bộ dữ liệu học tập của bạn..."):
                    active_key_to_use = load_groq_api_key()
                    if is_admin() and api_key_input.strip():
                        active_key_to_use = api_key_input.strip()

                    advice = get_chatgpt_advice(
                        mssv=selected_mssv,
                        class_code=ma_lop,
                        major_info=major_info,
                        major_key=major_key,
                        batch_code=batch_code,
                        meta=meta,
                        cpa_10=cpa_10,
                        cpa_4=cpa_4,
                        dtbchk_4=dtbchk_4,
                        tc_earned=tc_earned_total,
                        tc_no_dong=tc_no_dong,
                        passed_courses_list=list(passed_courses),
                        failed_courses_list=list(failed_courses),
                        low_grade_courses_list=list(low_grade_courses),
                        rec_mandatory=rec_results["mandatory"],
                        rec_elective=rec_results["elective"],
                        rec_projects=rec_results["projects"],
                        gdtc_summary_list=rec_results["gdtc_summary"],
                        fl_data=fl_data,
                        gdqp_data=gdqp_data,
                        api_key=active_key_to_use,
                        user_question=user_q,
                        prompt_text=load_prompt_template(),
                        prompt_version=get_prompt_version()
                    )
                    st.markdown(advice)

else:
    st.error("Không tìm thấy dữ liệu sinh viên trong tập dữ liệu (Wide Matrix CSV). Vui lòng kiểm tra lại đường dẫn file.")