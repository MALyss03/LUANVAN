"""
course_groups.py
----------------
Taxonomy học phần dùng cho Personalized Learning Path Recommendation.

Nguồn xây dựng:
- Các chương trình đào tạo CTU mà người dùng cung cấp:
  Mạng máy tính & Truyền thông dữ liệu, An toàn thông tin,
  và MMT-TTDL Chất lượng cao.
- Bao quát các mã học phần xuất hiện trong các phiên bản chương trình
  được cung cấp cho khóa 47-51.

Mục đích:
1. Chuẩn hóa học phần theo nhóm kiến thức.
2. Xây vector điểm theo cùng một thứ tự chiều cho mọi sinh viên.
3. Hỗ trợ tìm sinh viên tương đồng.
4. Hỗ trợ XAI: giải thích môn nào/nhóm kiến thức nào đóng góp vào đề xuất.

Lưu ý:
- Đây là lớp taxonomy, không thay thế curriculum_engine.py.
- Điều kiện tiên quyết, bắt buộc/tự chọn, khóa học và khả năng đăng ký
  phải tiếp tục lấy từ dữ liệu CTĐT chính thức.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ============================================================
# 1. NHÓM KIẾN THỨC
# ============================================================

COURSE_GROUPS: Dict[str, dict] = {
    "programming_foundation": {
        "label": "Lập trình nền tảng",
        "description": "Lập trình căn bản, hướng đối tượng và kỹ năng coding nền tảng.",
        "weight": 1.00,
    },
    "algorithms_data_structures": {
        "label": "Cấu trúc dữ liệu & Thuật toán",
        "description": "Cấu trúc dữ liệu, thuật toán, lý thuyết đồ thị và nền tảng tính toán.",
        "weight": 1.00,
    },
    "web_programming": {
        "label": "Lập trình Web",
        "description": "Phát triển Web và ứng dụng Web.",
        "weight": 1.15,
    },
    "software_engineering": {
        "label": "Công nghệ phần mềm",
        "description": "Phân tích thiết kế hệ thống, mô hình hóa và phát triển phần mềm.",
        "weight": 0.95,
    },
    "database": {
        "label": "Cơ sở dữ liệu",
        "description": "Cơ sở dữ liệu, quản trị cơ sở dữ liệu và quản trị dữ liệu.",
        "weight": 1.00,
    },
    "computer_systems": {
        "label": "Hệ thống máy tính",
        "description": "Kiến trúc máy tính, hệ điều hành và quản trị hệ thống.",
        "weight": 0.95,
    },
    "computer_networks": {
        "label": "Mạng máy tính",
        "description": "Mạng máy tính, thiết kế, quản trị, hiệu năng và mạng không dây.",
        "weight": 1.15,
    },
    "network_programming": {
        "label": "Lập trình mạng",
        "description": "Phát triển phần mềm/ứng dụng trên nền mạng.",
        "weight": 1.10,
    },
    "network_security": {
        "label": "An ninh mạng",
        "description": "Tấn công mạng, phòng thủ mạng, firewall, sự cố và bảo vệ hệ thống mạng.",
        "weight": 1.20,
    },
    "cybersecurity": {
        "label": "An toàn thông tin",
        "description": "Bảo mật hệ thống, mã độc, pháp y, an toàn dữ liệu và security engineering.",
        "weight": 1.20,
    },
    "cryptography": {
        "label": "Mật mã",
        "description": "Mật mã và nền tảng lý thuyết thông tin.",
        "weight": 1.10,
    },
    "web_security": {
        "label": "An ninh Web",
        "description": "Bảo mật ứng dụng và hệ thống Web.",
        "weight": 1.20,
    },
    "iot": {
        "label": "Internet of Things",
        "description": "Phát triển ứng dụng IoT và bảo mật IoT.",
        "weight": 1.10,
    },
    "cloud_distributed": {
        "label": "Cloud & Tính toán phân tán",
        "description": "Cloud, hệ thống hướng dịch vụ và tính toán song song.",
        "weight": 1.00,
    },
    "big_data": {
        "label": "Dữ liệu lớn",
        "description": "Xử lý, phân lớp và khai thác dữ liệu lớn.",
        "weight": 1.05,
    },
    "artificial_intelligence": {
        "label": "AI & Machine Learning",
        "description": "Trí tuệ nhân tạo và máy học ứng dụng.",
        "weight": 1.00,
    },
    "mobile_development": {
        "label": "Lập trình di động",
        "description": "Phát triển ứng dụng trên thiết bị di động.",
        "weight": 0.95,
    },
    "blockchain": {
        "label": "Blockchain",
        "description": "Blockchain và công nghệ sổ cái phân tán.",
        "weight": 0.90,
    },
    "human_computer_interaction": {
        "label": "Tương tác người-máy",
        "description": "Giao diện và tương tác người-máy.",
        "weight": 0.85,
    },
    "mathematics_theory": {
        "label": "Toán & Cơ sở lý thuyết",
        "description": "Toán, xác suất, phương pháp tính, lý thuyết thông tin.",
        "weight": 0.80,
    },
    "data_science": {
        "label": "Khoa học dữ liệu",
        "description": "Xử lý và phân tích dữ liệu.",
        "weight": 1.00,
    },
    "professional_practice": {
        "label": "Thực tập & Đồ án",
        "description": "Niên luận, thực tập, tiểu luận và luận văn.",
        "weight": 0.70,
    },
    "management_professional": {
        "label": "Quản lý & Nghề nghiệp",
        "description": "Quản lý dự án, pháp luật nghề nghiệp và kỹ năng nghề.",
        "weight": 0.65,
    },
    "general_education": {
        "label": "Giáo dục đại cương",
        "description": "Các học phần đại cương/điều kiện không nên chi phối similarity chuyên ngành.",
        "weight": 0.20,
    },
}


# ============================================================
# 2. MÃ HỌC PHẦN → NHÓM KIẾN THỨC
# ============================================================

COURSES: Dict[str, dict] = {}


def _add(
    code: str,
    name: str,
    primary_group: str,
    credits: float | None = None,
    secondary_groups: Tuple[str, ...] = (),
) -> None:
    COURSES[code.upper()] = {
        "name": name,
        "credits": credits,
        "primary_group": primary_group,
        "secondary_groups": secondary_groups,
    }


# ---------- Lập trình / thuật toán ----------
_add("CT101", "Lập trình căn bản A", "programming_foundation", 4)
_add("CT176", "Lập trình hướng đối tượng", "programming_foundation", 3)
_add("CT177", "Cấu trúc dữ liệu", "algorithms_data_structures", 3)
_add("CT175", "Lý thuyết đồ thị", "algorithms_data_structures", 3)
_add("CT174", "Phân tích và thiết kế thuật toán", "algorithms_data_structures", 3)
_add("CT172", "Toán rời rạc", "mathematics_theory", 4, ("algorithms_data_structures",))
_add("CT172E", "Toán rời rạc", "mathematics_theory", 4, ("algorithms_data_structures",))
_add("CT121", "Tin học lý thuyết", "algorithms_data_structures", 3, ("mathematics_theory",))

# ---------- Web ----------
_add("CT188", "Nhập môn lập trình Web", "web_programming", 3, ("programming_foundation",))
_add("CT428", "Lập trình Web", "web_programming", 3, ("programming_foundation", "software_engineering"))
_add("CT490", "An ninh Web", "web_security", 3, ("web_programming", "cybersecurity"))

# ---------- Phần mềm ----------
_add("CT296", "Phân tích và thiết kế hệ thống thông tin", "software_engineering", 3, ("database",))
_add("CT182", "Ngôn ngữ mô hình hóa", "software_engineering", 3)
_add("CT224", "Công nghệ J2EE", "software_engineering", 2, ("web_programming",))
_add("CT224E", "Công nghệ J2EE", "software_engineering", 2, ("web_programming",))
_add("CT207", "Phát triển phần mềm mã nguồn mở", "software_engineering", 3)
_add("CT207E", "Phát triển phần mềm mã nguồn mở", "software_engineering", 3)
_add("CT230", "Phát triển ứng dụng hướng dịch vụ", "software_engineering", 3, ("cloud_distributed",))
_add("CT251", "Phát triển ứng dụng trên Windows", "software_engineering", 3, ("programming_foundation",))
_add("CT206", "Phát triển ứng dụng trên Linux", "software_engineering", 3, ("computer_systems", "programming_foundation"))
_add("CT223", "Quản lý dự án phần mềm", "management_professional", 3, ("software_engineering",))

# ---------- Database / Data ----------
_add("CT180", "Cơ sở dữ liệu", "database", 3)
_add("CT205", "Quản trị cơ sở dữ liệu", "database", 3)
_add("CT467", "Quản trị dữ liệu", "database", 3)
_add("CT238", "Phân lớp dữ liệu lớn", "big_data", 3, ("data_science",))
_add("CT238E", "Phân lớp dữ liệu lớn", "big_data", 3, ("data_science",))
_add("CT482", "Xử lý dữ liệu lớn", "big_data", 3, ("data_science",))

# ---------- Hệ thống ----------
_add("CT173", "Kiến trúc máy tính", "computer_systems", 3)
_add("CT178", "Nguyên lý hệ điều hành", "computer_systems", 3)
_add("CT179", "Quản trị hệ thống", "computer_systems", 3)
_add("CT200", "Nền tảng công nghệ thông tin", "computer_systems", 4)
_add("CT005", "Nền tảng công nghệ số", "computer_systems", 4)

# ---------- Mạng ----------
_add("CT112", "Mạng máy tính", "computer_networks", 3)
_add("CT335", "Thiết kế và cài đặt mạng", "computer_networks", 3)
_add("CT212", "Quản trị mạng", "computer_networks", 3)
_add("CT235", "Quản trị mạng trên MS Windows", "computer_networks", 3, ("computer_systems",))
_add("CT338", "Mạng không dây và di động", "computer_networks", 2, ("mobile_development",))
_add("CT232", "Đánh giá hiệu năng mạng", "computer_networks", 3, ("mathematics_theory",))
_add("CT232E", "Đánh giá hiệu năng mạng", "computer_networks", 3, ("mathematics_theory",))
_add("CT344", "Giải quyết sự cố mạng", "computer_networks", 2, ("network_security",))
_add("CT221", "Lập trình mạng", "network_programming", 3, ("computer_networks", "programming_foundation"))

# ---------- An ninh / ATTT ----------
_add("CT211", "An ninh mạng", "network_security", 3, ("cybersecurity", "computer_networks"))
_add("CT197", "Cơ sở lý thuyết mật mã", "cryptography", 3, ("cybersecurity", "mathematics_theory"))
_add("CT127", "Lý thuyết thông tin", "cryptography", 2, ("mathematics_theory",))
_add("CT485", "Các kỹ thuật tấn công mạng", "network_security", 3, ("cybersecurity",))
_add("CT227", "Kỹ thuật phát hiện tấn công mạng", "network_security", 3, ("computer_networks",))
_add("CT227E", "Kỹ thuật phát hiện tấn công mạng", "network_security", 3, ("computer_networks",))
_add("CT228", "Tường lửa", "network_security", 3, ("computer_networks", "cybersecurity"))
_add("CT222", "An toàn hệ thống", "cybersecurity", 3, ("computer_systems", "network_security"))
_add("CT486", "Phát hiện và phân tích mã độc", "cybersecurity", 3, ("network_security",))
_add("CT297", "Pháp y máy tính", "cybersecurity", 3, ("network_security",))
_add("CT489", "Luật an ninh mạng và đạo đức trong lĩnh vực CNTT", "cybersecurity", 2, ("management_professional",))
_add("CT489E", "Luật an ninh mạng và đạo đức trong lĩnh vực CNTT", "cybersecurity", 2, ("management_professional",))

# ---------- IoT / Mobile / Blockchain ----------
_add("CT098", "Phát triển ứng dụng IoT", "iot", 3, ("programming_foundation", "computer_networks"))
_add("CT488", "Bảo mật hệ thống IoT", "cybersecurity", 2, ("iot", "network_security"))
_add("CT274", "Lập trình cho thiết bị di động", "mobile_development", 3, ("programming_foundation",))
_add("CT279", "Blockchain", "blockchain", 3, ("cryptography", "cloud_distributed"))

# ---------- Cloud / Distributed ----------
_add("CT233", "Điện toán đám mây", "cloud_distributed", 3, ("computer_networks",))
_add("CT233E", "Điện toán đám mây", "cloud_distributed", 3)
_add("CT231", "Lập trình song song", "cloud_distributed", 3, ("programming_foundation",))

# ---------- AI ----------
_add("CT190", "Nhập môn trí tuệ nhân tạo", "artificial_intelligence", 2)
_add("CT190E", "Nhập môn trí tuệ nhân tạo", "artificial_intelligence", 2)
_add("CT294", "Máy học ứng dụng", "artificial_intelligence", 3, ("data_science",))
_add("CT332", "Trí tuệ nhân tạo", "artificial_intelligence", 3, ("data_science",))

# ---------- Toán / lý thuyết ----------
_add("CT126", "Lý thuyết xếp hàng", "mathematics_theory", 2, ("computer_networks",))
_add("CT479", "Phương pháp tính", "mathematics_theory", 3)

# ---------- Chuyên môn / thực tập / tốt nghiệp ----------
for _code, _name, _credits in [
    ("CT226E", "Niên luận cơ sở mạng máy tính và truyền thông", 3),
    ("CT439E", "Niên luận mạng máy tính và truyền thông", 3),
    ("CT517E", "Thực tập doanh nghiệp - MMT&TTDL", 5),
    ("CT555E", "Luận văn tốt nghiệp - TT&MMT", 15),
    ("CT507E", "Tiểu luận tốt nghiệp - TT&MMT", 6),
    ("CT491", "Niên luận cơ sở an toàn thông tin", 3),
    ("CT491E", "Niên luận cơ sở an toàn thông tin", 3),
    ("CT492", "Niên luận an toàn thông tin", 3),
    ("CT492E", "Niên luận an toàn thông tin", 3),
    ("CT493", "Thực tập doanh nghiệp - An toàn thông tin", 5),
    ("CT493E", "Thực tập doanh nghiệp - An toàn thông tin", 5),
    ("CT556", "Luận văn tốt nghiệp - ATTT", 15),
    ("CT556E", "Luận văn tốt nghiệp - ATTT", 15),
    ("CT520", "Tiểu luận tốt nghiệp - ATTT", 6),
    ("CT520E", "Tiểu luận tốt nghiệp - ATTT", 6),
]:
    _add(_code, _name, "professional_practice", _credits)


# ============================================================
# 3. HỌC PHẦN ĐẠI CƯƠNG
# ============================================================

_GENERAL = {
    "QP010E": ("Giáo dục quốc phòng và An ninh 1", 2),
    "QP011E": ("Giáo dục quốc phòng và An ninh 2", 2),
    "QP012": ("Giáo dục quốc phòng và An ninh 3", 2),
    "QP013": ("Giáo dục quốc phòng và An ninh 4", 2),
    "TC100": ("Giáo dục thể chất 1+2+3", 3),
    "XH023": ("Anh văn căn bản 1", 4),
    "XH024": ("Anh văn căn bản 2", 3),
    "XH025": ("Anh văn căn bản 3", 3),
    "XH031": ("Anh văn tăng cường 1", 4),
    "XH032": ("Anh văn tăng cường 2", 3),
    "XH033": ("Anh văn tăng cường 3", 3),
    "FL001": ("Pháp văn căn bản 1", 4),
    "FL002": ("Pháp văn căn bản 2", 3),
    "FL003": ("Pháp văn căn bản 3", 3),
    "FL007": ("Pháp văn tăng cường 1", 4),
    "FL008": ("Pháp văn tăng cường 2", 3),
    "FL009": ("Pháp văn tăng cường 3", 3),
    "ML014": ("Triết học Mác - Lênin", 3),
    "ML016": ("Kinh tế chính trị Mác - Lênin", 2),
    "ML018": ("Chủ nghĩa xã hội khoa học", 2),
    "ML019": ("Lịch sử Đảng Cộng sản Việt Nam", 2),
    "ML021": ("Tư tưởng Hồ Chí Minh", 2),
    "KL001E": ("Pháp luật đại cương", 2),
    "ML007": ("Logic học đại cương", 2),
    "XH028": ("Xã hội học đại cương", 2),
    "XH011": ("Cơ sở văn hóa Việt Nam", 2),
    "XH011E": ("Cơ sở văn hóa Việt Nam", 2),
    "XH012": ("Tiếng Việt thực hành", 2),
    "XH014": ("Văn bản và lưu trữ học đại cương", 2),
    "KN001": ("Kỹ năng mềm", 2),
    "KN001E": ("Kỹ năng mềm", 2),
    "KN002": ("Đổi mới sáng tạo và khởi nghiệp", 2),
    "KN002E": ("Đổi mới sáng tạo và khởi nghiệp", 2),
    "TN001": ("Vi - Tích phân A1", 3),
    "TN002": ("Vi - Tích phân A2", 4),
    "TN010": ("Xác suất thống kê", 3),
    "TN012": ("Đại số tuyến tính và hình học", 4),
    "CT100": ("Kỹ năng học đại học", 2),
}

for _code, (_name, _credits) in _GENERAL.items():
    _add(_code, _name, "general_education", _credits)


# ============================================================
# 4. GROUP → COURSES
# ============================================================

GROUP_TO_COURSES: Dict[str, List[str]] = {
    group: [] for group in COURSE_GROUPS
}

for _code, _meta in COURSES.items():
    GROUP_TO_COURSES[_meta["primary_group"]].append(_code)

    for _group in _meta["secondary_groups"]:
        GROUP_TO_COURSES[_group].append(_code)

for _group in GROUP_TO_COURSES:
    GROUP_TO_COURSES[_group] = sorted(set(GROUP_TO_COURSES[_group]))


# ============================================================
# 5. DOMAIN BUNDLES
#
# Đây là các "chuyên đề" để xây vector similarity.
# ============================================================

DOMAIN_BUNDLES: Dict[str, Tuple[str, ...]] = {
    "web_development": (
        "programming_foundation",
        "database",
        "web_programming",
        "software_engineering",
        "network_programming",
        "web_security",
    ),

    "network_engineering": (
        "computer_networks",
        "network_programming",
        "computer_systems",
        "network_security",
        "cloud_distributed",
    ),

    "cybersecurity": (
        "network_security",
        "cybersecurity",
        "cryptography",
        "web_security",
        "computer_networks",
        "computer_systems",
    ),

    "data_ai": (
        "database",
        "data_science",
        "big_data",
        "artificial_intelligence",
        "algorithms_data_structures",
    ),

    "iot": (
        "programming_foundation",
        "computer_networks",
        "iot",
        "network_security",
        "computer_systems",
    ),

    "software_development": (
        "programming_foundation",
        "algorithms_data_structures",
        "database",
        "web_programming",
        "software_engineering",
        "mobile_development",
        "cloud_distributed",
    ),
}


# ============================================================
# 6. KEYWORD FALLBACK
# ============================================================

KEYWORD_RULES: Tuple[Tuple[str, str], ...] = (
    ("an ninh web", "web_security"),
    ("an ninh mạng", "network_security"),
    ("an toàn hệ thống", "cybersecurity"),
    ("bảo mật", "cybersecurity"),
    ("mật mã", "cryptography"),
    ("mã độc", "cybersecurity"),
    ("pháp y máy tính", "cybersecurity"),
    ("tấn công mạng", "network_security"),
    ("tường lửa", "network_security"),
    ("mạng không dây", "computer_networks"),
    ("mạng máy tính", "computer_networks"),
    ("quản trị mạng", "computer_networks"),
    ("lập trình mạng", "network_programming"),
    ("lập trình web", "web_programming"),
    ("web", "web_programming"),
    ("cơ sở dữ liệu", "database"),
    ("quản trị cơ sở dữ liệu", "database"),
    ("quản trị dữ liệu", "database"),
    ("dữ liệu lớn", "big_data"),
    ("điện toán đám mây", "cloud_distributed"),
    ("lập trình song song", "cloud_distributed"),
    ("iot", "iot"),
    ("internet of things", "iot"),
    ("blockchain", "blockchain"),
    ("trí tuệ nhân tạo", "artificial_intelligence"),
    ("máy học", "artificial_intelligence"),
    ("lập trình cho thiết bị di động", "mobile_development"),
    ("cấu trúc dữ liệu", "algorithms_data_structures"),
    ("thuật toán", "algorithms_data_structures"),
    ("lý thuyết đồ thị", "algorithms_data_structures"),
    ("lập trình hướng đối tượng", "programming_foundation"),
    ("lập trình căn bản", "programming_foundation"),
    ("kiến trúc máy tính", "computer_systems"),
    ("hệ điều hành", "computer_systems"),
    ("quản trị hệ thống", "computer_systems"),
    ("phân tích và thiết kế hệ thống", "software_engineering"),
    ("ngôn ngữ mô hình hóa", "software_engineering"),
    ("phát triển phần mềm", "software_engineering"),
    ("quản lý dự án phần mềm", "management_professional"),
    ("niên luận", "professional_practice"),
    ("thực tập doanh nghiệp", "professional_practice"),
    ("luận văn", "professional_practice"),
    ("tiểu luận tốt nghiệp", "professional_practice"),
    ("xác suất thống kê", "mathematics_theory"),
    ("toán rời rạc", "mathematics_theory"),
    ("phương pháp tính", "mathematics_theory"),
    ("lý thuyết thông tin", "mathematics_theory"),
)


# ============================================================
# 7. API CHO RECOMMENDATION ENGINE
# ============================================================

def normalize_code(code: str) -> str:
    return str(code).strip().upper().replace(" ", "")


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def get_course(code: str) -> dict | None:
    return COURSES.get(normalize_code(code))


def infer_group(code: str | None = None, name: str | None = None) -> str:
    """
    Ưu tiên mapping theo mã học phần.
    Chỉ dùng keyword fallback khi mã chưa có trong taxonomy.
    """
    if code:
        meta = get_course(code)
        if meta:
            return meta["primary_group"]

    if name:
        text = normalize_name(name)
        for keyword, group in KEYWORD_RULES:
            if keyword in text:
                return group

    return "general_education"


def get_course_groups(code: str, name: str | None = None) -> Tuple[str, ...]:
    meta = get_course(code)

    if meta:
        return (
            meta["primary_group"],
            *meta["secondary_groups"],
        )

    return (infer_group(code, name),)


def get_group_weight(group: str) -> float:
    return float(
        COURSE_GROUPS.get(group, {}).get("weight", 1.0)
    )


def get_domain_courses(domain: str) -> List[str]:
    if domain not in DOMAIN_BUNDLES:
        raise KeyError(f"Unknown domain: {domain}")

    result: List[str] = []

    for group in DOMAIN_BUNDLES[domain]:
        result.extend(
            GROUP_TO_COURSES.get(group, [])
        )

    return sorted(set(result))


def build_course_vector_schema(
    course_codes: List[str],
    domain: str | None = None,
) -> List[str]:
    """
    Tạo thứ tự chiều vector dùng chung cho toàn bộ sinh viên.

    QUAN TRỌNG:
    Không được tạo vector bằng cách sort riêng từng sinh viên.
    Tất cả sinh viên phải dùng cùng course_schema.
    """
    allowed = (
        set(get_domain_courses(domain))
        if domain
        else None
    )

    result = []

    for code in course_codes:
        code = normalize_code(code)

        if allowed is not None and code not in allowed:
            continue

        result.append(code)

    return sorted(set(result))


def explain_course_group(
    code: str,
    name: str | None = None,
) -> dict:
    """
    Metadata dùng cho Dashboard/XAI.
    """
    meta = get_course(code)

    if meta:
        primary = meta["primary_group"]

        return {
            "course_code": normalize_code(code),
            "course_name": meta["name"],
            "primary_group": primary,
            "primary_label": COURSE_GROUPS[primary]["label"],
            "secondary_groups": list(
                meta["secondary_groups"]
            ),
            "group_weight": get_group_weight(primary),
            "known_course": True,
        }

    group = infer_group(code, name)

    return {
        "course_code": normalize_code(code),
        "course_name": name,
        "primary_group": group,
        "primary_label": COURSE_GROUPS[group]["label"],
        "secondary_groups": [],
        "group_weight": get_group_weight(group),
        "known_course": False,
    }


def is_specialized_course(
    code: str,
    name: str | None = None,
) -> bool:
    return infer_group(code, name) != "general_education"


__all__ = [
    "COURSE_GROUPS",
    "COURSES",
    "GROUP_TO_COURSES",
    "DOMAIN_BUNDLES",
    "KEYWORD_RULES",
    "normalize_code",
    "normalize_name",
    "get_course",
    "infer_group",
    "get_course_groups",
    "get_group_weight",
    "get_domain_courses",
    "build_course_vector_schema",
    "explain_course_group",
    "is_specialized_course",
]
