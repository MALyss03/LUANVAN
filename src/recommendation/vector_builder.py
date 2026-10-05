"""
vector_builder.py
=================

Xây dựng course-grade vectors cho hệ thống Personalized Learning Path.

Thiết kế:
    transcript
        -> chuẩn hóa mã học phần
        -> mapping course_groups.py
        -> course schema dùng chung
        -> vector điểm
        -> mask đã học/chưa học
        -> vector theo domain
        -> metadata phục vụ similarity + XAI

Nguyên tắc quan trọng:
1. Mọi sinh viên trong cùng một cohort/domain phải dùng CÙNG course_schema.
2. Chưa học = NaN + observed_mask = 0, KHÔNG biến thành điểm 0.
3. Điểm không hợp lệ được chuyển thành NaN.
4. Similarity chỉ nên tính trên các chiều mà cả hai sinh viên đều đã học.
5. course_groups.py chịu trách nhiệm taxonomy; module này chịu trách nhiệm vector hóa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.recommendation.course_groups import (
    COURSE_GROUPS,
    COURSES,
    DOMAIN_BUNDLES,
    build_course_vector_schema,
    get_course,
    get_course_groups,
    get_domain_courses,
    get_group_weight,
    infer_group,
    normalize_code,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_SCORE_MIN = 0.0
DEFAULT_SCORE_MAX = 10.0

# Các tên cột thường gặp trong dataset.
STUDENT_ID_ALIASES = (
    "student_id",
    "studentid",
    "student_code",
    "student_code_id",
    "mssv",
    "ma_sv",
    "ma sinh vien",
    "masv",
)

COURSE_CODE_ALIASES = (
    "course_code",
    "coursecode",
    "course_id",
    "course",
    "ma_hp",
    "ma hoc phan",
    "mahocphan",
)

COURSE_NAME_ALIASES = (
    "course_name",
    "coursename",
    "course_title",
    "ten_hp",
    "ten hoc phan",
    "tenhocphan",
)

SCORE_ALIASES = (
    "score",
    "grade",
    "final_score",
    "numeric_grade",
    "diem",
    "diem_hp",
    "diem_thi",
)

SEMESTER_ALIASES = (
    "semester",
    "hoc_ky",
    "hoc ky",
    "hk",
    "term",
)

# Một số dataset có điểm chữ. Chỉ map các ký hiệu phổ biến.
LETTER_GRADE_MAP = {
    "A+": 10.0,
    "A": 9.0,
    "B+": 8.5,
    "B": 8.0,
    "C+": 7.0,
    "C": 6.5,
    "D+": 5.5,
    "D": 5.0,
    "F": 0.0,
}


# ============================================================
# Data class
# ============================================================

@dataclass
class StudentVector:
    """
    Vector của một sinh viên.

    values:
        Điểm theo course_schema; môn chưa học = NaN.

    observed_mask:
        1 nếu sinh viên đã có điểm hợp lệ, 0 nếu chưa học/không quan sát.

    course_schema:
        Thứ tự chiều vector.

    metadata:
        Thông tin phục vụ recommendation/XAI.
    """

    student_id: str
    values: pd.Series
    observed_mask: pd.Series
    course_schema: list[str]
    metadata: dict[str, Any]


# ============================================================
# Generic helpers
# ============================================================

def _normalize_column_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _find_column(
    columns: Iterable[str],
    aliases: Sequence[str],
) -> str | None:
    normalized = {
        _normalize_column_name(col): col
        for col in columns
    }

    for alias in aliases:
        key = _normalize_column_name(alias)
        if key in normalized:
            return normalized[key]

    # fallback: contains
    for key, original in normalized.items():
        for alias in aliases:
            alias_key = _normalize_column_name(alias)
            if alias_key in key:
                return original

    return None


def _clean_student_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    # Tránh MSSV dạng 1234567.0
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    return text


def _parse_score(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        text = value.strip().upper()

        if text in LETTER_GRADE_MAP:
            return LETTER_GRADE_MAP[text]

        # Loại dấu phẩy/thừa khoảng trắng
        text = text.replace(",", ".")

        try:
            value = float(text)
        except ValueError:
            return np.nan

    try:
        score = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not DEFAULT_SCORE_MIN <= score <= DEFAULT_SCORE_MAX:
        return np.nan

    return score


def normalize_transcript(
    df: pd.DataFrame,
    student_col: str | None = None,
    course_code_col: str | None = None,
    score_col: str | None = None,
    course_name_col: str | None = None,
    semester_col: str | None = None,
) -> pd.DataFrame:
    """
    Chuẩn hóa transcript dạng long.

    Input tối thiểu:
        student_id | course_code | score

    Có thể có:
        course_name | semester

    Output chuẩn:
        student_id
        course_code
        score
        course_name
        semester
        course_group
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame")

    if df.empty:
        raise ValueError("Transcript đang rỗng.")

    student_col = student_col or _find_column(
        df.columns, STUDENT_ID_ALIASES
    )
    course_code_col = course_code_col or _find_column(
        df.columns, COURSE_CODE_ALIASES
    )
    score_col = score_col or _find_column(
        df.columns, SCORE_ALIASES
    )
    course_name_col = course_name_col or _find_column(
        df.columns, COURSE_NAME_ALIASES
    )
    semester_col = semester_col or _find_column(
        df.columns, SEMESTER_ALIASES
    )

    missing = []

    if student_col is None:
        missing.append("student_id")
    if course_code_col is None:
        missing.append("course_code")
    if score_col is None:
        missing.append("score")

    if missing:
        raise ValueError(
            "Không tìm thấy các cột bắt buộc: "
            + ", ".join(missing)
            + ". "
            "Hãy truyền rõ student_col/course_code_col/score_col."
        )

    result = pd.DataFrame()

    result["student_id"] = (
        df[student_col].map(_clean_student_id)
    )

    result["course_code"] = (
        df[course_code_col]
        .map(normalize_code)
    )

    result["score"] = (
        df[score_col]
        .map(_parse_score)
    )

    if course_name_col is not None:
        result["course_name"] = (
            df[course_name_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        result["course_name"] = ""

    if semester_col is not None:
        result["semester"] = df[semester_col]
    else:
        result["semester"] = np.nan

    result = result[
        (result["student_id"] != "")
        & (result["course_code"] != "")
    ].copy()

    result["course_group"] = [
        infer_group(code, name)
        for code, name in zip(
            result["course_code"],
            result["course_name"],
        )
    ]

    # Nếu một sinh viên có nhiều record cho cùng học phần,
    # giữ lần xuất hiện có điểm cao nhất.
    result = (
        result.sort_values(
            ["student_id", "course_code", "score"],
            na_position="first",
        )
        .drop_duplicates(
            ["student_id", "course_code"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# Course schema
# ============================================================

def build_schema_from_transcript(
    transcript: pd.DataFrame,
    domain: str | None = None,
    course_codes: Sequence[str] | None = None,
) -> list[str]:
    """
    Tạo schema cố định dùng chung cho toàn bộ sinh viên.

    Nếu course_codes được truyền:
        dùng chính danh sách đó.

    Nếu không:
        lấy các mã học phần xuất hiện trong transcript.

    domain:
        ví dụ:
            "web_development"
            "network_engineering"
            "cybersecurity"
            "data_ai"
    """

    if course_codes is None:
        course_codes = (
            transcript["course_code"]
            .dropna()
            .map(normalize_code)
            .unique()
            .tolist()
        )

    return build_course_vector_schema(
        list(course_codes),
        domain=domain,
    )


def build_global_schema(
    transcripts: Sequence[pd.DataFrame],
    domain: str | None = None,
) -> list[str]:
    """
    Tạo schema từ nhiều dataset/transcript.

    Dùng khi cần đảm bảo toàn bộ tập train/validation/test
    dùng cùng một không gian vector.
    """

    codes: set[str] = set()

    for df in transcripts:
        if "course_code" in df.columns:
            codes.update(
                df["course_code"]
                .dropna()
                .map(normalize_code)
                .tolist()
            )

    return build_course_vector_schema(
        sorted(codes),
        domain=domain,
    )


# ============================================================
# Single student vector
# ============================================================

def build_student_vector(
    transcript: pd.DataFrame,
    student_id: str,
    course_schema: Sequence[str],
) -> StudentVector:
    """
    Xây vector cho một sinh viên theo schema đã cố định.
    """

    student_id = _clean_student_id(student_id)

    student_df = transcript[
        transcript["student_id"].astype(str) == student_id
    ].copy()

    if student_df.empty:
        raise ValueError(
            f"Không tìm thấy student_id={student_id}"
        )

    series = (
        student_df
        .set_index("course_code")["score"]
        .reindex(list(course_schema))
    )

    observed = series.notna().astype(np.int8)

    group_by_course = {
        code: infer_group(
            code,
            (
                student_df.loc[
                    student_df["course_code"] == code,
                    "course_name",
                ].iloc[0]
                if not student_df.loc[
                    student_df["course_code"] == code
                ].empty
                else None
            ),
        )
        for code in course_schema
    }

    completed_courses = [
        code for code in course_schema
        if observed.loc[code] == 1
    ]

    metadata = {
        "n_courses_in_schema": len(course_schema),
        "n_completed_courses": len(completed_courses),
        "n_unobserved_courses": (
            len(course_schema) - len(completed_courses)
        ),
        "completion_ratio": (
            len(completed_courses) / len(course_schema)
            if course_schema
            else 0.0
        ),
        "completed_courses": completed_courses,
        "course_groups": group_by_course,
    }

    return StudentVector(
        student_id=student_id,
        values=series.astype(float),
        observed_mask=observed,
        course_schema=list(course_schema),
        metadata=metadata,
    )


# ============================================================
# All-student matrix
# ============================================================

def build_student_matrix(
    transcript: pd.DataFrame,
    course_schema: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trả về:

    values:
        index = student_id
        columns = course_schema
        value = score / NaN

    mask:
        index = student_id
        columns = course_schema
        value = 1 đã học, 0 chưa học
    """

    if transcript.empty:
        raise ValueError("Transcript đang rỗng.")

    pivot = transcript.pivot_table(
        index="student_id",
        columns="course_code",
        values="score",
        aggfunc="max",
    )

    pivot = pivot.reindex(
        columns=list(course_schema)
    )

    pivot.index = pivot.index.map(_clean_student_id)
    pivot.columns.name = None

    mask = pivot.notna().astype(np.int8)

    return pivot.astype(float), mask


# ============================================================
# Domain vectors
# ============================================================

def build_domain_vector(
    student_vector: StudentVector,
    domain: str,
) -> StudentVector:
    """
    Lấy ra vector con theo một domain.

    Ví dụ:
        web_development

    Chỉ giữ các học phần thuộc domain.
    """

    domain_courses = get_domain_courses(domain)

    schema = [
        code
        for code in student_vector.course_schema
        if code in domain_courses
    ]

    values = student_vector.values.reindex(schema)
    mask = student_vector.observed_mask.reindex(schema)

    metadata = dict(student_vector.metadata)

    metadata["domain"] = domain
    metadata["domain_courses"] = schema
    metadata["domain_completed_courses"] = [
        code for code in schema if mask.loc[code] == 1
    ]

    return StudentVector(
        student_id=student_vector.student_id,
        values=values,
        observed_mask=mask,
        course_schema=schema,
        metadata=metadata,
    )


# ============================================================
# Similarity-ready pair
# ============================================================

def align_for_similarity(
    a: StudentVector,
    b: StudentVector,
    min_common_courses: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Chỉ lấy các môn mà cả A và B đều đã có điểm.

    Đây là bước bắt buộc trước cosine/Pearson/RMSE.

    Không thay NaN bằng 0.
    """

    if a.course_schema != b.course_schema:
        raise ValueError(
            "Hai vector không dùng cùng course_schema."
        )

    common_mask = (
        a.observed_mask.astype(bool)
        & b.observed_mask.astype(bool)
    )

    common_courses = [
        code
        for code in a.course_schema
        if common_mask.loc[code]
    ]

    if len(common_courses) < min_common_courses:
        raise ValueError(
            f"Chỉ có {len(common_courses)} môn chung, "
            f"cần ít nhất {min_common_courses}."
        )

    a_values = (
        a.values.loc[common_courses]
        .to_numpy(dtype=float)
    )

    b_values = (
        b.values.loc[common_courses]
        .to_numpy(dtype=float)
    )

    return a_values, b_values, common_courses


# ============================================================
# Weighted vectors
# ============================================================

def get_course_weight(code: str) -> float:
    """
    Trọng số của course dựa trên primary group.
    """

    meta = get_course(code)

    if meta is None:
        return 1.0

    return get_group_weight(
        meta["primary_group"]
    )


def build_weight_vector(
    course_schema: Sequence[str],
) -> pd.Series:
    """
    Tạo vector trọng số song song với course_schema.
    """

    return pd.Series(
        {
            code: get_course_weight(code)
            for code in course_schema
        },
        dtype=float,
    )


# ============================================================
# Recommendation candidates
# ============================================================

def get_unlearned_courses(
    student_vector: StudentVector,
    domain: str | None = None,
) -> list[str]:
    """
    Các môn chưa học của sinh viên.

    Nếu domain được chỉ định:
        chỉ trả môn chưa học trong domain đó.
    """

    if domain is not None:
        allowed = set(get_domain_courses(domain))
    else:
        allowed = set(student_vector.course_schema)

    return [
        code
        for code in student_vector.course_schema
        if code in allowed
        and student_vector.observed_mask.loc[code] == 0
    ]


def get_completed_courses(
    student_vector: StudentVector,
    domain: str | None = None,
) -> list[str]:
    """
    Các môn đã học của sinh viên.
    """

    if domain is not None:
        allowed = set(get_domain_courses(domain))
    else:
        allowed = set(student_vector.course_schema)

    return [
        code
        for code in student_vector.course_schema
        if code in allowed
        and student_vector.observed_mask.loc[code] == 1
    ]


# ============================================================
# Group-level profile
# ============================================================

def build_group_profile(
    student_vector: StudentVector,
) -> pd.DataFrame:
    """
    Tổng hợp profile theo nhóm kiến thức.

    Output:
        group
        label
        mean_score
        course_count
        observed_count
        completion_ratio
        weighted_score

    Đây là dữ liệu rất hữu ích cho XAI.
    """

    rows = []

    for code in student_vector.course_schema:
        if student_vector.observed_mask.loc[code] == 0:
            continue

        meta = get_course(code)

        if meta is None:
            group = infer_group(code)
            name = code
        else:
            group = meta["primary_group"]
            name = meta["name"]

        rows.append(
            {
                "course_code": code,
                "course_name": name,
                "group": group,
                "group_label": COURSE_GROUPS[group]["label"],
                "score": float(
                    student_vector.values.loc[code]
                ),
                "weight": get_group_weight(group),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group",
                "group_label",
                "mean_score",
                "course_count",
                "observed_count",
                "completion_ratio",
                "weighted_score",
            ]
        )

    df = pd.DataFrame(rows)

    grouped = (
        df.groupby(
            ["group", "group_label"],
            as_index=False,
        )
        .agg(
            mean_score=("score", "mean"),
            course_count=("course_code", "count"),
            observed_count=("score", "count"),
            weight=("weight", "first"),
        )
    )

    # completion_ratio ở đây được tính trên các course
    # xuất hiện trong vector của group.
    group_total = (
        pd.Series(
            {
                group: sum(
                    1
                    for code in student_vector.course_schema
                    if infer_group(code) == group
                )
                for group in grouped["group"]
            },
            dtype=float,
        )
    )

    grouped["completion_ratio"] = [
        (
            row["observed_count"]
            / group_total.get(row["group"], 1)
        )
        for _, row in grouped.iterrows()
    ]

    grouped["weighted_score"] = (
        grouped["mean_score"]
        * grouped["weight"]
    )

    return grouped


# ============================================================
# Batch preparation
# ============================================================

def prepare_recommendation_vectors(
    raw_df: pd.DataFrame,
    domain: str | None = None,
    course_schema: Sequence[str] | None = None,
    student_col: str | None = None,
    course_code_col: str | None = None,
    score_col: str | None = None,
    course_name_col: str | None = None,
    semester_col: str | None = None,
) -> dict[str, Any]:
    """
    Pipeline tiện dụng:

        raw DataFrame
            -> normalize
            -> schema
            -> matrix
            -> mask
            -> metadata

    Trả về dict để recommendation engine sử dụng.
    """

    transcript = normalize_transcript(
        raw_df,
        student_col=student_col,
        course_code_col=course_code_col,
        score_col=score_col,
        course_name_col=course_name_col,
        semester_col=semester_col,
    )

    if course_schema is None:
        course_schema = build_schema_from_transcript(
            transcript,
            domain=domain,
        )
    else:
        course_schema = build_course_vector_schema(
            list(course_schema),
            domain=domain,
        )

    values, mask = build_student_matrix(
        transcript,
        course_schema,
    )

    weights = build_weight_vector(
        course_schema
    )

    return {
        "transcript": transcript,
        "course_schema": course_schema,
        "values": values,
        "observed_mask": mask,
        "weights": weights,
        "domain": domain,
        "n_students": len(values),
        "n_courses": len(course_schema),
    }


# ============================================================
# Export helpers
# ============================================================

def matrix_to_long(
    values: pd.DataFrame,
    observed_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Chuyển matrix về dạng long để lưu/debug.

    Nếu có mask:
        thêm cột observed.
    """

    long_df = (
        values
        .rename_axis("student_id")
        .reset_index()
        .melt(
            id_vars="student_id",
            var_name="course_code",
            value_name="score",
        )
    )

    if observed_mask is not None:
        mask_long = (
            observed_mask
            .rename_axis("student_id")
            .reset_index()
            .melt(
                id_vars="student_id",
                var_name="course_code",
                value_name="observed",
            )
        )

        long_df["observed"] = (
            mask_long["observed"].astype(np.int8)
        )

    return long_df


def save_vector_artifacts(
    output_dir: str | Any,
    values: pd.DataFrame,
    observed_mask: pd.DataFrame,
    prefix: str = "course_vectors",
) -> dict[str, str]:
    """
    Lưu vector matrix + observed mask.

    Không ghi đè dữ liệu raw.
    """

    output_dir = __import__("pathlib").Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values_path = (
        output_dir / f"{prefix}_scores.csv"
    )
    mask_path = (
        output_dir / f"{prefix}_mask.csv"
    )

    values.to_csv(
        values_path,
        encoding="utf-8-sig",
    )

    observed_mask.to_csv(
        mask_path,
        encoding="utf-8-sig",
    )

    return {
        "scores": str(values_path),
        "mask": str(mask_path),
    }


# ============================================================
# Self-check
# ============================================================

def validate_schema(
    values: pd.DataFrame,
    observed_mask: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kiểm tra các invariant quan trọng trước khi đưa vào similarity.
    """

    if list(values.columns) != list(
        observed_mask.columns
    ):
        raise ValueError(
            "values và observed_mask không cùng schema."
        )

    if list(values.index) != list(
        observed_mask.index
    ):
        raise ValueError(
            "values và observed_mask không cùng student index."
        )

    invalid_observed = ~observed_mask.isin([0, 1]).all().all()

    if invalid_observed:
        raise ValueError(
            "observed_mask phải chỉ chứa 0/1."
        )

    # Các ô observed=1 phải có score.
    bad_observed = (
        observed_mask.eq(1)
        & values.isna()
    ).any().any()

    if bad_observed:
        raise ValueError(
            "Có observed=1 nhưng score lại NaN."
        )

    return {
        "valid": True,
        "n_students": len(values),
        "n_courses": len(values.columns),
        "total_observations": int(
            observed_mask.to_numpy().sum()
        ),
        "sparsity": float(
            1.0
            - (
                observed_mask.to_numpy().sum()
                / max(observed_mask.size, 1)
            )
        ),
    }


__all__ = [
    "StudentVector",
    "normalize_transcript",
    "build_schema_from_transcript",
    "build_global_schema",
    "build_student_vector",
    "build_student_matrix",
    "build_domain_vector",
    "align_for_similarity",
    "get_course_weight",
    "build_weight_vector",
    "get_unlearned_courses",
    "get_completed_courses",
    "build_group_profile",
    "prepare_recommendation_vectors",
    "matrix_to_long",
    "save_vector_artifacts",
    "validate_schema",
]
