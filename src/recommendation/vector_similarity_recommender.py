from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Optional, Tuple
import math
import numpy as np
import pandas as pd

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None


# -----------------------------------------------------------------------------
# DOMAIN / COURSE GROUPING
# -----------------------------------------------------------------------------
# You can add exact course codes here when the curriculum dictionary is ready.
# Keyword matching is only the fallback; exact mappings always take precedence.
EXACT_COURSE_GROUPS: Dict[str, str] = {
    # Example:
    # "CT123": "web_programming",
}

COURSE_GROUP_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "web_programming": (
        "LẬP TRÌNH WEB", "LAP TRINH WEB", "LẬP TRÌNH ỨNG DỤNG WEB",
        "ỨNG DỤNG WEB", "UNG DUNG WEB", "CÔNG NGHỆ WEB", "CONG NGHE WEB",
        "THIẾT KẾ WEB", "THIET KE WEB", "WEB", "HTML", "CSS", "JAVASCRIPT",
        "PHP", "FRONTEND", "FRONT-END", "BACKEND", "BACK-END", "FULLSTACK",
        "FULL-STACK", "NODEJS", "REACT", "ANGULAR", "LARAVEL",
    ),
    "programming": (
        "LẬP TRÌNH", "LAP TRINH", "PROGRAMMING", "PYTHON", "JAVA", "C++",
        "C#", "ALGORITHM", "THUẬT TOÁN", "THUAT TOAN",
    ),
    "database": (
        "CƠ SỞ DỮ LIỆU", "CO SO DU LIEU", "DATABASE", "SQL", "MONGODB",
        "HỆ QUẢN TRỊ CSDL", "HE QUAN TRI CSDL",
    ),
    "network": (
        "MẠNG", "MANG", "NETWORK", "TCP/IP", "ĐỊNH TUYẾN", "DINH TUYEN",
        "WIRELESS", "KHÔNG DÂY", "KHONG DAY",
    ),
    "information_security": (
        "AN TOÀN THÔNG TIN", "AN TOAN THONG TIN", "BẢO MẬT", "BAO MAT",
        "SECURITY", "MẬT MÃ", "MAT MA", "CRYPTO",
    ),
    "iot": (
        "INTERNET OF THINGS", "IOT", "CẢM BIẾN", "CAM BIEN", "NHÚNG", "NHUNG",
        "EMBEDDED",
    ),
    "data_ai": (
        "DỮ LIỆU", "DU LIEU", "BIG DATA", "MÁY HỌC", "MAY HOC", "MACHINE LEARNING",
        "TRÍ TUỆ NHÂN TẠO", "TRI TUE NHAN TAO", "ARTIFICIAL INTELLIGENCE", "DEEP LEARNING",
    ),
}


def normalize_code(x) -> str:
    return str(x).strip().upper()


def normalize_name(x) -> str:
    return " ".join(str(x or "").strip().upper().split())


def infer_course_group(course_code: str, course_name: str) -> str:
    code = normalize_code(course_code)
    if code in EXACT_COURSE_GROUPS:
        return EXACT_COURSE_GROUPS[code]
    name = normalize_name(course_name)
    code_name = f"{code} {name}"
    for group, keywords in COURSE_GROUP_KEYWORDS.items():
        if any(k in code_name for k in keywords):
            return group
    return "general"


@dataclass
class Neighbor:
    student_id: str
    similarity: float
    common_courses: List[str]
    vector: Dict[str, float] = field(default_factory=dict)


@dataclass
class VectorRecommendation:
    course_id: str
    course_name: str
    group: str
    predicted_score: float
    recommendation_score: float
    rank: int
    supporting_neighbors: List[Neighbor]
    explanation: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class VectorSimilarityRecommender:
    """
    Example-based personalized learning-path recommender.

    Core idea:
      Target student vector = grades on courses already completed.
      Find students with the most similar vectors on the same course group.
      For each unlearned course, estimate the target student's expected score
      from the similar students' observed grades.

    This is a student-based collaborative filtering / kNN recommender.
    """

    def __init__(
        self,
        df_students: pd.DataFrame,
        courses: Iterable[dict],
        student_id_col_candidates: Tuple[str, ...] = ("MSSV", "mssv", "StudentID", "student_id"),
        top_neighbors: int = 10,
        min_common_courses: int = 3,
        same_major_only: bool = True,
    ):
        self.df = df_students.copy()
        self.courses = list(courses)
        self.top_neighbors = int(top_neighbors)
        self.min_common_courses = int(min_common_courses)
        self.student_id_col = self._find_student_id_col(student_id_col_candidates)
        self.same_major_only = same_major_only

        self.course_info: Dict[str, dict] = {}
        for c in self.courses:
            code = normalize_code(c.get("code", c.get("course_id", "")))
            if not code:
                continue
            name = str(c.get("name", c.get("course_name", code)))
            self.course_info[code] = {
                "code": code,
                "name": name,
                "credits": float(c.get("credits", 0) or 0),
                "required": bool(c.get("required", str(c.get("type", "")).lower() == "bắt buộc")),
                "prereq": [normalize_code(x) for x in c.get("prereq", c.get("prerequisites", []))],
                "group": c.get("group") or infer_course_group(code, name),
            }

        self.grade_cols: Dict[str, str] = {}
        for col in self.df.columns:
            s = str(col)
            if s.startswith("DIEM_"):
                code = normalize_code(s.replace("DIEM_", "", 1))
                self.grade_cols[code] = col

    def _find_student_id_col(self, candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        raise ValueError("Không tìm thấy cột MSSV/StudentID trong Wide Matrix.")

    def _major_value(self, row):
        for c in ("NganhHoc", "nganh_hoc", "Major", "major"):
            if c in row.index and pd.notna(row[c]):
                return str(row[c]).strip().upper()
        return ""

    def _batch_value(self, row):
        for c in ("MaLop", "ma_lop", "Batch", "batch"):
            if c in row.index and pd.notna(row[c]):
                return str(row[c]).strip().upper()
        return ""

    def _grade_vector(self, row: pd.Series, course_codes: Iterable[str]) -> Dict[str, float]:
        result = {}
        for code in course_codes:
            col = self.grade_cols.get(code)
            if not col or col not in row.index:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            s = str(value).strip().upper()
            if s in {"M", "MIEN", "MIỄN", "N/A", "NA", ""}:
                continue
            try:
                score = float(value)
            except (ValueError, TypeError):
                continue
            if 0.0 <= score <= 10.0:
                result[code] = score
        return result

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _similarity(self, target: Dict[str, float], other: Dict[str, float], group_codes: List[str]):
        common = [c for c in group_codes if c in target and c in other]
        if len(common) < self.min_common_courses:
            return 0.0, common
        a = np.asarray([target[c] for c in common], dtype=np.float32)
        b = np.asarray([other[c] for c in common], dtype=np.float32)
        cosine = self._cosine(a, b)
        pearson = self._pearson(a, b)
        pearson_norm = (pearson + 1.0) / 2.0

        # Absolute grade closeness is important for the use case in the thesis:
        # {8,9,9,10} should be closer to {7,9,8,10} than to {5,6,5.5,7}.
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        magnitude_similarity = math.exp(-rmse / 2.0)

        # Blend absolute closeness, profile shape and vector direction.
        similarity = (0.55 * magnitude_similarity
                      + 0.30 * pearson_norm
                      + 0.15 * max(0.0, cosine))
        # More shared courses provide more reliable evidence.
        similarity *= min(1.0, 0.50 + len(common) / 6.0)
        return float(np.clip(similarity, 0.0, 1.0)), common

    def _eligible_rows(self, target_row):
        target_major = self._major_value(target_row)
        rows = self.df
        if self.same_major_only and target_major:
            mask = rows.apply(lambda r: self._major_value(r) == target_major, axis=1)
            filtered = rows[mask]
            if len(filtered) >= max(20, self.top_neighbors * 2):
                rows = filtered
        return rows

    def recommend(
        self,
        student_id: str,
        top_k: int = 8,
        group: Optional[str] = None,
        min_predicted_score: float = 0.0,
    ) -> Tuple[List[VectorRecommendation], Dict[str, object]]:
        sid = str(student_id).strip()
        matches = self.df[self.df[self.student_id_col].astype(str).str.strip() == sid]
        if matches.empty:
            raise ValueError(f"Không tìm thấy sinh viên {sid} trong Wide Matrix.")
        target_row = matches.iloc[0]

        # Candidate groups are based on the course catalogue.
        all_codes = list(self.course_info.keys())
        target_vector_all = self._grade_vector(target_row, all_codes)
        passed_codes = set(target_vector_all.keys())

        if group is None:
            # Choose the group with the most completed graded courses.
            counts: Dict[str, int] = {}
            for code in passed_codes:
                g = self.course_info.get(code, {}).get("group", "general")
                counts[g] = counts.get(g, 0) + 1
            group = max(counts, key=counts.get) if counts else "general"

        group_codes = [c for c, info in self.course_info.items() if info.get("group") == group]
        target_vector = {c: target_vector_all[c] for c in group_codes if c in target_vector_all}

        if len(target_vector) < self.min_common_courses:
            # Fall back to all known graded courses while keeping candidate
            # recommendations restricted to the selected group.
            group_codes_for_similarity = [c for c in all_codes if c in target_vector_all]
            target_vector_similarity = target_vector_all
        else:
            group_codes_for_similarity = group_codes
            target_vector_similarity = target_vector

        eligible_rows = self._eligible_rows(target_row)
        neighbors: List[Neighbor] = []
        for _, row in eligible_rows.iterrows():
            other_id = str(row[self.student_id_col]).strip()
            if other_id == sid:
                continue
            other_vector = self._grade_vector(row, group_codes_for_similarity)
            sim, common = self._similarity(target_vector_similarity, other_vector, group_codes_for_similarity)
            if sim <= 0:
                continue
            neighbors.append(Neighbor(other_id, sim, common, other_vector))

        neighbors.sort(key=lambda n: n.similarity, reverse=True)
        neighbors = neighbors[: self.top_neighbors]

        # Courses not yet graded by target, within the selected group.
        candidates = []
        for c in group_codes:
            if c not in self.course_info or c in passed_codes:
                continue
            prereq = set(self.course_info[c].get("prereq", []))
            # A recommendation may not violate curriculum prerequisites.
            if not prereq.issubset(passed_codes):
                continue
            candidates.append(c)

        recs: List[VectorRecommendation] = []
        for code in candidates:
            contributors = [n for n in neighbors if code in self._grade_vector_by_id(n.student_id, [code])]
            if not contributors:
                continue
            weights = np.asarray([n.similarity for n in contributors], dtype=np.float64)
            scores = np.asarray([self._grade_vector_by_id(n.student_id, [code])[code] for n in contributors], dtype=np.float64)
            if weights.sum() <= 0:
                continue
            pred = float(np.dot(weights, scores) / weights.sum())
            if pred < min_predicted_score:
                continue
            # Recommendation score combines predicted grade and evidence.
            evidence = min(1.0, len(contributors) / max(3, self.top_neighbors))
            rec_score = 0.80 * (pred / 10.0) + 0.20 * evidence
            recs.append(VectorRecommendation(
                course_id=code,
                course_name=self.course_info[code]["name"],
                group=group,
                predicted_score=pred,
                recommendation_score=rec_score,
                rank=0,
                supporting_neighbors=contributors,
                reasons=[],
                warnings=[],
            ))

        recs.sort(key=lambda r: (r.recommendation_score, r.predicted_score), reverse=True)
        recs = recs[:top_k]
        for i, r in enumerate(recs, 1):
            r.rank = i
            r.reasons = [
                f"Vector của bạn tương đồng cao nhất với {len(r.supporting_neighbors)} sinh viên có dữ liệu ở học phần này.",
                f"Điểm dự đoán từ nhóm sinh viên tương đồng: {r.predicted_score:.2f}/10.",
                f"Hệ thống ưu tiên các học phần chưa học thuộc nhóm '{group}'.",
            ]
            if r.supporting_neighbors:
                best = r.supporting_neighbors[0]
                r.reasons.append(
                    f"Sinh viên tương đồng nhất: {best.student_id} (similarity={best.similarity:.3f})."
                )

        diagnostics = {
            "student_id": sid,
            "group": group,
            "target_vector": target_vector,
            "similarity_courses": group_codes_for_similarity,
            "num_target_group_courses": len(target_vector),
            "neighbors": neighbors,
            "candidate_count": len(candidates),
        }
        return recs, diagnostics

    def _grade_vector_by_id(self, student_id: str, course_codes: Iterable[str]) -> Dict[str, float]:
        rows = self.df[self.df[self.student_id_col].astype(str).str.strip() == str(student_id).strip()]
        if rows.empty:
            return {}
        return self._grade_vector(rows.iloc[0], course_codes)
