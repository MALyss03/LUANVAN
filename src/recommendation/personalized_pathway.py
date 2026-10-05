
from __future__ import annotations
from typing import Iterable, List, Dict, Optional
import numpy as np

from .schemas import Course, StudentState, Recommendation
from .candidate_generator import CandidateGenerator
from .feature_builder import build_course_features, FEATURE_NAMES
from .transformer_recommender import TransformerRanker


class PersonalizedPathway:
    """Constraint-aware + Transformer + XAI-ready recommendation pipeline.

    Final score:
        transformer_score * transformer_weight
        + interpretable_score * feature_weight

    The deterministic rules are applied before ranking, so infeasible
    courses cannot enter the recommendation list.
    """

    def __init__(
        self,
        courses: Iterable[Course],
        ranker: Optional[TransformerRanker] = None,
        transformer_weight: float = 0.70,
        feature_weight: float = 0.30,
    ):
        self.generator = CandidateGenerator(courses)
        self.ranker = ranker
        self.transformer_weight = transformer_weight
        self.feature_weight = feature_weight

    def _interpretable_score(self, student: StudentState, course: Course) -> float:
        score = 0.0

        # Priority for graduation-critical courses.
        if course.required:
            score += 0.35

        # Retaking a failed course is usually highly relevant.
        if course.course_id in set(student.failed_courses):
            score += 0.30

        # Credit deficit makes required courses more valuable.
        if student.credit_deficit > 0 and course.required:
            score += 0.15

        # A high risk student should receive a conservative workload.
        if student.risk_probability >= 0.70:
            score += 0.05

        return min(score, 1.0)

    def _reasons(self, student: StudentState, course: Course) -> tuple[list[str], list[str]]:
        reasons, warnings = [], []

        if course.required:
            reasons.append("Học phần bắt buộc/quan trọng đối với hoàn thành chương trình.")
        if course.course_id in set(student.failed_courses):
            reasons.append("Học phần đã từng chưa đạt, ưu tiên học lại để giảm tín chỉ nợ.")
        if student.credit_deficit > 0 and course.required:
            reasons.append("Giúp giảm khoảng thiếu tín chỉ so với tiến độ mục tiêu.")

        if student.risk_probability >= 0.70:
            warnings.append("Sinh viên đang có xác suất rủi ro cao; cần kiểm soát tải học tập.")
        if student.fail_rate > 0.20:
            warnings.append("Tỷ lệ học phần không đạt cao; nên ưu tiên học phần tiên quyết và bắt buộc.")

        return reasons, warnings

    def recommend(
        self,
        student: StudentState,
        top_k: int = 5,
        candidate_limit: int = 50,
    ) -> List[Recommendation]:
        candidates = self.generator.generate(student, limit=candidate_limit)
        if not candidates:
            return []

        ids = [c.course_id for c in candidates]
        transformer_scores = (
            self.ranker.score_candidates(student.history, ids)
            if self.ranker else {cid: 0.0 for cid in ids}
        )

        scored = []
        for course in candidates:
            rule_score = self._interpretable_score(student, course)
            t_score = transformer_scores.get(course.course_id, 0.0)

            if self.ranker:
                final = (
                    self.transformer_weight * t_score
                    + self.feature_weight * rule_score
                )
            else:
                final = rule_score

            reasons, warnings = self._reasons(student, course)
            scored.append((final, course, reasons, warnings))

        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for rank, (score, course, reasons, warnings) in enumerate(scored[:top_k], 1):
            result.append(
                Recommendation(
                    course_id=course.course_id,
                    course_name=course.course_name,
                    score=float(score),
                    rank=rank,
                    reasons=reasons,
                    warnings=warnings,
                    xai={},
                )
            )
        return result
