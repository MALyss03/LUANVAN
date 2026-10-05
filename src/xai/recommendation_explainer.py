
from __future__ import annotations
from typing import Dict, List, Callable, Optional
import numpy as np

try:
    import shap
except ImportError:
    shap = None

from src.recommendation.schemas import Course, StudentState
from src.recommendation.feature_builder import FEATURE_NAMES, build_course_features


class RecommendationXAI:
    """SHAP explanation for the interpretable recommendation layer.

    Important methodological distinction:
    SHAP here explains the final interpretable/rule-based component that
    influences ranking. It does NOT claim that SHAP directly explains all
    internal attention weights of the Transformer.
    """

    def __init__(self, score_function: Optional[Callable[[np.ndarray], float]] = None):
        self.score_function = score_function

    def _default_score(self, x: np.ndarray) -> float:
        # x follows FEATURE_NAMES.
        credit_deficit = x[0]
        gpa = x[2]
        fail_rate = x[6]
        risk = x[7]
        is_failed = x[8]
        is_required = x[9]

        value = (
            0.25 * np.tanh(credit_deficit / 20.0)
            + 0.20 * np.clip((3.0 - gpa) / 3.0, 0, 1)
            + 0.20 * fail_rate
            + 0.20 * risk
            + 0.10 * is_failed
            + 0.05 * is_required
        )
        return float(value)

    def explain(
        self,
        student: StudentState,
        course: Course,
        background: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        x = build_course_features(student, course).reshape(1, -1)
        fn = self.score_function or self._default_score

        if shap is None:
            # Graceful fallback: transparent feature contributions.
            values = x[0]
            return {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, values)
            }

        if background is None:
            background = np.zeros((1, x.shape[1]), dtype=np.float32)

        def predict(X):
            return np.asarray([fn(row) for row in X], dtype=float)

        explainer = shap.KernelExplainer(predict, background)
        shap_values = explainer.shap_values(x, nsamples="auto")

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[0])[0]
        else:
            values = np.asarray(shap_values)[0]

        return {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, values)
        }

    @staticmethod
    def top_reasons(xai_values: Dict[str, float], top_n: int = 5) -> List[tuple[str, float]]:
        return sorted(
            xai_values.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:top_n]
