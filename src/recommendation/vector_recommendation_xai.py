from __future__ import annotations

from typing import Dict, List
import numpy as np

try:
    import shap
except ImportError:
    shap = None


class VectorRecommendationXAI:
    """Explain example-based recommendations.

    Primary explanation is example-based: which similar students supported the
    predicted grade. Optional SHAP explains how each anchor-course grade in
    the target vector changes the local predicted score.
    """

    def explain_neighbors(self, recommendation):
        rows = []
        for n in recommendation.supporting_neighbors[:8]:
            candidate_grade = n.vector.get(recommendation.course_id)
            if candidate_grade is None:
                continue
            rows.append({
                "student_id": n.student_id,
                "similarity": float(n.similarity),
                "predicted_course_grade": float(candidate_grade),
                "common_courses": len(n.common_courses),
            })
        return rows

    def explain_vector(self, recommender, diagnostics, recommendation, max_features=8):
        target_vector = diagnostics.get("target_vector", {})
        feature_names = list(target_vector.keys())
        if not feature_names:
            return {}

        # If SHAP is unavailable, use a transparent leave-one-feature-out proxy.
        if shap is None:
            return self._leave_one_out(recommender, diagnostics, recommendation, feature_names, max_features)

        baseline = np.asarray([target_vector[c] for c in feature_names], dtype=np.float32)
        bg = np.tile(baseline, (1, 1))

        def predict(X):
            out = []
            for row in np.asarray(X):
                sim_scores = []
                target = {c: float(v) for c, v in zip(feature_names, row)}
                for n in diagnostics.get("neighbors", []):
                    common = [c for c in feature_names if c in n.vector]
                    if len(common) < recommender.min_common_courses:
                        continue
                    a = np.asarray([target[c] for c in common], dtype=np.float32)
                    b = np.asarray([n.vector[c] for c in common], dtype=np.float32)
                    sim, _ = recommender._similarity(target, n.vector, feature_names)
                    grade = n.vector.get(recommendation.course_id)
                    if grade is not None:
                        sim_scores.append((sim, float(grade)))
                if sim_scores:
                    w = np.asarray([x[0] for x in sim_scores])
                    y = np.asarray([x[1] for x in sim_scores])
                    out.append(float(np.dot(w, y) / max(w.sum(), 1e-8)))
                else:
                    out.append(0.0)
            return np.asarray(out, dtype=float)

        try:
            explainer = shap.KernelExplainer(predict, bg)
            values = explainer.shap_values(baseline.reshape(1, -1), nsamples=min(100, 2 * len(feature_names) + 20))
            if isinstance(values, list):
                values = values[0]
            values = np.asarray(values)[0]
            pairs = [(name, float(value)) for name, value in zip(feature_names, values)]
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return dict(pairs[:max_features])
        except Exception:
            return self._leave_one_out(recommender, diagnostics, recommendation, feature_names, max_features)

    def _leave_one_out(self, recommender, diagnostics, recommendation, feature_names, max_features):
        target = diagnostics.get("target_vector", {})
        full = recommendation.predicted_score
        effects = []
        for feature in feature_names:
            reduced = {k: v for k, v in target.items() if k != feature}
            if len(reduced) < recommender.min_common_courses:
                continue
            vals = []
            weights = []
            for n in diagnostics.get("neighbors", []):
                common = [c for c in reduced if c in n.vector]
                if len(common) < recommender.min_common_courses or recommendation.course_id not in n.vector:
                    continue
                sim, _ = recommender._similarity(reduced, n.vector, list(reduced.keys()))
                if sim > 0:
                    vals.append(n.vector[recommendation.course_id])
                    weights.append(sim)
            reduced_pred = float(np.dot(weights, vals) / sum(weights)) if weights else full
            effects.append((feature, full - reduced_pred))
        effects.sort(key=lambda x: abs(x[1]), reverse=True)
        return dict(effects[:max_features])
