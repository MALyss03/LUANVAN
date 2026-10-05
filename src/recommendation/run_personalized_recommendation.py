
from __future__ import annotations
import json
from pathlib import Path

from .schemas import Course, StudentState
from .personalized_pathway import PersonalizedPathway
from src.xai.recommendation_explainer import RecommendationXAI


def explain_recommendations(student: StudentState, recommendations, courses_by_id):
    explainer = RecommendationXAI()

    for rec in recommendations:
        course = courses_by_id[rec.course_id]
        values = explainer.explain(student, course)
        rec.xai = values

        top = explainer.top_reasons(values, top_n=3)
        rec.reasons.extend(
            f"XAI: {name} ({value:+.3f})"
            for name, value in top
            if abs(value) > 0
        )

    return recommendations


if __name__ == "__main__":
    # Replace this example with your curriculum/database loader.
    courses = [
        Course("CT001", "Mạng máy tính", 3, [], required=True, category="core"),
        Course("CT002", "An toàn thông tin", 3, ["CT001"], required=True, category="core"),
        Course("CT003", "Internet of Things", 3, [], required=False, category="elective"),
    ]

    student = StudentState(
        student_id="DEMO001",
        current_semester=6,
        passed_courses=["CT001"],
        failed_courses=[],
        current_courses=[],
        remaining_credits=29,
        credit_deficit=10,
        cumulative_gpa=2.55,
        latest_semester_gpa=2.40,
        gpa_trend=-0.08,
        credit_velocity=14.0,
        fail_rate=0.08,
        history=["CT001"],
        risk_probability=0.62,
    )

    courses_by_id = {c.course_id: c for c in courses}
    engine = PersonalizedPathway(courses)
    recommendations = engine.recommend(student, top_k=3)
    recommendations = explain_recommendations(student, recommendations, courses_by_id)

    print(json.dumps([
        {
            "rank": r.rank,
            "course_id": r.course_id,
            "course_name": r.course_name,
            "score": r.score,
            "reasons": r.reasons,
            "warnings": r.warnings,
            "xai": r.xai,
        }
        for r in recommendations
    ], ensure_ascii=False, indent=2))
