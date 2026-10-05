
from __future__ import annotations
import numpy as np
from .schemas import Course, StudentState

FEATURE_NAMES = [
    "credit_deficit",
    "remaining_credits",
    "cumulative_gpa",
    "latest_semester_gpa",
    "gpa_trend",
    "credit_velocity",
    "fail_rate",
    "risk_probability",
    "is_failed_course",
    "is_required_course",
    "prerequisite_count",
    "current_semester",
]


def build_course_features(student: StudentState, course: Course) -> np.ndarray:
    return np.asarray([
        float(student.credit_deficit),
        float(student.remaining_credits),
        float(student.cumulative_gpa),
        float(student.latest_semester_gpa),
        float(student.gpa_trend),
        float(student.credit_velocity),
        float(student.fail_rate),
        float(student.risk_probability),
        float(course.course_id in set(student.failed_courses)),
        float(course.required),
        float(len(course.prerequisites)),
        float(student.current_semester),
    ], dtype=np.float32)
