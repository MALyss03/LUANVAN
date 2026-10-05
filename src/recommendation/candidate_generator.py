
from __future__ import annotations
from typing import Iterable, List, Set
from .schemas import Course, StudentState


class CandidateGenerator:
    """Generate feasible courses before ML ranking.

    The recommender must never recommend a course whose prerequisites are
    not satisfied. This constraint layer is intentionally deterministic.
    """

    def __init__(self, courses: Iterable[Course]):
        self.courses = list(courses)

    def _prerequisites_satisfied(self, course: Course, passed: Set[str]) -> bool:
        return set(course.prerequisites).issubset(passed)

    def generate(self, student: StudentState, limit: int | None = None) -> List[Course]:
        passed = set(student.passed_courses)
        current = set(student.current_courses)

        candidates = []
        for course in self.courses:
            if course.course_id in passed or course.course_id in current:
                continue
            if not self._prerequisites_satisfied(course, passed):
                continue
            candidates.append(course)

        # Failed courses are placed first because retaking them can remove
        # prerequisite/credit deficits.
        failed = set(student.failed_courses)
        candidates.sort(key=lambda c: (c.course_id not in failed, not c.required, c.course_id))

        return candidates[:limit] if limit else candidates
