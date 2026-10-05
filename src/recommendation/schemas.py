
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Course:
    course_id: str
    course_name: str
    credits: float
    prerequisites: List[str] = field(default_factory=list)
    semester_hint: Optional[int] = None
    required: bool = True
    category: str = "general"


@dataclass
class StudentState:
    student_id: str
    current_semester: int
    passed_courses: List[str]
    failed_courses: List[str]
    current_courses: List[str]
    remaining_credits: float
    credit_deficit: float
    cumulative_gpa: float
    latest_semester_gpa: float
    gpa_trend: float
    credit_velocity: float
    fail_rate: float
    prerequisite_block_count: int = 0
    risk_probability: float = 0.0
    history: List[str] = field(default_factory=list)


@dataclass
class Recommendation:
    course_id: str
    course_name: str
    score: float
    rank: int
    reasons: List[str]
    warnings: List[str]
    xai: Dict[str, float] = field(default_factory=dict)
