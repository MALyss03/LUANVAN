from __future__ import annotations

from typing import Optional
from .vector_similarity_recommender import VectorSimilarityRecommender
from .vector_recommendation_xai import VectorRecommendationXAI


class VectorRecommendationService:
    def __init__(self, df_students, courses, top_neighbors=10, min_common_courses=3):
        self.recommender = VectorSimilarityRecommender(
            df_students=df_students,
            courses=courses,
            top_neighbors=top_neighbors,
            min_common_courses=min_common_courses,
        )
        self.xai = VectorRecommendationXAI()

    def recommend(self, student_id: str, top_k=8, group: Optional[str] = None):
        recs, diagnostics = self.recommender.recommend(student_id, top_k=top_k, group=group)
        for rec in recs:
            rec.explanation = self.xai.explain_vector(self.recommender, diagnostics, rec)
        return recs, diagnostics
