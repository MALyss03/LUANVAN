# Personalized Learning Path Recommendation + XAI

## Pipeline

Raw student history
→ semester features
→ risk prediction
→ candidate generation by curriculum/prerequisites
→ Transformer sequential ranking
→ interpretable reranking
→ SHAP explanation
→ personalized learning path

## Important methodological boundary

SHAP in `src/xai/recommendation_explainer.py` explains the interpretable feature/rule component that contributes to the final ranking.

It does not claim that SHAP directly explains every attention weight inside the Transformer.

For the thesis, report these as two complementary explanation levels:

1. **Global model explanation**: SHAP for student-risk/feature-based scoring.
2. **Recommendation explanation**: SHAP + deterministic curriculum reasons for each recommended course.

## Output

Each recommendation contains:

- rank
- course
- score
- reasons
- warnings
- XAI feature contributions

Example:

`Credit deficit +0.21` means the feature increases the recommendation score under the explainer.
