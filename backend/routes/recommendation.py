from fastapi import APIRouter

from backend.BAAI.ranking.recommendation_service import (
    RecommendationService,
)

router = APIRouter()


@router.get("/recommend/{task_id}")
def recommend(task_id: int):

    result = RecommendationService.recommend(task_id)

    return result
