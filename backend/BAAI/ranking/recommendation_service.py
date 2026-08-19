"""
recommendation_service.py

Returns the best employee recommendations.
"""

from backend.BAAI.ranking.ranking_engine import RankingEngine


class RecommendationService:

    @staticmethod
    def recommend(task_id, top_k=5):

        ranking = RankingEngine.rank(task_id)

        return ranking[:top_k]