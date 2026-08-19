"""
score_combiner.py

Combines BGE similarity score with existing business scores
to produce one overall ranking score.
"""


class ScoreCombiner:

    # Change these anytime
    SKILL_WEIGHT = 0.40
    TECHNICAL_WEIGHT = 0.20
    EXECUTION_WEIGHT = 0.10
    LEARNING_WEIGHT = 0.10
    ADAPTABILITY_WEIGHT = 0.10
    WORKLOAD_WEIGHT = 0.10

    @classmethod
    def calculate(
        cls,
        skill_match_score: float,
        technical_score: float,
        execution_score: float,
        learning_score: float,
        adaptability_score: float,
        workload_score: float
    ) -> float:

        # Normalize 0-100 scores to 0-1
        technical = technical_score / 100
        execution = execution_score / 100
        learning = learning_score / 100
        adaptability = adaptability_score / 100

        # Lower workload is better
        workload = 1 - (workload_score / 100)

        overall_score = (

            cls.SKILL_WEIGHT * skill_match_score +

            cls.TECHNICAL_WEIGHT * technical +

            cls.EXECUTION_WEIGHT * execution +

            cls.LEARNING_WEIGHT * learning +

            cls.ADAPTABILITY_WEIGHT * adaptability +

            cls.WORKLOAD_WEIGHT * workload

        )

        return round(overall_score, 4)