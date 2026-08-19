"""
ranking_engine.py

Main Ranking Engine.

Workflow

1. Fetch task from Supabase
2. Fetch all employees
3. Generate Skill Match Score using fine-tuned BAAI model
4. Combine with existing employee scores
5. Rank employees
6. Return sorted recommendation list
"""

from backend.services.ranking_repository import get_all_employees
from backend.services.task_repository import get_task

from backend.BAAI.fine_tuning.flow.inference.inference_engine import (
    InferenceEngine,
)

from backend.BAAI.ranking.score_combiner import ScoreCombiner


class RankingEngine:

    @staticmethod
    def rank(task_id):

        # ----------------------------------------
        # Fetch Task
        # ----------------------------------------

        task = get_task(task_id)

        if task is None:
            raise Exception("Task not found.")

        # ----------------------------------------
        # Fetch Employees
        # ----------------------------------------

        employees = get_all_employees()

        ranking = []

        # ----------------------------------------
        # Process Every Employee
        # ----------------------------------------

        for employee in employees:

            # -----------------------------
            # BAAI Skill Match
            # -----------------------------

            result = InferenceEngine.calculate_skill_match(
                employee,
                task
            )

            skill_match_score = result["skill_match_score"]

            # -----------------------------
            # Existing Scores
            # -----------------------------

            technical_score = employee.get(
                "technical_score",
                0
            )

            execution_score = employee.get(
                "execution_score",
                0
            )

            learning_score = employee.get(
                "learning_score",
                0
            )

            adaptability_score = employee.get(
                "adaptability_score",
                0
            )

            workload_score = employee.get(
                "workload_score",
                0
            )

            active_tasks = employee.get(
                "active_tasks",
                0
            )

            available_hours = employee.get(
                "available_hours",
                0
            )

            # -----------------------------
            # Final Score
            # -----------------------------

            final_score = ScoreCombiner.combine(

                skill_match_score=skill_match_score,

                technical_score=technical_score,

                execution_score=execution_score,

                learning_score=learning_score,

                adaptability_score=adaptability_score,

                workload_score=workload_score,

                active_tasks=active_tasks,

                available_hours=available_hours

            )

            ranking.append({

                "emp_id":
                    employee["emp_id"],

                "name":
                    employee["name"],

                "role":
                    employee["role"],

                "skill_match_score":
                    skill_match_score,

                "technical_score":
                    technical_score,

                "execution_score":
                    execution_score,

                "learning_score":
                    learning_score,

                "adaptability_score":
                    adaptability_score,

                "workload_score":
                    workload_score,

                "active_tasks":
                    active_tasks,

                "available_hours":
                    available_hours,

                "final_score":
                    final_score

            })

        # ----------------------------------------
        # Sort Highest First
        # ----------------------------------------

        ranking.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return ranking