"""
By Drashti
inference_engine.py

Main inference engine for the Skill Matching System.

Responsibilities:
1. Convert employee data to text
2. Convert project/task data to text
3. Generate embeddings
4. Calculate similarity
5. Return Skill Match Score
"""

from backend.BAAI.fine_tuning.preprocessing.text_builder import TextBuilder

from backend.BAAI.fine_tuning.flow.inference.embedding_generator import (
    EmbeddingGenerator,
)

from backend.BAAI.fine_tuning.flow.inference.similarity import (
    SimilarityCalculator,
)


class InferenceEngine:

    @staticmethod
    def calculate_skill_match(employee: dict, project: dict) -> dict:
        """
        Calculate Skill Match Score between one employee
        and one project.
        """

        # --------------------------------------------------
        # Build Text
        # --------------------------------------------------

        employee_text = TextBuilder.build_employee_text(employee)

        project_text = TextBuilder.build_project_text(project)

        # --------------------------------------------------
        # Generate Embeddings
        # --------------------------------------------------

        employee_embedding = EmbeddingGenerator.generate(employee_text)

        project_embedding = EmbeddingGenerator.generate(project_text)

        # --------------------------------------------------
        # Similarity
        # --------------------------------------------------

        similarity = SimilarityCalculator.cosine_similarity(
            employee_embedding,
            project_embedding
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        return {

            "employee_id": employee.get("employee_id"),

            "project_id": project.get("project_id"),

            "skill_match_score": round(similarity, 4)

        }
