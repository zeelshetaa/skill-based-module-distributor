"""
similarity.py

Calculate cosine similarity between two embeddings.
"""

import torch
import torch.nn.functional as F


class SimilarityCalculator:

    @staticmethod
    def cosine_similarity(
        employee_embedding: torch.Tensor,
        task_embedding: torch.Tensor
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Returns:
            float: similarity score between -1 and 1
        """

        similarity = F.cosine_similarity(
            employee_embedding.unsqueeze(0),
            task_embedding.unsqueeze(0),
            dim=1
        )

        return similarity.item()