"""
evaluator.py

Metrics for triplet-trained embeddings. Unlike labeled-pair
classification, there's no threshold to sweep here -- the question a
triplet model needs to answer is simpler and more direct:

    "Is the anchor closer to its positive than to its negative?"

triplet_accuracy = fraction of triplets where that's true.
mean_margin = average (pos_similarity - neg_similarity) -- how
confidently separated positives are from negatives, not just whether
they're on the right side of 0.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fine_tuning.flow.training.trainer import encode


@dataclass
class EvalResult:
    triplet_accuracy: float
    mean_margin: float
    mean_positive_similarity: float
    mean_negative_similarity: float


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
    pooling: str = "cls",
) -> EvalResult:
    """
    Encodes every triplet in data_loader and reports how well the
    model separates positives from negatives. This is the direct
    triplet-loss analogue of classification accuracy -- run this on
    test_pairs.csv after training to see if fine-tuning actually
    improved skill matching over the base model.
    """

    model.eval()

    all_pos_sim = []
    all_neg_sim = []

    for batch in data_loader:
        anchor_emb = encode(
            model,
            batch["anchor_input_ids"].to(device),
            batch["anchor_attention_mask"].to(device),
            pooling=pooling,
        )
        positive_emb = encode(
            model,
            batch["positive_input_ids"].to(device),
            batch["positive_attention_mask"].to(device),
            pooling=pooling,
        )
        negative_emb = encode(
            model,
            batch["negative_input_ids"].to(device),
            batch["negative_attention_mask"].to(device),
            pooling=pooling,
        )

        pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=-1)

        all_pos_sim.append(pos_sim.cpu())
        all_neg_sim.append(neg_sim.cpu())

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    correct = (pos_sim > neg_sim).float()
    triplet_accuracy = correct.mean().item()

    margins = pos_sim - neg_sim
    mean_margin = margins.mean().item()

    return EvalResult(
        triplet_accuracy=triplet_accuracy,
        mean_margin=mean_margin,
        mean_positive_similarity=pos_sim.mean().item(),
        mean_negative_similarity=neg_sim.mean().item(),
    )


def print_eval_result(result: EvalResult, split_name: str = "test") -> None:
    print(f"--- {split_name} metrics ---")
    print(f"triplet accuracy (pos closer than neg): {result.triplet_accuracy:.4f}")
    print(f"mean margin (pos_sim - neg_sim):         {result.mean_margin:.4f}")
    print(f"mean cosine sim (positive):              {result.mean_positive_similarity:.4f}")
    print(f"mean cosine sim (negative):               {result.mean_negative_similarity:.4f}")