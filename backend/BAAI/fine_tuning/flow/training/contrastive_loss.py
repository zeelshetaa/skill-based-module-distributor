"""
contrastive_loss.py

Margin-based contrastive loss (Hadsell et al., 2006) adapted for
cosine-distance embeddings, used to fine-tune BAAI/bge-base-en-v1.5
as a bi-encoder on (employee, project, label) pairs.

label == 1 (skill match)      -> pull embeddings together
label == 0 (no skill match)   -> push embeddings apart, past `margin`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Standard margin-based triplet loss (Schroff et al., 2015 / FaceNet),
    adapted to cosine distance for the (anchor, positive, negative)
    triplets produced by pair_builder.py.

    distance(a, b) = 1 - cosine_similarity(a, b)   # in [0, 2]

    loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)

    Intuition: the anchor should be closer to its positive than to its
    negative by at least `margin`. If it already is (loss term goes
    negative), relu clips it to 0 -- no gradient, that triplet is
    already "easy" and doesn't need further pushing.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be > 0")
        self.margin = margin

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=-1)
        neg_sim = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=-1)

        pos_distance = 1.0 - pos_sim
        neg_distance = 1.0 - neg_sim

        loss = F.relu(pos_distance - neg_distance + self.margin)

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    distance = 1 - cosine_similarity(employee_emb, project_emb)   # in [0, 2]

    loss = label       * distance^2
         + (1 - label) * max(0, margin - distance)^2

    A margin of 0.5 means: negative pairs aren't penalized once they're
    already at least 0.5 cosine-distance apart (~cosine similarity <= 0.5).
    Positive pairs are always pulled to distance 0 (similarity 1.0).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be > 0")
        self.margin = margin

    def forward(
        self,
        employee_embeddings: torch.Tensor,
        project_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # Embeddings are expected to already be L2-normalized (see trainer.py's
        # mean_pooling/cls_pooling + F.normalize step), so cosine similarity
        # reduces to a dot product. We still call cosine_similarity directly
        # so this loss is safe to reuse even if normalization is skipped upstream.
        cosine_sim = F.cosine_similarity(employee_embeddings, project_embeddings, dim=-1)
        distance = 1.0 - cosine_sim

        labels = labels.float()

        positive_term = labels * distance.pow(2)
        negative_term = (1.0 - labels) * F.relu(self.margin - distance).pow(2)

        loss = positive_term + negative_term

        return loss.mean()


if __name__ == "__main__":
    # Quick sanity check with synthetic tensors -- no model/tokenizer needed.
    torch.manual_seed(0)

    # --- TripletLoss checks (this is what trainer.py now uses) ---
    triplet_loss_fn = TripletLoss(margin=0.3)

    anchor = F.normalize(torch.randn(8, 16), dim=-1)
    positive = F.normalize(torch.randn(8, 16), dim=-1)
    negative = F.normalize(torch.randn(8, 16), dim=-1)

    loss = triplet_loss_fn(anchor, positive, negative)
    print("Sample triplet loss (random embeddings):", loss.item())

    # anchor == positive, anchor far from negative -> loss should be ~0 (already correctly separated)
    identical_pos = anchor.clone()
    far_negative = -anchor.clone()
    print(
        "anchor==positive, negative=opposite (should be ~0):",
        triplet_loss_fn(anchor, identical_pos, far_negative).item(),
    )

    # anchor == negative (worst case) -> loss should be near margin (maximally wrong)
    print(
        "anchor==negative, positive=opposite (worst case, should be large ~2+margin):",
        triplet_loss_fn(anchor, far_negative, anchor.clone()).item(),
    )

    print()

    # --- Legacy ContrastiveLoss checks (kept for reference / pairwise use elsewhere) ---
    emp = F.normalize(torch.randn(8, 16), dim=-1)
    proj = F.normalize(torch.randn(8, 16), dim=-1)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float)

    loss_fn = ContrastiveLoss(margin=0.5)
    loss = loss_fn(emp, proj, labels)
    print("Sample contrastive loss:", loss.item())

    # loss should drop to ~0 for identical embeddings on positive-only labels
    identical = F.normalize(torch.randn(4, 16), dim=-1)
    all_positive = torch.ones(4)
    print("Identical + all-positive loss (should be ~0):", loss_fn(identical, identical, all_positive).item())