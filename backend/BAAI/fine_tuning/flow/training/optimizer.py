"""
optimizer.py

Builds the AdamW optimizer (with the standard "no weight decay on
bias/LayerNorm" parameter grouping) plus a linear warmup + linear
decay learning-rate scheduler, for fine-tuning BAAI/bge-base-en-v1.5.
"""

from typing import Tuple
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

# Parameters whose names contain these substrings skip weight decay --
# this is the standard BERT/transformer fine-tuning convention, since
# decaying bias terms and LayerNorm scale/shift hurts convergence.
NO_DECAY_KEYWORDS = ("bias", "LayerNorm.weight", "layer_norm.weight")


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> AdamW:
    """
    Groups model parameters into "decay" and "no_decay" buckets and
    returns an AdamW optimizer over both groups.
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in NO_DECAY_KEYWORDS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=learning_rate)


def build_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup for `warmup_ratio` of total steps, then linear decay
    to 0 for the remainder. `num_training_steps` should be
    len(train_dataloader) * num_epochs (i.e. counted in optimizer
    steps, not epochs).
    """

    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    num_warmup_steps = int(num_training_steps * warmup_ratio)

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def build_optimizer_and_scheduler(
    model: nn.Module,
    num_training_steps: int,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> Tuple[AdamW, LambdaLR]:
    """Convenience wrapper used directly by trainer.py / train.py."""

    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, num_training_steps=num_training_steps, warmup_ratio=warmup_ratio)
    return optimizer, scheduler


if __name__ == "__main__":
    # Sanity check with a tiny dummy model -- no BGE download needed.
    import torch.nn as nn

    dummy_model = nn.Sequential(
        nn.Linear(16, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 8),
    )

    optimizer, scheduler = build_optimizer_and_scheduler(dummy_model, num_training_steps=100)

    decay_count = len(optimizer.param_groups[0]["params"])
    no_decay_count = len(optimizer.param_groups[1]["params"])
    print(f"decay params: {decay_count}, no-decay params: {no_decay_count}")
    print("initial LR:", scheduler.get_last_lr())