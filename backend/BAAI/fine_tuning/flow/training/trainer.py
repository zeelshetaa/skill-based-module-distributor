"""
trainer.py

Core training loop for fine-tuning BAAI/bge-base-en-v1.5 as a bi-encoder
on (employee_text, project_text, label) pairs, using ContrastiveLoss
and the AdamW optimizer/scheduler from optimizer.py.
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fine_tuning.flow.training.contrastive_loss import TripletLoss


def encode(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "cls",
) -> torch.Tensor:
    """
    Runs the encoder and pools token embeddings into one sentence
    embedding per input, then L2-normalizes it.

    BGE models are trained with CLS-token pooling (the official model
    card recommends using the first token's hidden state, not mean
    pooling), so "cls" is the default. "mean" is provided as a fallback
    in case you swap in a different base encoder later.
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)

    if pooling == "cls":
        pooled = last_hidden_state[:, 0]  # CLS token
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")

    return F.normalize(pooled, p=2, dim=-1)


class Trainer:
    """
    Wraps the training loop: forward pass on both towers, contrastive
    loss, backward, optimizer/scheduler step, periodic validation via
    an injected `evaluate_fn`, and best-checkpoint saving.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        device: Optional[torch.device] = None,
        pooling: str = "cls",
        margin: float = 0.3,
        grad_clip_norm: float = 1.0,
        output_dir: str = "fine_tuning/flow/saved_model/bge_finetuned",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.loss_fn = TripletLoss(margin=margin)
        self.grad_clip_norm = grad_clip_norm
        self.output_dir = output_dir

        self.model.to(self.device)
        self.best_val_loss = float("inf")

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        anchor_emb = encode(
            self.model,
            batch["anchor_input_ids"].to(self.device),
            batch["anchor_attention_mask"].to(self.device),
            pooling=self.pooling,
        )
        positive_emb = encode(
            self.model,
            batch["positive_input_ids"].to(self.device),
            batch["positive_attention_mask"].to(self.device),
            pooling=self.pooling,
        )
        negative_emb = encode(
            self.model,
            batch["negative_input_ids"].to(self.device),
            batch["negative_attention_mask"].to(self.device),
            pooling=self.pooling,
        )

        return self.loss_fn(anchor_emb, positive_emb, negative_emb)

    def train_one_epoch(self, train_loader: DataLoader, epoch: int, log_every: int = 20) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            self.optimizer.zero_grad()

            loss = self._forward_batch(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if step % log_every == 0 or step == num_batches:
                elapsed = time.time() - start
                print(
                    f"[epoch {epoch}] step {step}/{num_batches} "
                    f"loss={loss.item():.4f} avg_loss={total_loss / step:.4f} "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s"
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(valid_loader)

        for batch in valid_loader:
            loss = self._forward_batch(batch)
            total_loss += loss.item()

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, tokenizer=None, tag: str = "best") -> str:
        save_path = os.path.join(self.output_dir, tag)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)

        print(f"Saved checkpoint to {save_path}")
        return save_path

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        tokenizer=None,
        save_every_epoch: bool = False,
    ):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate(valid_loader)

            print(f"== epoch {epoch}/{num_epochs} == train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(tokenizer=tokenizer, tag="best")

            if save_every_epoch:
                self.save_checkpoint(tokenizer=tokenizer, tag=f"epoch_{epoch}")

        # Always leave a "final" checkpoint too, even if it wasn't the best.
        self.save_checkpoint(tokenizer=tokenizer, tag="final")