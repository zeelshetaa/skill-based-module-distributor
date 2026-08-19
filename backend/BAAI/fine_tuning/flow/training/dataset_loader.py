"""
dataset_loader.py

Loads train_pairs.csv / test_pairs.csv (produced by the triplet-based
pair_builder.py) and wraps them in a PyTorch Dataset + DataLoader,
tokenized for BAAI/bge-base-en-v1.5, ready for trainer.py.

Schema: anchor, positive, negative (+ metadata columns we don't need
for training: project_id, positive_employee_id, negative_employee_id,
positive_score, negative_score).
"""

from typing import List, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_MAX_LENGTH = 256


class TripletDataset(Dataset):
    """
    Wraps one row of (anchor, positive, negative) per item.
    Tokenization is lazy per-item; batching/padding happens in the
    collate_fn so we don't pad every row to a fixed global length.
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        required_cols = {"anchor", "positive", "negative"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): {missing}. "
                f"Did pair_builder.py finish successfully? Expected anchor/positive/negative "
                f"triplet columns, not the old employee_text/project_text/label pair format."
            )

        df = df.dropna(subset=["anchor", "positive", "negative"])

        self.anchors = df["anchor"].astype(str).tolist()
        self.positives = df["positive"].astype(str).tolist()
        self.negatives = df["negative"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "anchor": self.anchors[idx],
            "positive": self.positives[idx],
            "negative": self.negatives[idx],
        }


class TripletCollator:
    """
    Tokenizes a batch of anchor/positive/negative triplets. Three
    separate towers are tokenized (not concatenated) since BGE is
    used as a bi-encoder: embed each side independently, then compare
    via cosine similarity / triplet loss in trainer.py.
    """

    def __init__(self, tokenizer_name: str = DEFAULT_MODEL_NAME, max_length: int = DEFAULT_MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def _tokenize(self, texts: List[str]) -> Dict:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded

    def __call__(self, batch: List[Dict]) -> Dict:
        anchor_texts = [item["anchor"] for item in batch]
        positive_texts = [item["positive"] for item in batch]
        negative_texts = [item["negative"] for item in batch]

        anchor_enc = self._tokenize(anchor_texts)
        positive_enc = self._tokenize(positive_texts)
        negative_enc = self._tokenize(negative_texts)

        return {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "negative_input_ids": negative_enc["input_ids"],
            "negative_attention_mask": negative_enc["attention_mask"],
        }


def get_dataloader(
    csv_path: str,
    tokenizer_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience factory used by trainer.py / train.py:

        train_loader = get_dataloader("datasets/train_pairs.csv", shuffle=True)
        test_loader  = get_dataloader("datasets/test_pairs.csv", shuffle=False)
    """

    dataset = TripletDataset(csv_path)
    collator = TripletCollator(tokenizer_name=tokenizer_name, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # Smoke test: load a batch and print tensor shapes.
    loader = get_dataloader("datasets/train_pairs.csv", batch_size=4, shuffle=False)
    batch = next(iter(loader))

    for key, value in batch.items():
        print(key, tuple(value.shape))