"""
full file changes by drashti
embedding_generator.py

Generate embeddings using the fine-tuned BGE model.
Pooling strategy MUST match what was used during training
(see train.py --pooling, default = "cls").
"""

import torch
import torch.nn.functional as F

from backend.BAAI.fine_tuning.flow.inference.model_loader import ModelLoader

# Set this to whatever pooling was actually used when train.py was run.
# Default in train.py is "cls" -- change to "mean" only if training was
# explicitly run with --pooling mean.
POOLING_MODE = "cls"


class EmbeddingGenerator:

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9
        )

    @staticmethod
    def _cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    @staticmethod
    def generate(text: str):

        tokenizer, model = ModelLoader.load()

        encoded_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = model(**encoded_input)

        if POOLING_MODE == "cls":
            embedding = EmbeddingGenerator._cls_pooling(model_output)
        else:
            embedding = EmbeddingGenerator._mean_pooling(
                model_output,
                encoded_input["attention_mask"]
            )

        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0)
