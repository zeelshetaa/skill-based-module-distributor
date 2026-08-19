"""
full file changes by drashti
embedding_service.py

Generates embeddings using the fine-tuned BGE model
instead of the generic all-MiniLM-L6-v2 model.
"""

from backend.BAAI.fine_tuning.flow.inference.embedding_generator import EmbeddingGenerator


def generate_embedding(text):
    embedding = EmbeddingGenerator.generate(text)   # torch tensor, shape (384,)
    return embedding.detach().cpu().numpy()          # so .tolist() downstream still works
