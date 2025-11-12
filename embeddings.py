from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]

def get_embedding_function():
    return SentenceTransformerEmbeddings()
