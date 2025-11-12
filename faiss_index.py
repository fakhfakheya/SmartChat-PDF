import faiss
from typing import List
from embeddings import get_embedding_function

embeddings = get_embedding_function()

DOCUMENTS = []
EMBEDDINGS = None
INDEX = None

def add_to_faiss(documents: List[str]):
    global DOCUMENTS, EMBEDDINGS, INDEX
    DOCUMENTS = documents
    EMBEDDINGS = embeddings.embed_documents(DOCUMENTS)
    dim = EMBEDDINGS.shape[1]
    INDEX = faiss.IndexFlatL2(dim)
    INDEX.add(EMBEDDINGS)
    return INDEX

def query_faiss(q: str, top_k=1):
    if INDEX is None:
        raise ValueError("FAISS index non initialisé")
    q_emb = embeddings.embed_query(q).reshape(1, -1)
    D, I = INDEX.search(q_emb, top_k)
    results = [(DOCUMENTS[i], float(D[0][j])) for j, i in enumerate(I[0])]
    return results
