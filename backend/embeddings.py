from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """Convert text into embeddings for similarity comparison"""
    return model.encode(texts).tolist()
