import os
import requests
import numpy as np

HF_API_KEY = os.getenv("HF_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def embed_texts(texts):
    """Return embeddings using Hugging Face hosted API (lightweight)."""
    vectors = []
    for text in texts:
        payload = {"inputs": text}
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Handle nested lists from HF output
        vec = np.mean(np.array(data), axis=0)
        vectors.append(vec)
    return vectors
