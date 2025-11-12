import os, requests, numpy as np

HF_API_KEY = os.getenv("HF_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_URL = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def embed_texts(texts):
    vectors = []
    for text in texts:
        try:
            r = requests.post(HF_URL, headers=headers, json={"inputs": text}, timeout=60)
            r.raise_for_status()
            data = r.json()
            vec = np.mean(np.array(data), axis=0)
            vectors.append(vec)
        except Exception as e:
            print("Embedding Error:", e, flush=True)
            raise
    return vectors
