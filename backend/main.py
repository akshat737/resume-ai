from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llm_chain import build_resume_chain
from embeddings import embed_texts
import numpy as np

app = FastAPI(title="Resume AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store = {}

@app.post("/upload_resume")
async def upload_resume(user_id: str = Form(...), resume_text: str = Form(...)):
    texts = resume_text.split("\n\n")
    embs = embed_texts(texts)
    memory_store[user_id] = list(zip(texts, embs))
    return {"message": "Resume uploaded successfully", "chunks": len(texts)}

@app.post("/generate")
async def generate(user_id: str = Form(...), job_description: str = Form(...)):
    if user_id not in memory_store:
        return JSONResponse({"error": "No resume found. Please upload first."}, status_code=400)

    user_data = memory_store[user_id]
    job_vec = embed_texts([job_description])[0]
    sims = [float(np.dot(job_vec, e) / (np.linalg.norm(job_vec)*np.linalg.norm(e))) for _, e in user_data]
    top_idxs = np.argsort(sims)[::-1][:5]
    top_ctx = "\n\n".join([user_data[i][0] for i in top_idxs])

    chain = build_resume_chain()
    result = chain.invoke({"job_description": job_description, "resume_context": top_ctx})
    text = result["text"] if isinstance(result, dict) else str(result)

    if "===COVER_LETTER===" in text:
        resume, cover = text.split("===COVER_LETTER===", 1)
    else:
        resume, cover = text, ""

    return {"tailored_resume": resume.strip(), "cover_letter": cover.strip()}

@app.get("/")
def root():
    return {"message": "Resume AI backend is running successfully!"}
