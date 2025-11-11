from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llm_chain import build_resume_chain
from embeddings import embed_texts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="Resume AI")

# Allow frontend (Vercel) to call backend (Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your Vercel domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary memory (resets if server restarts)
memory_store = {}

@app.post("/upload_resume")
async def upload_resume(user_id: str = Form(...), resume_text: str = Form(...)):
    """Store resume temporarily in memory"""
    texts = resume_text.split("\n\n")
    embs = embed_texts(texts)
    memory_store[user_id] = list(zip(texts, embs))
    return {"message": "Resume uploaded successfully", "chunks": len(texts)}

@app.post("/generate")
async def generate(user_id: str = Form(...), job_description: str = Form(...)):
    """Generate tailored resume and cover letter"""
    if user_id not in memory_store:
        return JSONResponse({"error": "No resume found. Please upload first."}, status_code=400)

    user_data = memory_store[user_id]
    job_vec = embed_texts([job_description])[0]
    doc_vecs = np.array([e for _, e in user_data])
    sims = cosine_similarity([job_vec], doc_vecs)[0]
    idxs = np.argsort(sims)[::-1][:5]
    top_ctx = "\n\n".join([user_data[i][0] for i in idxs])

    # 👇 FIXED: call build_resume_chain() with no user_id (LangChain doesn't need it)
    chain = build_resume_chain()
    output = chain.invoke({
        "job_description": job_description,
        "resume_context": top_ctx
    })

    if isinstance(output, dict) and "text" in output:
        output = output["text"]

    if "===COVER_LETTER===" in output:
        resume, cover = output.split("===COVER_LETTER===", 1)
    else:
        resume, cover = output, ""

    return {
        "tailored_resume": resume.strip(),
        "cover_letter": cover.strip(),
        "message": "Generated successfully!"
    }

@app.get("/")
def root():
    return {"message": "Resume AI backend is running successfully!"}
