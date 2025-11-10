import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_core.tracers import context as langsmith_context

def build_resume_chain(user_id: str = "anonymous"):
    """Create a resume and cover letter generator using LangChain 1.x Runnable syntax."""

    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        raise ValueError("HF_API_KEY not found in environment variables")

    # Initialize the LLM (from Hugging Face Hub)
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_key,
        model_kwargs={"temperature": 0.3, "max_new_tokens": 800},
    )

    # Define the prompt
    prompt = PromptTemplate.from_template(
        """You are an expert resume and cover letter writer.

        Job Description:
        {job_description}

        Candidate Resume Extracts:
        {resume_context}

        Task:
        1️⃣ Rewrite the resume to match the job description perfectly.
        2️⃣ Then write a 300-word professional cover letter.
        Separate outputs using '===COVER_LETTER==='."""
    )

    # Runnable pipeline: Prompt → LLM
    chain = prompt | llm

    # Wrap with LangSmith tracing context
    return langsmith_context.with_run_metadata({"user_id": user_id})(chain)
