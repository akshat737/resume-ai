from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
import os

def build_resume_chain():
    prompt = PromptTemplate.from_template("""
    You are an expert resume writer.
    Improve the following resume and create a short cover letter.
    Separate them with "===COVER_LETTER===".

    Job Description:
    {job_description}

    Resume:
    {resume_context}
    """)

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        temperature=0.6,
        max_new_tokens=512,
    )

    return LLMChain(prompt=prompt, llm=llm)
