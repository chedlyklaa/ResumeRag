from typing import Dict, Any, List, Optional
import os
import numpy as np

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Try available LLM wrappers
_HAS_GOOGLE = False
_HAS_OPENAI = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GOOGLE = True
except Exception:
    _HAS_GOOGLE = False

try:
    from langchain_openai import ChatOpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

CHROMA_PATH = "db/chroma"
_EMBED_MODEL = "all-MiniLM-L6-v2"


def _cosine(a: List[float], b: List[float]) -> float:
    a = np.array(a); b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _get_llm(provider: str, model: str):
    if provider == "google" and _HAS_GOOGLE:
        return ChatGoogleGenerativeAI(model=model)
    if provider == "openai" and _HAS_OPENAI:
        return ChatOpenAI(model=model, temperature=0.0)
    return None


def evaluate_resume_match(
    resume_filename: str,
    job_description: str,
    top_k: int = 5,
    use_llm: bool = False,
    llm_provider: str = "google",
    llm_model: str = "gemini-1.5",
) -> Dict[str, Any]:
    """
    Return a semantic match score (0..1), top passages, and optional LLM verdict string.
    """
    embeddings = SentenceTransformerEmbeddings(model_name=_EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # try filtered similarity search by source metadata (common when ingesting)
    try:
        docs = db.similarity_search(job_description, k=top_k, filter={"source": resume_filename})
    except Exception:
        docs_all = db.similarity_search(job_description, k=top_k * 3)
        docs = [d for d in docs_all if resume_filename in d.metadata.get("source", "")][:top_k]

    passages = [{"text": d.page_content, "metadata": d.metadata} for d in docs]

    if not docs:
        return {"score": 0.0, "passages": passages, "llm_verdict": None}

    # compute semantic score
    try:
        job_vec = embeddings.embed_query(job_description)
    except Exception:
        job_vec = embeddings.embed_documents([job_description])[0]

    doc_texts = [d.page_content for d in docs]
    doc_vecs = embeddings.embed_documents(doc_texts)
    sims = [_cosine(job_vec, v) for v in doc_vecs]
    score = float(np.mean(sims))

    llm_verdict: Optional[str] = None
    if use_llm:
        llm = _get_llm(llm_provider, llm_model)
        if llm is not None:
            # build a compact prompt asking for a JSON response
            prompt = (
                "You are an assistant that decides if a resume matches a job description. "
                "Respond with a JSON object: {\"match\":\"yes\"/\"no\", \"score\":0-1, \"reasons\":[...]}.\n\n"
                f"Job description:\n{job_description}\n\nResume excerpts:\n"
            )
            for i, p in enumerate(passages, 1):
                prompt += f"\n[{i}] {p['text'][:800]}\n"

            prompt += "\n\nProvide the JSON only."

            try:
                # Try common APIs; fall back gracefully
                if hasattr(llm, "predict"):
                    llm_verdict = llm.predict(prompt)
                else:
                    gen = llm.generate([prompt])
                    # Best-effort extraction
                    if getattr(gen, "generations", None):
                        llm_verdict = gen.generations[0][0].text
                    else:
                        llm_verdict = str(gen)
            except Exception as e:
                llm_verdict = f"LLM call failed: {e}"
        else:
            llm_verdict = "LLM provider not available in environment."

    return {"score": score, "passages": passages, "llm_verdict": llm_verdict}