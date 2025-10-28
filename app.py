import streamlit as st
from ingest import ingest_resumes
from main import get_rag_chain
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from utils import list_uploaded_resumes
from job_matcher import evaluate_resume_match

st.set_page_config(page_title="Resume RAG Analyzer", layout="wide")

# Sidebar with instructions and repo link
with st.sidebar:
    st.title("Resume RAG Analyzer")
    st.markdown("""
    **How to use:**
    1. Upload one or more PDF resumes.
    2. Click **Ingest** to process resumes.
    3. Ask questions about the resumes.
    ---
    [GitHub Repo](https://github.com/chedlyklaa/ResumeRag)
    """)

st.title("📄 Resume Analyzer (RAG + Gemini)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if os.path.exists("data/resumes"):
            shutil.rmtree("data/resumes")
        os.makedirs("data/resumes", exist_ok=True)
        for file in uploaded_files:
            file_path = f"data/resumes/{file.name}"
            base, ext = os.path.splitext(file.name)
            counter = 1
            while os.path.exists(file_path):
                file_path = f"data/resumes/{base}_{counter}{ext}"
                counter += 1
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} resume(s).")

    if st.button("Ingest Resumes into Vector Store", disabled=not uploaded_files):
        with st.spinner("Ingesting resumes..."):
            try:
                ingest_resumes()
                st.success("Resumes ingested into vector store!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

with col2:
    st.header("Ask a Question")
    query = st.text_input("Type your question about the resumes")
    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                qa_chain = get_rag_chain()
                output = qa_chain.invoke({"query": query})
                st.markdown("#### Answer")
                st.info(output["result"])
                if "source_documents" in output:
                    with st.expander("Show Source Documents"):
                        for i, doc in enumerate(output["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error during query execution: {e}")

    st.markdown("---")
    st.header("Resume ⇄ Job Match")
    resumes = list_uploaded_resumes()
    chosen = st.selectbox("Select a resume to evaluate", options=["--select--"] + resumes)
    job_desc = st.text_area("Paste the job description here", height=240)
    use_llm = st.checkbox("Use LLM for detailed verdict (requires GOOGLE_API_KEY)", value=False)
    if st.button("Evaluate match", disabled=(chosen == "--select--" or not job_desc.strip())):
        with st.spinner("Evaluating..."):
            try:
                result = evaluate_resume_match(
                    resume_filename=chosen,
                    job_description=job_desc,
                    top_k=5,
                    use_llm=use_llm,
                )
                st.metric("Semantic match score (0-1)", f"{result['score']:.3f}")
                st.subheader("Top relevant passages")
                for i, p in enumerate(result["passages"], 1):
                    st.markdown(f"**Passage {i}** — metadata: {p.get('metadata')}")
                    st.write(p["text"])
                    st.markdown("---")

                if result.get("llm_verdict"):
                    st.subheader("LLM verdict")
                    st.write(result["llm_verdict"])
            except Exception as e:
                st.error(f"Error during evaluation: {e}")

st.markdown("---")
st.caption("Made with Streamlit & LangChain | [GitHub](https://github.com/chedlyklaa/ResumeRag)")
