import streamlit as st
from ingest import ingest_resumes
from main import get_rag_chain
import os
import shutil

st.set_page_config(page_title="Resume RAG Analyzer", layout="centered")

st.title("📄 Resume Analyzer (RAG + Ollama)")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Clear old resumes before saving new ones
    if os.path.exists("data/resumes"):
        shutil.rmtree("data/resumes")
    os.makedirs("data/resumes", exist_ok=True)
    for file in uploaded_files:
        file_path = f"data/resumes/{file.name}"
        # Handle duplicate filenames
        base, ext = os.path.splitext(file.name)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"data/resumes/{base}_{counter}{ext}"
            counter += 1
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} resume(s).")

# Ingest button
if st.button("Ingest Resumes into Vector Store", disabled=not uploaded_files):
    with st.spinner("Ingesting resumes..."):
        try:
            ingest_resumes()
            st.success("Resumes ingested into vector store!")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")

st.markdown("---")
st.markdown("### Ask a question about the resumes:")
query = st.text_input("Your question here")

if query:
    with st.spinner("Searching and generating answer..."):
        try:
            qa_chain = get_rag_chain()
            output = qa_chain.invoke({"query": query})
            st.markdown("**Answer:**")
            st.write(output["result"])
            # Optionally show sources:
            if "source_documents" in output:
                st.markdown("**Source Documents:**")
                for doc in output["source_documents"]:
                    st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error during query execution: {e}")
