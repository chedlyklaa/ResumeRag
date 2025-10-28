Resume Analyzer (RAG-Powered)

The Resume Analyzer is an intelligent system that leverages Retrieval-Augmented Generation (RAG) to analyze resumes and evaluate candidate fit against job descriptions.

Key Features

Resume Upload via Streamlit UI – Upload one or more PDF resumes directly through an intuitive web interface.

Text Extraction & Chunking – Automatically converts PDFs to text and splits content into manageable semantic chunks.

Semantic Embeddings – Uses SentenceTransformer to embed chunks and store them in a Chroma vector database for efficient similarity search.

RAG-Based Question Answering – On query, retrieves the most relevant resume sections and feeds them to a configurable cloud LLM (e.g., Google Gemini, OpenAI GPT), generating precise and context-aware answers.

Resume ↔ Job Matching – Compares resumes against job descriptions using semantic similarity scoring. Optionally, the LLM provides a human-readable assessment explaining the candidate’s suitability.

Tech Stack

Frontend: Streamlit

Embedding: SentenceTransformer

Database: Chroma (Vector DB)

LLM Integration: Google Gemini / OpenAI (configurable)

Backend Logic: Python

Use Cases

Recruiters or HR professionals seeking automated resume screening.

Job seekers analyzing how well their resume matches a specific job description.

Developers exploring practical RAG implementations with real-world data.

Run Locally
$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
pip install -r requirements.txt
streamlit run app.py
