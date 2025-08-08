import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_PATH = "db/chroma"

def ingest_resumes():
    resumes_folder = "data/resumes"
    all_chunks = []

    for filename in os.listdir(resumes_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(resumes_folder, filename))
            documents = loader.load()

            # Improved chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,      # Increase for more context per chunk
                chunk_overlap=100,   # Overlap to preserve context between chunks
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(all_chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"Ingested {len(all_chunks)} chunks into Chroma DB.")

if __name__ == "__main__":
    ingest_resumes()
