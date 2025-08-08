import os

def list_uploaded_resumes(folder="data/resumes"):
    return [f for f in os.listdir(folder) if f.endswith(".pdf")]
