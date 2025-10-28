from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from job_matcher import evaluate_resume_match
import uvicorn

app = FastAPI(title="ResumeRag API")

class MatchRequest(BaseModel):
    resume_filename: str
    job_description: str
    top_k: Optional[int] = 5
    use_llm: Optional[bool] = False

@app.post("/match")
def match(req: MatchRequest):
    try:
        result = evaluate_resume_match(
            resume_filename=req.resume_filename,
            job_description=req.job_description,
            top_k=req.top_k,
            use_llm=req.use_llm,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)