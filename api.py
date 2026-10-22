from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from matcher import analyze_resume

app = FastAPI(title="Resume Job Matcher API", version="1.0.0")

class MatchRequest(BaseModel):
    resume_text: str = Field(min_length=30, description="Extracted resume text")
    job_description: str = Field(min_length=30)
    use_llm: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/match")
def match(request: MatchRequest):
    try:
        return analyze_resume(request.resume_text, request.job_description, request.use_llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
