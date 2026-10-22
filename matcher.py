import io
import os
import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Transparent baseline vocabulary; extend this for your target roles.
SKILLS = {
    "python", "java", "javascript", "typescript", "sql", "mysql", "postgresql",
    "mongodb", "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github actions", "git", "linux", "bash", "powershell",
    "ci/cd", "prometheus", "grafana", "helm", "eks", "aks", "ec2", "s3",
    "iam", "vpc", "lambda", "fastapi", "flask", "django", "react", "node.js",
    "rest api", "machine learning", "deep learning", "nlp", "rag", "langchain",
    "faiss", "hugging face", "pytorch", "pandas", "communication", "leadership",
    "monitoring", "security", "networking", "microservices"
}

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def extract_text(filename: str, data: bytes) -> str:
    """Extract text from PDF, DOCX, or plain text bytes."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    if lower.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts).strip()
    if lower.endswith(".txt"):
        return data.decode("utf-8", errors="replace").strip()
    raise ValueError("Unsupported file. Upload a PDF, DOCX, or TXT file.")

def chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    """Simple paragraph-aware chunker with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:max_chars])
            para = para[max_chars-overlap:]
        candidate = f"{current}\n{para}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def find_skills(text: str) -> list[str]:
    normalized = text.lower()
    found = []
    for skill in sorted(SKILLS, key=len, reverse=True):
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(set(found))

def retrieve_relevant(resume_text: str, jd_text: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Embed resume chunks and retrieve the passages most relevant to the job description."""
    resume_chunks = chunk_text(resume_text)
    jd_chunks = chunk_text(jd_text)
    if not resume_chunks or not jd_chunks:
        return []
    model = get_model()
    resume_vecs = model.encode(resume_chunks, normalize_embeddings=True)
    jd_vecs = model.encode(jd_chunks, normalize_embeddings=True)
    sims = cosine_similarity(jd_vecs, resume_vecs)
    best_for_resume = sims.max(axis=0)
    ranked = np.argsort(best_for_resume)[::-1][:top_k]
    return [
        {"resume_passage": resume_chunks[int(i)],
         "similarity": round(float(best_for_resume[int(i)]), 4)}
        for i in ranked
    ]

def explain_with_ollama(resume_text: str, jd_text: str, evidence: list[dict[str, Any]]) -> str:
    """Optional local LLM explanation. Requires Ollama and a pulled model."""
    try:
        import requests
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        prompt = f"""You are an evidence-based resume analysis assistant.
Compare the resume with the job description using only the evidence below.
Do not invent experience or claim a skill is present without evidence.
Return concise sections: Overall assessment, Strong matches, Gaps to verify, and
three actionable resume improvements. This is career guidance, not a hiring decision.

JOB DESCRIPTION:
{jd_text[:7000]}

RESUME:
{resume_text[:7000]}

RETRIEVED RESUME EVIDENCE:
{evidence}
"""
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as exc:
        return ("LLM explanation unavailable. Start Ollama, pull the configured model, "
                f"and try again. Technical detail: {type(exc).__name__}: {exc}")

def analyze_resume(resume_text: str, jd_text: str, use_llm: bool = False) -> dict[str, Any]:
    if not resume_text.strip() or not jd_text.strip():
        raise ValueError("Both resume text and job description are required.")

    resume_skills = set(find_skills(resume_text))
    jd_skills = set(find_skills(jd_text))
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    # Skill coverage is transparent and interpretable. Embedding similarity is shown separately.
    skill_score = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0
    evidence = retrieve_relevant(resume_text, jd_text)
    semantic_score = (
        float(np.mean([item["similarity"] for item in evidence])) * 100
        if evidence else 0.0
    )
    # Avoid presenting a raw embedding cosine as a calibrated probability.
    overall_score = round(0.7 * skill_score + 0.3 * max(0.0, min(100.0, semantic_score)), 1)

    result = {
        "match_score": overall_score,
        "skill_coverage_score": round(skill_score, 1),
        "semantic_similarity_score": round(semantic_score, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_resume_skills": extra,
        "evidence": evidence,
        "recommendations": [
            f"If accurate, add a concrete project or experience example demonstrating {skill}."
            for skill in missing[:5]
        ],
        "methodology_note": (
            "Heuristic estimate: 70% skill-vocabulary coverage and 30% mean retrieved "
            "embedding similarity. Vocabulary matching can miss synonyms; similarity is "
            "not a calibrated probability. Review evidence manually."
        ),
    }
    if use_llm:
        result["llm_analysis"] = explain_with_ollama(resume_text, jd_text, evidence)
    return result
