from matcher import chunk_text, find_skills, analyze_resume

def test_chunk_text_returns_content():
    chunks = chunk_text("Python developer\nUses Docker and AWS.")
    assert chunks
    assert "Python" in " ".join(chunks)

def test_find_skills_detects_common_skills():
    found = find_skills("Built Docker containers and deployed on AWS using Python.")
    assert "docker" in found
    assert "aws" in found
    assert "python" in found

def test_analyze_returns_expected_fields(monkeypatch):
    # Keep the unit test offline and fast by replacing embedding retrieval.
    monkeypatch.setattr(
        "matcher.retrieve_relevant",
        lambda resume, jd, top_k=4: [
            {"resume_passage": "Used Docker and Python in a project.", "similarity": 0.8}
        ],
    )
    result = analyze_resume(
        "Python developer with Docker experience and Linux.",
        "Required: Python, Docker, Kubernetes and AWS."
    )
    assert 0 <= result["match_score"] <= 100
    assert "python" in result["matched_skills"]
    assert "kubernetes" in result["missing_skills"]
