import streamlit as st
from matcher import analyze_resume, extract_text

st.set_page_config(page_title="Resume ↔ Job Matcher", page_icon="🧭", layout="wide")
st.title("🧭 Resume vs Job Description Matcher")
st.caption("RAG-powered career guidance • Evidence-first analysis • Local embeddings")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Upload a resume\n"
        "2. Paste a job description\n"
        "3. Retrieve relevant resume passages\n"
        "4. Review skill coverage and evidence"
    )
    use_llm = st.toggle("Generate local LLM explanation", value=False)
    st.caption("LLM mode requires Ollama and a downloaded model.")

uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
jd_text = st.text_area("Paste the complete job description", height=260,
                       placeholder="Responsibilities, required skills, qualifications...")
analyze = st.button("Analyze match", type="primary", use_container_width=True)

if analyze:
    if not uploaded:
        st.error("Please upload a resume first.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        try:
            with st.spinner("Extracting text, embedding passages and retrieving evidence..."):
                resume_text = extract_text(uploaded.name, uploaded.getvalue())
                if len(resume_text) < 30:
                    st.error("Very little text was extracted. If this is a scanned PDF, run OCR first.")
                    st.stop()
                result = analyze_resume(resume_text, jd_text, use_llm=use_llm)

            st.subheader("Match overview")
            a, b, c = st.columns(3)
            a.metric("Estimated match", f"{result['match_score']}%")
            b.metric("Skill coverage", f"{result['skill_coverage_score']}%")
            c.metric("Semantic similarity", f"{result['semantic_similarity_score']}%")
            st.progress(min(1.0, max(0.0, result["match_score"] / 100)))
            st.info(result["methodology_note"])

            left, right = st.columns(2)
            with left:
                st.markdown("### ✅ Matched skills")
                st.write(", ".join(result["matched_skills"]) or "No vocabulary matches detected.")
                st.markdown("### ➕ Additional resume skills")
                st.write(", ".join(result["additional_resume_skills"]) or "None detected.")
            with right:
                st.markdown("### 🔎 Skills to review")
                st.write(", ".join(result["missing_skills"]) or "No missing vocabulary items detected.")
                st.caption("A missing item means it was not detected in the resume text, not necessarily that you lack it.")

            st.markdown("### Evidence retrieved from the resume")
            for i, item in enumerate(result["evidence"], start=1):
                with st.expander(f"Passage {i} · similarity {item['similarity']:.3f}"):
                    st.write(item["resume_passage"])

            st.markdown("### Suggested improvements")
            if result["recommendations"]:
                for suggestion in result["recommendations"]:
                    st.write(f"- {suggestion}")
            else:
                st.write("Highlight measurable outcomes and tailor relevant project descriptions to the role.")

            if result.get("llm_analysis"):
                st.markdown("### Local LLM analysis")
                st.write(result["llm_analysis"])

            st.download_button(
                "Download analysis as JSON",
                data=__import__("json").dumps(result, indent=2),
                file_name="resume_match_report.json",
                mime="application/json"
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

st.divider()
st.caption("Privacy: this starter app processes uploaded files in the running application. Avoid uploading sensitive personal data to any deployment you do not control.")
