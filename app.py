import streamlit as st
from resume_processing import extract_text_from_pdf, preprocess_text, extract_skills, rank_resumes

# Predefined skill set
SKILLS_LIST = ["Python", "Machine Learning", "Data Science", "NLP", "Deep Learning", "SQL", "TensorFlow", "PyTorch"]

st.title("AI Resume Screening & Candidate Ranking System")
st.subheader("Upload Resumes & Enter Job Description")

# Job Description Input
job_desc = st.text_area("Paste the Job Description here:")

# Upload Resumes (Multiple PDFs)
uploaded_files = st.file_uploader("Upload Resumes (PDFs only)", type=["pdf"], accept_multiple_files=True)

if st.button("Process Resumes"):
    if not job_desc or not uploaded_files:
        st.warning("Please enter a job description and upload resumes.")
    else:
        resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        preprocessed_resumes = [preprocess_text(text) for text in resume_texts]
        preprocessed_job_desc = preprocess_text(job_desc)

        # Rank resumes
        ranked_resumes, similarity_scores = rank_resumes(preprocessed_job_desc, preprocessed_resumes)

        # Display Results
        st.subheader("📊 Ranked Resumes:")
        for rank, (index, score) in enumerate(ranked_resumes):
            extracted_skills = extract_skills(resume_texts[index], SKILLS_LIST)
            st.write(f"*Rank {rank+1}:* {uploaded_files[index].name} (Match Score: {score:.2f})")
            st.write(f"✅ Extracted Skills: {', '.join(extracted_skills)}")
        st.write("---")
