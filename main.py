import streamlit as st
import pickle
import pandas as pd
import joblib
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model():
    model = joblib.load("matcher_assets.pkl")  
    return model
assets  = load_model()
model = assets['pipeline']
resume_vectors = assets['resume_vectors']
df = assets['original_df']
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
st.title("Resume Scoring")
st.write("Paste Resume here")

job_description = st.text_input("Job Description ")
if uploaded_files and job_description:
    job_vector = model.transform([job_description])
results = []
if uploaded_files:
    for uploaded_file in uploaded_files:
   
        resume_text = extract_text_from_pdf(uploaded_file)
        
        st.write(f"Processed: {uploaded_file.name}")
        new_resume_vector = model.transform([resume_text])
        score = cosine_similarity(new_resume_vector, job_vector)


match_percentage = round(score[0][0] * 100, 2)
st.write(f"Match Score: {match_percentage}%")
