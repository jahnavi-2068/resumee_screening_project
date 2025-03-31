import pdfplumber
import spacy
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Download NLTK tokenizer
nltk.download("punkt")

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face Transformer for keyword extraction
keyword_extractor = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Function to extract skills from text
def extract_skills(text, skills_list):
    skills = keyword_extractor(text, candidate_labels=skills_list, multi_label=True)
    return [skills_list[i] for i, score in enumerate(skills["scores"]) if score > 0.5]

# Function to rank resumes based on job description
def rank_resumes(job_desc, resumes):
    documents = [job_desc] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    ranked_resumes = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)

    return ranked_resumes, similarity_scores
