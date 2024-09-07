import streamlit as st
import warnings
import re
import pandas as pd
from pymilvus import Collection, connections
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from transformers import BertTokenizer, BertModel
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import numpy as np

# Initialize the connection to Milvus
warnings.filterwarnings("ignore")
ENDPOINT="https://in03-0008120f0b8b227.serverless.gcp-us-west1.cloud.zilliz.com"
connections.connect(
   uri=ENDPOINT,
   token="2e021870ed0adebedcf7a869bc5df9905510a5d53114ee7aff6fd8f08d7799bb511c6de82f13d3c4bf45740ca0f0d4d26c0fdcb7")

collection_name = "demo"
collection = Collection(name=collection_name)

# BERT-based text processing pipeline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')

# Custom BERT embedding transformer
class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text = [X]
        encoding = self.tokenizer.batch_encode_plus(text, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state
        sentence_embedding = word_embeddings.mean(dim=1)
        return sentence_embedding.numpy().flatten()

# Text cleaning functions
def lower_transform(text):
    return text.lower()

def remove_excess_whitespace(text):
    stripped_text = text.strip()
    cleaned_text = ' '.join(stripped_text.split())
    return cleaned_text

# Initialize pipeline
lowercase_transformer = FunctionTransformer(lower_transform, validate=False)
whitespace_transformer = FunctionTransformer(remove_excess_whitespace, validate=False)
bert_embedding_transformer = BertEmbeddingTransformer(tokenizer, model)

embeddings_pipeline = Pipeline([
    ('lowercase', lowercase_transformer),
    ('whitespace', whitespace_transformer),
    ('bert_embedding', bert_embedding_transformer)
])

# Store description in Milvus
def get_description(desc, email):
    embeddings = embeddings_pipeline.transform(desc)
    data_rows = [{"vector": embeddings, "pk": int(email)}]
    collection.insert(data_rows)
    collection.flush()
    return "Data stored successfully."

# Retrieve recommendations from Milvus
def get_recommendation(description):
    embeddings = embeddings_pipeline.transform(description)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(data=[embeddings], anns_field="vector", output_fields=["pk"], limit=3, param=search_params)
    pattern = r"'pk': (\d+)"
    ids = []
    for result in results[0]:
        result_str = str(result)
        match = re.search(pattern, result_str)
        if match:
            ids.append(match.group(1))
    return ids

# Streamlit UI
st.set_page_config(page_title="Job Matching Platform", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Job Matching Platform</h1>
    """,
    unsafe_allow_html=True
)

# Define layout
col1, col2 = st.columns(2)

# User Input Section
with col1:
    st.header("Store Your Data")
    with st.expander("User Input"):
        skills = st.text_input("Enter your description", placeholder="Describe your skills and experience")
        email = st.text_input("Enter your email", placeholder="Your email")
    if st.button("Store My Data"):
        if skills and email:
            result = get_description(skills, email)
            st.success(result)
            st.session_state.messages = st.session_state.get('messages', [])
            st.session_state.messages.append({"role": "assistant", "content": "Your data has been stored successfully!"})
        else:
            st.error("Please enter both description and email.")

# Recommendation Section
with col2:
    st.header("Get Applicant Recommendations")
    search_description = st.text_input("Enter a job description to find matching applicants", placeholder="Describe what you're looking for")
    if st.button("Get Recommendations"):
        if search_description:
            recommended_ids = get_recommendation(search_description)
            st.write("Recommended Applicant IDs:", recommended_ids)
        else:
            st.error("Please enter a description for recommendations.")
