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

collection_name = "jp"
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
    # Check if the email already exists in the collection
    expr = f"email == '{email}'"
    results = collection.query(expr, output_fields=["email"], limit=1)

    # Generate embeddings for the provided description
    embeddings = embeddings_pipeline.transform(desc)
    data_rows = [{"vector": embeddings, "email": email,"description":desc}]

    # If the email exists, update (upsert) the data
    if len(results) > 0:
        collection.upsert(data_rows)  # Replace the existing data
        collection.flush()
        return "Data updated successfully."
    else:
        # If the email does not exist, insert new data
        collection.insert(data_rows)
        collection.flush()
        return "Data stored successfully."

# Retrieve recommendations from Milvus
def get_recommendation(description):
    embeddings = embeddings_pipeline.transform(description)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(data=[embeddings], anns_field="vector", output_fields=["email","description"], limit=3, param=search_params)
    for hits in results:
        emails=[]
        descriptions=[]
        for hit in hits:
            # gets the value of an output field specified in the search request.
            # dynamic fields are supported, but vector fields are not supported yet.    
            emails.append(hit.entity.get('email'))
            descriptions.append(hit.entity.get('description'))
    return emails,descriptions

def delete_data(email):
    expr = f"email == '{email}'"
    results = collection.query(expr, output_fields=["email"], limit=1)
    if len(results) > 0:
        res = collection.delete(
        
            expr=f'email == "{email}"' 
                    
        )
        collection.flush()
        return "Your data has been deleted successfully!"
    else:
     
        return "No data found for the given email."

# Streamlit UI
st.set_page_config(page_title="Job Matching Platform", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Job Matching Platform</h1>
    """,
    unsafe_allow_html=True
)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Define layout
col1, col2 = st.columns(2)

# User Input Section
with col1:
    st.header("Store/update Your Data")
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
            email,des = get_recommendation(search_description)
            st.write("email:", email)
            st.write("description:", des)

        else:
            st.error("Please enter a description for recommendations.")
st.markdown("---")

# Second row with one column (Delete My Data)
col3 = st.columns(1)[0]

with col3:
    st.header("Delete My Data")
    with st.expander("User Input"):
        email_to_delete = st.text_input("Enter your email to delete your data", placeholder="Your email", key='delete_email_input')
    if st.button("Delete My Data"):
        if email_to_delete:
            result = delete_data(email_to_delete)
            st.success(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        else:
            st.error("Please enter your email.")
