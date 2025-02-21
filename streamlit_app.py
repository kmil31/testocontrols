import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize session state
if "selected_control" not in st.session_state:
    st.session_state.selected_control = None

# Load models and data with persistent caching
@st.cache_data(persist="disk")
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_data(persist="disk")
def load_controls():
    return pd.read_csv("controls1.csv")

@st.cache_data(persist="disk")
def load_guidances():
    return pd.read_csv("guidance.csv")

# Persistent embedding storage
@st.cache_data(persist="disk")
def get_control_embeddings(_model, controls_df):
    return _model.encode(controls_df['Control_Description'].tolist())

@st.cache_data(persist="disk")
def get_guidance_embeddings(_model, guidance_df):
    return _model.encode(guidance_df['Guidance_Description'].tolist())

# Helper function to sort Control IDs naturally
def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def main():
    st.title("🔒 Cybersecurity Controls & Guidances Search")
    
    # Load data and model
    controls_df = load_controls()
    guidance_df = load_guidances()
    model = load_model()
    
    # Generate or load embeddings
    control_embeddings = get_control_embeddings(model, controls_df)
    guidance_embeddings = get_guidance_embeddings(model, guidance_df)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Controls Search", "Guidance Search"])
    
    # Controls Tab
    with tab1:
        st.header("Search Parent Controls")
        control_query = st.text_input("Search controls (e.g., 'production testing'):", 
                                    key="control_search")
        
        if control_query or st.session_state.selected_control:
            if st.session_state.selected_control:
                # Show linked control
                result = controls_df[controls_df['Control_ID'] == st.session_state.selected_control]
                st.subheader("Linked Control")
                st.markdown(f"**Control ID**: {result['Control_ID'].values[0]}")
                st.markdown(f"**Description**: {result['Control_Description'].values[0]}")
                st.session_state.selected_control = None
            else:
                # Semantic search
                query_embedding = model.encode([control_query])
                similarities = cosine_similarity(query_embedding, control_embeddings)
                top_indices = np.argsort(similarities[0])[::-1][:10]
                
                
                for idx in top_indices:
                    st.markdown(f"### {controls_df.iloc[idx]['Control_ID']}")
                    st.markdown(f"**Description**: {controls_df.iloc[idx]['Control_Description']}")
                    st.markdown("---")  # Optional separator
                
          
    
    # Guidance Tab
    with tab2:
        st.header("Search Guidances")
        guidance_query = st.text_input("Search guidances (e.g., 'OWASP testing'):", 
                                     key="guidance_search")
        
        if guidance_query:
            # Semantic search
            query_embedding = model.encode([guidance_query])
            similarities = cosine_similarity(query_embedding, guidance_embeddings)
            top_indices = np.argsort(similarities[0])[::-1][:10]
            
            for idx in top_indices:
                guidance = guidance_df.iloc[idx]
                st.markdown(f"### {guidance['Guidance_ID']}")
                st.markdown(f"**Description**: {guidance['Guidance_Description']}")
                st.markdown(f"**Linked Control ID**: {guidance['Control_ID']}")
                st.markdown(f"**Control Name**: {guidance['Control_Description']}")
                st.markdown("---")  # Optional separator
            

if __name__ == "__main__":
    main()