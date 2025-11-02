import streamlit as st
from rag_pipeline import load_knowledge, build_or_load_index, retrieve_context, get_rag_response
import os
import nltk

# Page config
st.set_page_config(
    page_title="DisasterGPT",
    page_icon="🚨",
    layout="centered"
)

# API key mgmt
if 'GROQ_API_KEY' not in os.environ:
    # If key is not in environment, try to get from Streamlit secrets
    try:
        os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
    except:
        st.error("GROQ_API_KEY not found. Please set it in your environment or Streamlit secrets.")
        st.stop()

# Ecplicit NLTK download
def download_nltk_data():
    packages = ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
    print("Checking NLTK packages...")
    for package in packages:
        try:
            if package in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{package}')
            elif package == 'averaged_perceptron_tagger':
                nltk.data.find(f'taggers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
            # print(f"NLTK package '{package}' already downloaded.")
        except LookupError:
            print(f"NLTK package '{package}' not found. Downloading...")
            nltk.download(package, quiet=True)
            print(f"Downloaded '{package}'.")
    print("NLTK check complete.")

# Caching
@st.cache_resource
def load_data_and_index():
    st.info("Loading knowledge base... (This happens once per session)")
    knowledge_df = load_knowledge(st)
    index = build_or_load_index(knowledge_df)
    st.success("Knowledge base loaded!")
    return index, knowledge_df

# Main Application
st.title("🚨 DisasterGPT")
st.markdown("Your AI assistant for disaster information and real-time news.")

# Load and index data
try:
    index, knowledge_df = load_data_and_index()

    # User Input
    query = st.text_input("Ask a question:", placeholder="e.g., What should I do during an earthquake?")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Finding the best answer..."):
                # Retrieve context
                context = retrieve_context(query, index, knowledge_df)
                # Generate response
                response = get_rag_response(query, context)
                
                st.markdown(response)
                
                # Optional: Show the sources
                with st.expander("Show retrieved context"):
                    st.text(context)
        else:
            st.warning("Please enter a question.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Could not load the knowledge base. Please check the data files and restart the app.")