# Importing necessary libraries
import faiss
import pickle
import nltk
import re
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from groq import Groq
import streamlit as st

# Explicit NLTK downloader
def download_nltk_data_once():
    packages_to_check = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'wordnet': 'corpora/wordnet',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger_eng'
    }

    packages_to_download = []

    # Check which packages are missing
    for pkg_name, pkg_path in packages_to_check.items():
        try:
            nltk.data.find(pkg_path)
        except LookupError:
            packages_to_download.append(pkg_name)

    # If any are missing, download them all at once and trigger single rerun
    if packages_to_download:
        st.info(f"Downloading missing NLTK packages: {', '.join(packages_to_download)}...")
        for pkg_name in packages_to_download:
            nltk.download(pkg_name, quiet=True)
        st.info(f"Finished downloading necessary NLTK packages")

# Initializing models and components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tfmr = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Preprocessing functions
def wn_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return wordnet.NOUN
    
def preprocess(text):
    tokens=nltk.word_tokenize(re.sub('-',' ',text))
    filtered_tokens=[word.lower() for word in tokens if word.lower() not in stop_words and word not in [';','(',')','{','}',',','.']]
    pos_tags=pos_tag(filtered_tokens)
    lemmatized_tokens=[]
    for word,tag in pos_tags:
        lemmatized_tokens.append(lemmatizer.lemmatize(word,wn_tagger(tag)))
    return "".join(lemmatized_tokens)

# Data Loading
def load_knowledge(st_object):
    download_nltk_data_once()

    """
    Loads and combines static facts and dynamic news articles.
    """
    facts = pd.read_csv('disaster_knowledge.csv')
    try:
        news = pd.read_json('classified_disaster_news.json')
    except ValueError:
        news = pd.DataFrame(columns=['title', 'content'])

    facts.DisasterType = facts.DisasterType.str.replace('_', ' ', regex=False)
    facts.Information = facts.DisasterType + ': ' + facts.Information

    news.content = news.title + ": " + news.content

    if not news.empty:
        knowledge_df = pd.concat([facts.Information, news.content], axis=0, ignore_index=True)
    else:
        knowledge_df = facts.Information
            
    knowledge_df = knowledge_df.dropna()
    return knowledge_df

def build_or_load_index(knowledge_df, index_path="knowledge.index"):
    """
    Builds a new FAISS index or loads it if it already exists.
    """
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        return index
    
    print("Building new FAISS index...")
    
    # Preprocess the text data
    processed_knowledge = knowledge_df.apply(preprocess)
    # Encode the processed text
    knowledge_embds = tfmr.encode(processed_knowledge.tolist())
    
    # Create and add to FAISS index
    d = knowledge_embds.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(knowledge_embds.astype('float32')))
    
    print(f"Saving new index to {index_path}")
    faiss.write_index(index, index_path)
    return index

def retrieve_context(query, index, knowledge_df, k=10):
    """
    Retrieves the top-k relevant documents from the vector store.
    """
    query_embd = tfmr.encode([query])
    D, I = index.search(np.ascontiguousarray(query_embd.astype('float32')), k=k)
    
    # Get the actual text content from the knowledge dataframe
    context_list = [knowledge_df.iloc[idx] for idx in I[0]]
    context = "\n\n---\n\n".join(context_list)
    return context

def get_rag_response(query, context):
    """
    Generates a response from the LLM using the retrieved context.
    """
    prompt = f"""You are a disaster assistance bot. Your job is to answer the user's question with the help of the provided context. Don't mention this to the user.
                CONTEXT:
                {context}
 
                QUESTION:
                {query}

                ANSWER:
                """
    
    try:
        llm_eval = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
        )
        return llm_eval.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, I couldn't connect to the language model to generate an answer."