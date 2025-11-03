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
import json
from gdacs.api import GDACSAPIReader
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from googleapiclient.discovery import build

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
        # st.info(f"Downloading missing NLTK packages: {', '.join(packages_to_download)}...")
        for pkg_name in packages_to_download:
            nltk.download(pkg_name, quiet=True)
        # st.info(f"Finished downloading necessary NLTK packages")

# Initializing models and components
lemmatizer = WordNetLemmatizer()
tfmr = SentenceTransformer("all-MiniLM-L6-v2")

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
    stop_words = set(stopwords.words('english'))
    tokens=nltk.word_tokenize(re.sub('-',' ',text))
    filtered_tokens=[word.lower() for word in tokens if word.lower() not in stop_words and word not in [';','(',')','{','}',',','.']]
    pos_tags=pos_tag(filtered_tokens)
    lemmatized_tokens=[]
    for word,tag in pos_tags:
        lemmatized_tokens.append(lemmatizer.lemmatize(word,wn_tagger(tag)))
    return "".join(lemmatized_tokens)

# Data Loading
def load_knowledge(st_object):
    """
    Loads static facts and classified news, returning TWO separate dataframes.
    """
    download_nltk_data_once() # NLTK check
    
    # Static "what-to-do" facts
    facts_df = pd.read_csv('disaster_knowledge.csv')
    facts_df['DisasterType'] = facts_df['DisasterType'].str.replace('_', ' ', regex=False)
    facts_df['full_text'] = facts_df['DisasterType'] + ': ' + facts_df['Information']
    
    # Dynamic "what's-happening" news
    try:
        news_df = pd.read_json('classified_disaster_news.json')
        news_df['full_text'] = news_df['title'] + ": " + news_df['content']
        news_df = news_df.dropna(subset=['full_text', 'matched_keyword'])
    except ValueError:
        print("Classified news file not found or is empty. Proceeding with facts only.")
        news_df = pd.DataFrame(columns=['title', 'content', 'full_text', 'matched_keyword'])
        
    return facts_df, news_df

def build_or_load_index(facts_df, news_df, index_dir="vector_indices"):
    """
    Builds or loads a separate FAISS index for 'facts' and each news category.
    """
    os.makedirs(index_dir, exist_ok=True)
    
    indices = {}
    knowledge_bases = {}
    
    # Index for static 'facts'
    facts_index_path = os.path.join(index_dir, "facts.index")
    facts_data_path = os.path.join(index_dir, "facts_data.pkl")
    
    if os.path.exists(facts_index_path):
        print("Loading existing 'facts' index...")
        indices['facts'] = faiss.read_index(facts_index_path)
        with open(facts_data_path, 'rb') as f:
            knowledge_bases['facts'] = pickle.load(f)
    else:
        print("Building 'facts' index...")
        knowledge_bases['facts'] = facts_df['full_text'].reset_index(drop=True)
        processed_text = knowledge_bases['facts'].apply(preprocess)
        embeddings = tfmr.encode(processed_text.tolist())
        
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(embeddings.astype('float32')))
        
        indices['facts'] = index
        faiss.write_index(index, facts_index_path)
        with open(facts_data_path, 'wb') as f:
            pickle.dump(knowledge_bases['facts'], f)

    # Index for 'news' (one per category)
    news_categories = news_df['matched_keyword'].unique()
    
    for category in news_categories:
        if category == 'no_category':
            continue # Skip unclassified articles
        
        category_index_path = os.path.join(index_dir, f"news_{category}.index")
        category_data_path = os.path.join(index_dir, f"news_{category}_data.pkl")
        
        category_df = news_df[news_df['matched_keyword'] == category]
        
        if os.path.exists(category_index_path):
            print(f"Loading existing 'news_{category}' index...")
            indices[category] = faiss.read_index(category_index_path)
            with open(category_data_path, 'rb') as f:
                knowledge_bases[category] = pickle.load(f)
        else:
            print(f"Building 'news_{category}' index...")
            knowledge_bases[category] = category_df['full_text'].reset_index(drop=True)
            if knowledge_bases[category].empty:
                print(f"Skipping empty category: {category}")
                continue
                
            processed_text = knowledge_bases[category].apply(preprocess)
            embeddings = tfmr.encode(processed_text.tolist())
            
            d = embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(np.ascontiguousarray(embeddings.astype('float32')))
            
            indices[category] = index
            faiss.write_index(index, category_index_path)
            with open(category_data_path, 'wb') as f:
                pickle.dump(knowledge_bases[category], f)

    return indices, knowledge_bases

def intelligent_query_analyzer(query):
    """
    Uses an LLM to analyze a complex query and return a structured
    JSON search plan.
    """
    print(f"Analyzing complex query: {query}")
    
    # We use a smaller, faster model for this simple task
    # This is "lazily" instantiated to avoid API key errors
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # We give the LLM a "persona" and clear instructions
    system_prompt = """You are an expert query routing assistant. Your job is to analyze a user's question about disasters and generate a JSON "search_plan".

You have two types of indexes you can search:
1.  `"facts"`: This index contains static, evergreen "what-to-do" information. Use this for questions about preparation, safety procedures, definitions, first aid, warning signs, etc.
2.  `"news"`: This index contains real-time news articles. Use this for questions about "what's happening," "latest updates," or events in specific locations.

Your task is to:
1.  Break the user's question into 1-3 sub-queries.
2.  For each sub-query, decide which index to search (`"facts"` or `"news"`).
3.  If a sub-query is for "news," also detect the disaster type (e.g., 'earthquake', 'flood', 'wildfire') and use that as the index name.
4.  If no disaster type is mentioned for a news query, default to `"facts"`.
5.  Return ONLY a valid JSON object in the format: `{"search_plan": [{"sub_query": "...", "index_to_search": "..."}]}`

Example 1:
User: "What are the health risks of a wildfire and how do I prepare for one?"
{
  "search_plan": [
    {"sub_query": "health risks of wildfire", "index_to_search": "facts"},
    {"sub_query": "how to prepare for wildfire", "index_to_search": "facts"}
  ]
}

Example 2:
User: "What's the latest on the earthquake in Myanmar and what should I do?"
{
  "search_plan": [
    {"sub_query": "latest news on earthquake in Myanmar", "index_to_search": "earthquake"},
    {"sub_query": "what to do during an earthquake", "index_to_search": "facts"}
  ]
}

Example 3:
User: "Tell me about hurricanes."
{
  "search_plan": [
    {"sub_query": "what is a hurricane", "index_to_search": "facts"}
  ]
}
"""
    
    try:
        llm_eval = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}, # Ask Groq for JSON directly
        )
        response_text = llm_eval.choices[0].message.content
        search_plan_data = json.loads(response_text)
        
        # Basic validation
        if 'search_plan' in search_plan_data and isinstance(search_plan_data['search_plan'], list):
            return search_plan_data['search_plan']
        else:
            raise Exception("Invalid JSON structure")
            
    except Exception as e:
        print(f"Error in intelligent_query_analyzer: {e}. Defaulting to basic 'facts' search.")
        # Fallback for any error: just search 'facts' index
        return [{"sub_query": query, "index_to_search": "facts"}]

def retrieve_context(search_plan, indices, knowledge_bases, k=3):
    """
    Retrieves relevant documents based on a structured 'search_plan'.
    'search_plan' is a list of dictionaries:
    [{"sub_query": "...", "index_to_search": "..."}, ...]
    """
    all_context = []
    all_indices_searched = set()
    
    # If search_plan is None or not a list, fallback
    if not isinstance(search_plan, list):
        print("Invalid search plan. Defaulting to 'facts' search.")
        search_plan = [{"sub_query": str(search_plan), "index_to_search": "facts"}]

    for step in search_plan:
        query = step.get("sub_query")
        target_index_name = step.get("index_to_search")

        # If the analyzer picked a news category we don't have, default to 'facts'
        if target_index_name not in indices:
            print(f"No index found for '{target_index_name}', defaulting to 'facts' index.")
            target_index_name = 'facts'
        
        all_indices_searched.add(target_index_name)
            
        # Get the specific index and knowledge base
        index = indices[target_index_name]
        knowledge_df = knowledge_bases[target_index_name]
        print(f"Searching in '{target_index_name}' index for query: '{query}'")

        # Search for this sub-query
        query_embd = tfmr.encode([query])
        D, I = index.search(np.ascontiguousarray(query_embd.astype('float32')), k=k)
        
        context_list = [knowledge_df.iloc[idx] for idx in I[0] if idx < len(knowledge_df)]
        all_context.extend(context_list)

    # Remove duplicate documents
    unique_context = list(dict.fromkeys(all_context))
    
    combined_context = "\n\n---\n\n".join(unique_context)
    indices_str = ", ".join(all_indices_searched)
    
    return combined_context, indices_str

def get_rag_response(query, context):
    """
    Generates a response from the LLM using the retrieved context.
    """
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f""" Answer the question only from provided context as far as possible. Don't refer to provided context in response. 
                CONTEXT:
                {context}
 
                QUESTION:
                {query}
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
    
# GDACS real-time alerts
# --- EXTENSION 3: GDACS Real-time Alerts ---

@st.cache_data(ttl=600) # Cache alerts for 10 minutes
def fetch_gdacs_alerts(limit=10):
    """
    Fetches the latest real-time disaster alerts from GDACS.
    """
    print("Fetching latest GDACS alerts...")
    try:
        client = GDACSAPIReader()
        # client.latest_events() returns a GeoJSON object
        events_object = client.latest_events(limit=limit) 

        # The list we want is in the .features attribute
        if hasattr(events_object, 'features'):
            return events_object.features  # Return the list of feature objects
        else:
            return [] # Return an empty list if no features

    except Exception as e:
        print(f"Error fetching GDACS alerts: {e}")
        return None # Return None on error

# Multimedia "How-To"
@st.cache_data(ttl=3600) # Cache video links for 1 hour
def get_youtube_video_link(query):
    """
    Searches YouTube for a relevant "how-to" video and
    returns its link AND thumbnail.
    """
    try:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            print("YOUTUBE_API_KEY not found. Skipping video search.")
            return None

        youtube = build('youtube', 'v3', developerKey=api_key)
        search_query = f"{query} tutorial how-to"

        request = youtube.search().list(
            q=search_query,
            part='snippet',
            maxResults=1,
            type='video',
            videoEmbeddable='true',
            relevanceLanguage='en'
        )
        response = request.execute()

        if response['items']:
            first_result = response['items'][0]
            video_id = first_result['id']['videoId']

            # --- NEW: Get the thumbnail URL ---
            thumbnail_url = first_result['snippet']['thumbnails']['high']['url']

            video_link = f"https://www.youtube.com/watch?v={video_id}" # Use standard link

            # Return a dictionary with both links
            return {
                "link": video_link,
                "thumbnail": thumbnail_url
            }
        else:
            return None

    except Exception as e:
        print(f"Error calling YouTube API: {e}")
        return None
