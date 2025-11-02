import streamlit as st
from rag_pipeline import (
    load_knowledge, build_or_load_index, retrieve_context, 
    get_rag_response, fetch_gdacs_alerts, intelligent_query_analyzer,
    get_youtube_video_link
)
import os
import nltk
import pandas as pd
from PIL import Image
from datetime import datetime, timezone

# --- Page Config (Must be first Streamlit command) ---
st.set_page_config(
    page_title="DisasterGPT",
    page_icon="🚨",
    layout="wide"
)

# --- NLTK Downloader (from rag_pipeline) ---
def download_nltk_data_once():
    packages_to_check = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'wordnet': 'corpora/wordnet',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng'
    }
    packages_to_download = []
    for pkg_name, pkg_path in packages_to_check.items():
        try:
            nltk.data.find(pkg_path)
        except LookupError:
            packages_to_download.append(pkg_name)
            
    if packages_to_download:
        print(f"Downloading missing NLTK packages: {', '.join(packages_to_download)}...")
        for pkg_name in packages_to_download:
            nltk.download(pkg_name, quiet=True)
        
# --- Helper function for "Time Ago" ---
def format_time_ago(dt):
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - dt
    seconds = delta.total_seconds()
    
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"[{minutes} min ago]" if minutes > 0 else "[Just now]"
    if seconds < 86400:
        return f"[{int(seconds // 3600)} hours ago]"
    else:
        return f"[{int(seconds // 86400)} days ago]"

# --- Caching ---
@st.cache_resource
def load_data_and_index():
    download_nltk_data_once()
    class MockST:
        def info(self, msg): pass
        def rerun(self): pass
    
    st.spinner("Loading knowledge base... (This happens once per session)")
    facts_df, news_df = load_knowledge(MockST()) 
    indices, knowledge_bases = build_or_load_index(facts_df, news_df)
    st.spinner("Knowledge base loaded!")
    return indices, knowledge_bases

@st.cache_data(ttl=600)
def get_latest_headlines():
    try:
        news_df = pd.read_json('classified_disaster_news.json')
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values(by='date', ascending=False)
        
        headlines_data = []
        top_10 = news_df[news_df['title'] != 'No Title'].head(10)
        
        for _, row in top_10.iterrows():
            headlines_data.append({
                "title": row['title'],
                "url": row['url'],
                "time_ago": format_time_ago(row['date'])
            })
        return headlines_data
    except Exception as e:
        print(f"Error loading headlines: {e}")
        return [{"title": "Latest news headlines will appear here once the scraper runs...", "url": "#", "time_ago": ""}]

# --- API Key Management ---
if 'GROQ_API_KEY' not in os.environ:
    try:
        os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
    except:
        st.error("GROQ_API_KEY not found. Please set it in your environment or Streamlit secrets.")
        st.stop()

# --- Initialize Session State (FIXES COLLAPSING BUG) ---
if 'gdacs_events' not in st.session_state:
    st.session_state['gdacs_events'] = None # Initialize as None
if 'rag_response' not in st.session_state:
    st.session_state['rag_response'] = None
if 'rag_context' not in st.session_state:
    st.session_state['rag_context'] = None
if 'indices_searched' not in st.session_state:
    st.session_state['indices_searched'] = ""
if 'video_info' not in st.session_state:
    st.session_state['video_info'] = None

# =============================================================================
# --- UI LAYOUT ---
# =============================================================================

# --- 1. HEADER (Logo + Title) ---
try:
    logo = Image.open("logo.png")
    col1, col2 = st.columns([0.1, 0.9]) 
    with col1:
        st.image(logo, width=100)
    with col2:
        st.title("DisasterGPT", anchor=False)
        st.markdown("Your AI assistant for disaster information and real-time news.")
except FileNotFoundError:
    st.title("🚨 DisasterGPT", anchor=False)
    st.markdown("Your AI assistant for disaster information and real-time news.")

st.divider()

# --- 2. SCROLLING NEWS MARQUEE (st.html FIX) ---
headlines = get_latest_headlines()

# Build the HTML for each clickable item
headline_html_elements = []
for item in headlines:
    html_str = f"""
        <span style="padding: 0 2rem;">
            <a href="{item['url']}" target="_blank" style="color: #FAFAFA; text-decoration: none; font-weight: 500;">
           {item['title']}
            </a>
            <span style="color: #9A9A9A; font-size: 0.9em; margin-left: 8px;">
           {item['time_ago']}
            </span>
        </span>
    """
    headline_html_elements.append(html_str)

# Join all items into one long string, separated by a divider
headline_string = " &nbsp; | &nbsp; ".join(headline_html_elements)

# Build the FINAL, complete HTML string with CSS animation (replaces <marquee>)
final_html = f"""
<style>
@keyframes ticker-scroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-100%); }}
}}
.ticker-wrap {{
    width: 100%;
    overflow: hidden;
    background-color: #262730;
    padding: 12px 0;
    border-radius: 10px;
    border: 1px solid #4A4A4A;
    margin-bottom: 20px;
}}
.ticker {{
    display: inline-block;
    white-space: nowrap;
    padding-left: 100%;
    animation: ticker-scroll 60s linear infinite;
    font-family: 'sans serif';
}}
</style>
<div class="ticker-wrap">
    <div class="ticker">
        <span style="padding: 0 1rem;"><strong>LATEST NEWS:</strong></span>
        {headline_string}
    </div>
</div>
"""
# Use st.html() to render the HTML and CSS
st.html(final_html)


# --- 3. MAIN CONTENT (Alerts Left, RAG Right) ---
col_alert, col_chat = st.columns([0.3, 0.7]) 

# --- Column 1: Live Alerts (GDACS) ---
with col_alert:
    st.subheader("🚨 Live Global Alerts")

    st.caption("Click the button to fetch the latest 10 global disaster alerts.")

    if st.button("Check for Latest Alerts"):
        with st.spinner("Fetching alerts..."):
            st.session_state['gdacs_events'] = fetch_gdacs_alerts() 
            if st.session_state['gdacs_events'] is None:
                st.error("Could not retrieve alerts.")
            elif len(st.session_state['gdacs_events']) == 0:
                st.success("No active global alerts found.")

    # Display alerts from session state (so they don't disappear)
    if st.session_state.get('gdacs_events'):
        if isinstance(st.session_state['gdacs_events'], list) and len(st.session_state['gdacs_events']) > 0:
            for event in st.session_state['gdacs_events']:

                props = {} # Initialize empty props dict

                # --- THIS IS THE WORKING "PARANOID" LOGIC ---
                try:
                    if isinstance(event, dict):
                        props = event.get('properties', {})
                    elif hasattr(event, 'properties') and event.properties is not None:
                        props = event.properties
                    elif isinstance(event, (list, tuple)) and len(event) > 3 and isinstance(event[3], dict):
                        props = event[3]
                    elif isinstance(event, dict) and 'alertlevel' in event:
                        props = event

                    if not props:
                        st.warning(f"Could not parse 'properties' from event: {str(event)}")
                        continue 
                except Exception as e:
                    st.error(f"Error parsing one event: {e}")
                    st.code(str(event)) 
                    continue 
                # --- END OF "PARANOID" LOGIC ---

                # Now, safely get all values from the 'props' DICTIONARY
                alert_level = props.get('alertlevel', 'Info').lower()
                event_type = props.get('eventtype', 'Event')
                title = props.get('name', 'No Title')
                country = props.get('country', 'N/A')
                fromdate = props.get('fromdate', 'N/A')
                todate = props.get('todate', 'N/A')
                link = props.get('url', {}).get('report', 'https.www.gdacs.org')

                with st.container(border=True):
                    if alert_level == 'red':
                        st.error(f"**{event_type} - {alert_level.upper()} ALERT**")
                    elif alert_level == 'orange':
                        st.warning(f"**{event_type} - {alert_level.upper()} ALERT**")
                    elif alert_level == 'green':
                        st.success(f"**{event_type} - {alert_level.upper()} ALERT**")
                    else:
                        st.info(f"**{event_type} - {alert_level.upper()} ALERT**")

                    st.markdown(f"**{title}**")
                    st.markdown(f"**Location:** {country}")
                    st.markdown(f"**Dates:** {fromdate} to {todate}")
                    st.link_button("View Details", link, help="View full report on GDACS website")

                st.write("") 

        elif st.session_state['gdacs_events'] == []:
             st.success("No active global alerts found.")


# --- Column 2: Ask a Question (RAG) ---
with col_chat:
    st.subheader("🤖 Ask a Question")
    
    try:
        with st.spinner("Loading knowledge base... (This only happens on first load)"):
            indices, knowledge_bases = load_data_and_index()

        with st.container(border=True):
            query = st.text_input("Ask a question:", 
                                  placeholder="e.g., What's the news on the earthquake in Myanmar and what should I do?",
                                  label_visibility="collapsed")

            if st.button("Get Answer"):
                st.session_state['rag_response'] = None 
                if query:
                    with st.spinner("Analyzing and searching..."):
                        search_plan = intelligent_query_analyzer(query)
                        search_steps = [f"'{step['sub_query']}' (in {step['index_to_search']})" for step in search_plan]
                        st.info(f"Search Plan: {', '.join(search_steps)}")

                        context, indices_searched = retrieve_context(search_plan, indices, knowledge_bases)
                        response = get_rag_response(query, context)
                        
                        st.session_state['rag_response'] = response
                        st.session_state['rag_context'] = context
                        st.session_state['indices_searched'] = indices_searched
                        # --- NEW: YouTube Video Search ---
                        st.session_state['video_info'] = None # Clear old video

                        if "facts" in indices_searched:
                            facts_query = ""
                            for step in search_plan:
                                if step.get("index_to_search") == "facts":
                                    facts_query = step.get("sub_query")
                                    break 

                            if facts_query:
                                # This now returns a dictionary: {"link": ..., "thumbnail": ...}
                                st.session_state['video_info'] = get_youtube_video_link(facts_query)
                        # --- END NEW: YouTube Video Search ---
                else:
                    st.warning("Please enter a question.")

        if st.session_state['rag_response']:
            with st.container(border=True):
                st.markdown(st.session_state['rag_response'])

                # --- NEW: Display the clickable thumbnail ---
                if st.session_state['video_info']:
                    video_data = st.session_state['video_info']
                    st.divider()
                    st.markdown("**Related Video Tutorial:**")
                    st.image(video_data['thumbnail'], use_container_width=True)
                    st.link_button("Watch on YouTube", video_data['link'])
                # --- END NEW ---

                with st.expander(f"Show retrieved context (from {st.session_state['indices_searched']} indices)"):
                    st.text(st.session_state['rag_context'])

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
        st.error("Could not load the RAG knowledge base. Please check the data files and restart the app.")