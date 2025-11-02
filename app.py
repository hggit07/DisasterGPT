import streamlit as st
from rag_pipeline import load_knowledge, build_or_load_index, retrieve_context, get_rag_response
import os
import nltk
from rag_pipeline import (
    load_knowledge, build_or_load_index, retrieve_context, 
    get_rag_response, fetch_gdacs_alerts, intelligent_query_analyzer
)

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
    # print("Checking NLTK packages...")
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
            # print(f"NLTK package '{package}' not found. Downloading...")
            nltk.download(package, quiet=True)
            # print(f"Downloaded '{package}'.")
    print("NLTK check complete.")

# Caching
@st.cache_resource
def load_data_and_index():
    # download_nltk_data_once() 
    # st.info("Loading knowledge base... (This happens once per session)")
    
    facts_df, news_df = load_knowledge(st) 
    indices, knowledge_bases = build_or_load_index(facts_df, news_df)
    
    # st.success("Knowledge base loaded!")

    return indices, knowledge_bases

st.title("🚨 DisasterGPT")
st.markdown("Your AI assistant for disaster information and real-time news.")
tab1, tab2 = st.tabs(["🤖 Ask a Question (RAG)", "🚨 Live Alerts (GDACS)"])

with tab1:
    try:
        indices, knowledge_bases = load_data_and_index()

        with st.container(border=True):
            query = st.text_input("Ask a question:", 
                                  placeholder="e.g., What's the news on the earthquake in Myanmar and what should I do?")

            if st.button("Get Answer"):
                if query:
                    with st.spinner("Analyzing and searching..."):
                        
                        search_plan = intelligent_query_analyzer(query)
                        search_steps = [f"'{step['sub_query']}' (in {step['index_to_search']})" for step in search_plan]
                        st.info(f"Search Plan: {', '.join(search_steps)}")
                        context, indices_searched = retrieve_context(search_plan, indices, knowledge_bases)
                        
                        response = get_rag_response(query, context)

                        with st.container(border=True):
                            st.markdown(response)
                            with st.expander(f"Show retrieved context (from {indices_searched} indices)"):
                                st.text(context)
                else:
                    st.warning("Please enter a question.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
        st.error("Could not load the knowledge base. Please check the data files and restart the app.")

with tab2:
    st.subheader("Live Global Disaster Alerts")
    st.markdown("Powered by the Global Disaster Alert and Coordination System (GDACS). Alerts are cached for 10 minutes.")

    if st.button("Check for Latest Alerts"):
        with st.spinner("Fetching alerts..."):

            # 'events' is the list of Feature objects, or None
            events = fetch_gdacs_alerts() 

            if events and isinstance(events, list):
                for event in events:

                    props = {} # Initialize empty props dict

                    # --- This is the robust "paranoid" logic that worked ---
                    try:
                        # 1. Try to access it like a dictionary
                        if isinstance(event, dict):
                            props = event.get('properties', {})

                        # 2. Try to access it like an object
                        elif hasattr(event, 'properties') and event.properties is not None:
                            props = event.properties

                        # 3. Try to access it like a tuple
                        elif isinstance(event, (list, tuple)) and len(event) > 3 and isinstance(event[3], dict):
                            props = event[3]

                        # 4. Check if 'event' IS the properties dict
                        elif isinstance(event, dict) and 'alertlevel' in event:
                            props = event

                        if not props:
                            st.warning(f"Could not parse 'properties' from event: {str(event)}")
                            continue 

                    except Exception as e:
                        st.error(f"Error parsing one event: {e}")
                        st.code(str(event)) 
                        continue 
                    # --- End of robust logic ---

                    # Now, safely get all values from the 'props' DICTIONARY
                    alert_level = props.get('alertlevel', 'Info').lower()
                    event_type = props.get('eventtype', 'Event')
                    title = props.get('name', 'No Title')
                    country = props.get('country', 'N/A')
                    fromdate = props.get('fromdate', 'N/A')
                    todate = props.get('todate', 'N/A')
                    description = props.get('description', 'No description available.')

                    # --- FIX 1: Get the 'report' link ---
                    link = props.get('url', {}).get('report', 'https://www.gdacs.org')

                    # --- FIX 2: Add 'green' color ---
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
                    st.write(description)
                    st.link_button("View Details on GDACS", link)
                    st.divider() # Revert to the simple divider

            else:
                st.error("Could not retrieve alerts at this time. Please try again later.")