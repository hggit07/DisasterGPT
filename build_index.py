import nltk
import sys
from rag_pipeline import load_knowledge, build_or_load_index

# --- 1. Manual NLTK Download ---
def download_nltk_data():
    packages = ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'averaged_perceptron_tagger_eng']
    print("Checking NLTK packages...")
    for package in packages:
        try:
            if package in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{package}')
            elif package == 'averaged_perceptron_tagger_eng':
                nltk.data.find(f'taggers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
    print("NLTK check complete.")

download_nltk_data()

# --- 2. Mock Streamlit Object ---
class MockST:
    def info(self, msg):
        print(msg)
    
    def rerun(self):
        print("Mock rerun called.")

# --- 3. Run the Build Process ---
print("Loading knowledge base...")
facts_df, news_df = load_knowledge(MockST())

print("\nBuilding FAISS index... This will take 2-5 minutes. Please wait.")
build_or_load_index(facts_df, news_df)

print("\n--- Index build complete! 'knowledge.index' is saved. ---")
print("You can now safely run: streamlit run app.py")