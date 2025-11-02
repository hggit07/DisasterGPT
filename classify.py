import json
import csv
from sentence_transformers import SentenceTransformer, util

def run_classification():
    print("Starting classification process...")
    keywords = ['wildfire', 'avalanche', 'blizzard', 'heatwave', 'earthquake', 'flood', 'hurricane', 'drought', 'tsunami', 'landslide', 'tornado', 'volcano']

    input_file = 'disaster_news.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Run scrape.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not read '{input_file}'. File might be empty or corrupt.")
        return
    
    if not articles:
        print("No articles to classify.")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

    classified_rows = []

    for article in articles:
        text_to_classify = article.get('content', article.get('title', ''))
        if not text_to_classify:
            continue
            
        text_embedding = model.encode(text_to_classify, convert_to_tensor=True)

        cosine_scores = util.cos_sim(text_embedding, keyword_embeddings)[0]
        best_index = cosine_scores.argmax().item()
        best_score = cosine_scores[best_index].item()

        if best_score >= 0.3:  # Threshold
            best_keyword = keywords[best_index]
        else:
            best_keyword = 'no_category'

        classified_rows.append({
            'title': article.get('title'),
            'author': article.get('author'),
            'date': article.get('date'),
            'url': article.get('url'),
            'matched_keyword': best_keyword,
            'similarity_score': round(best_score, 4),
            'content': article.get('content')
        })

    csv_file = 'classified_disaster_news.csv'
    json_file = 'classified_disaster_news.json'

    # Save classified results
    if classified_rows:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=classified_rows[0].keys())
            writer.writeheader()
            writer.writerows(classified_rows)
        
        # --- Save to JSON ---
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(classified_rows, f, ensure_ascii=False, indent=4)
        
        print(f"Classified and saved {len(classified_rows)} articles to '{csv_file}' and '{json_file}'")
    else:
        print("No articles were classified.")

if __name__ == "__main__":
    run_classification()