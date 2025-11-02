import os
import requests
from newspaper import Article
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import json
from tqdm import tqdm

def run_scraper():
    # Config
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        print("Error: NEWS_API_KEY not found in environment variables.")
        return

    newsapi = NewsApiClient(api_key=api_key)

    keywords = ['wildfire', 'avalanche', 'blizzard', 'heatwave', 'earthquake', 'flood', 'hurricane', 'drought', 'tsunami', 'landslide', 'tornado', 'natural disaster']
    query = ' OR '.join(f'"{kw}"' for kw in keywords)
    from_date = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    language = 'en'
    page_size = 20
    target_article_count = 60
    page = 1
    max_pages = 5
    new_articles = []
    existing_urls = set()

    output_file = 'disaster_news.json'

    # Load existing URLs
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_articles = json.load(f)
                existing_urls = {article.get('url') for article in existing_articles if 'url' in article}
            except json.JSONDecodeError:
                existing_articles = []
    else:
        existing_articles = []

    def is_blocked_url(url):
        return "consent.yahoo.com" in url

    def extract_full_text(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text, article.authors
        except Exception:
            return None, None

    # Scraping loop
    print("Starting news scraping process...")
    while len(new_articles) < target_article_count and page <= max_pages:
        try:
            response = newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by='relevancy',
                page_size=page_size,
                page=page
            )
        except Exception as e:
            print(f"Error calling NewsAPI: {e}")
            break

        articles = response.get('articles', [])
        if not articles:
            print("No more articles found.")
            break

        for article in tqdm(articles, desc=f"Processing page {page}"):
            url = article['url']
            if url in existing_urls or is_blocked_url(url):
                continue
            existing_urls.add(url)

            title = article['title']
            published_at = article['publishedAt']

            if not any(kw.lower() in title.lower() for kw in keywords):
                continue

            full_text, authors = extract_full_text(url)

            if full_text and len(full_text.split()) > 100:
                article_data = {
                    'title': title,
                    'author': authors[0] if authors else None,
                    'date': published_at,
                    'url': url,
                    'content': full_text
                }
                new_articles.append(article_data)

            if len(new_articles) >= target_article_count:
                break

        if len(articles) < page_size:
            break

        page += 1

    # Save results
    all_articles = existing_articles + new_articles

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=4)

    print(f"Appended {len(new_articles)} new articles. Total saved: {len(all_articles)} to {output_file}")

if __name__ == "__main__":
    run_scraper()