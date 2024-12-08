import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
import spacy

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

aspect_categories = {
    "food_quality": [
        "food", "taste", "flavor", "dish", "meal", "delicious", "yummy", "fresh", "savor", "tasty", "flavorful", 
        "spicy", "bland", "sweet", "salty", "bitter", "sour", "crispy", "juicy", "hearty", "appetizing", "gourmet",
        "rich", "succulent", "overcooked", "undercooked", "raw", "burnt", "mouthwatering", "greasy"
    ],
    "service": [
        "service", "staff", "waiter", "waitress", "attentive", "friendly", "slow", "helpful", "fast", "polite", 
        "rude", "unprofessional", "professional", "efficient", "courteous", "wait time", "smiling", "assistance", 
        "knowledgeable", "approachable", "accommodating", "disappointing", "welcoming", "supportive", "prompt", 
        "unfriendly", "patient", "kind", "disinterested", "friendly"
    ],
    "ambiance": [
        "ambiance", "atmosphere", "vibe", "decor", "music", "lighting", "environment", "comfort", "cozy", "warm", 
        "inviting", "chilly", "modern", "classic", "elegant", "rustic", "lively", "calm", "romantic", "elegant", 
        "charming", "relaxed", "chilly", "dark", "bright", "stylish", "minimal", "warmth", "cool", "friendly", 
        "intimate", "space", "noise", "crowded", "quiet", "peaceful", "laid-back", "pleasant", "sophisticated"
    ],
    "cleanliness": [
        "clean", "dirty", "hygiene", "neat", "messy", "orderly", "spotless", "immaculate", "tidy", "unsanitary", 
        "disorganized", "cleanliness", "filthy", "sanitized", "disgusting", "stinky", "smelly", "fresh", "pristine", 
        "scruffy", "dusty", "unhygienic", "cluttered", "messy", "sterile", "decent", "polished"
    ],
    "price": [
        "price", "cost", "expensive", "cheap", "affordable", "value", "inexpensive", "overpriced", "reasonable", 
        "pricy", "value-for-money", "budget", "luxurious", "high-end", "expensive", "costly", "discounted", 
        "affordability", "bargain", "premium", "value", "reasonable", "low-cost", "high-price", "worth it", 
        "cheap", "underpriced", "overcharged"
    ]
}

def extract_aspects(text):
    """Extract noun chunks and match them with category keywords."""
    doc = nlp(text)
    aspects = []
    for chunk in doc.noun_chunks:
        for category, keywords in aspect_categories.items():
            if any(keyword.lower() in chunk.text.lower() for keyword in keywords):
                aspects.append((category, chunk.text))
    return aspects

def analyze_text_sentiment(text): 
    return sia.polarity_scores(text)['compound']

def get_word_sentiments(text):
    sentiments = []
    words = text.split()
    for word in words:
        sentiment_score = sia.polarity_scores(word)['compound']
        if sentiment_score != 0: 
            sentiments.append((word, sentiment_score))
    return sentiments

def process_review(row):
    review_text = row['body_clean']
    row['sentiment_score'] = analyze_text_sentiment(review_text)
    aspects = extract_aspects(review_text)
    word_sentiments = get_word_sentiments(review_text)
    
    aspect_sentiments = {}
    for category, aspect in aspects:
        category_sentiment = 0
        for word, sentiment in word_sentiments:
            if any(keyword.lower() in word.lower() for keyword in aspect_categories[category]):
                category_sentiment += sentiment
        if category_sentiment != 0:
            aspect_sentiments[category] = category_sentiment
    row['aspect_sentiments'] = aspect_sentiments
    return row

def analyze_sentiment(df):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_review, [row for _, row in df.iterrows()]))
    return pd.DataFrame(results)