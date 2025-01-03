import pandas as pd
import re
import nltk
from nltk import pos_tag
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt_tab')
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

aspect_categories = {
    "Food Quality": [
        "food", "taste", "flavor", "dish", "meal", "delicious", "yummy", "fresh", "savor", "tasty", "flavorful", 
        "spicy", "bland", "sweet", "salty", "bitter", "sour", "crispy", "juicy", "hearty", "appetizing", "gourmet",
        "rich", "succulent", "overcooked", "undercooked", "raw", "burnt", "mouthwatering", "greasy"
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "attentive", "friendly", "slow", "helpful", "fast", "polite", 
        "rude", "unprofessional", "professional", "efficient", "courteous", "wait time", "smiling", "assistance", 
        "knowledgeable", "approachable", "accommodating", "disappointing", "welcoming", "supportive", "prompt", 
        "unfriendly", "patient", "kind", "disinterested", "friendly"
    ],
    "Ambiance": [
        "ambiance", "atmosphere", "vibe", "decor", "music", "lighting", "environment", "comfort", "cozy", "warm", 
        "inviting", "chilly", "modern", "classic", "elegant", "rustic", "lively", "calm", "romantic", "elegant", 
        "charming", "relaxed", "chilly", "dark", "bright", "stylish", "minimal", "warmth", "cool", "friendly", 
        "intimate", "space", "noise", "crowded", "quiet", "peaceful", "laid-back", "pleasant", "sophisticated"
    ],
    "Cleanliness": [
        "clean", "dirty", "hygiene", "neat", "messy", "orderly", "spotless", "immaculate", "tidy", "unsanitary", 
        "disorganized", "cleanliness", "filthy", "sanitized", "disgusting", "stinky", "smelly", "fresh", "pristine", 
        "scruffy", "dusty", "unhygienic", "cluttered", "messy", "sterile", "decent", "polished"
    ],
    "Price": [
        "price", "cost", "expensive", "cheap", "affordable", "value", "inexpensive", "overpriced", "reasonable", 
        "pricy", "value-for-money", "budget", "luxurious", "high-end", "expensive", "costly", "discounted", 
        "affordability", "bargain", "premium", "value", "reasonable", "low-cost", "high-price", "worth it", 
        "cheap", "underpriced", "overcharged"
    ]
}

def extract_aspects(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged, binary=True) 

    aspects = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity = " ".join(c[0] for c in chunk.leaves())  # Get the words in the entity
        else:
            entity = chunk[0]

        for category, keywords in aspect_categories.items():
            if any(keyword in entity.lower() for keyword in keywords):
                aspects.append((category, entity))
    
    return aspects

def analyze_text_sentiment(text): 
    return sia.polarity_scores(text)['compound']

def calculate_aspect_sentiments(aspects, word_sentiments):
    aspect_sentiments = {}
    for category, aspect in aspects:
        category_sentiment = sum(
            sentiment for word, sentiment in word_sentiments
            if any(keyword in word.lower() for keyword in aspect_categories[category])
        )
        aspect_sentiments[category] = category_sentiment
    return aspect_sentiments

def get_word_sentiments(text):
    sentiments = []
    words = text.split()
    for word in words:
        sentiment_score = sia.polarity_scores(word)['compound']
        if sentiment_score != 0: 
            sentiments.append((word, sentiment_score))
    return sentiments


"""
def process_review(row):
    review_text = row['text']
    preprocessed_text = preprocess_text(review_text)
    row['sentiment_score'] = analyze_text_sentiment(preprocessed_text)
    aspects = extract_aspects(preprocessed_text)
    word_sentiments = get_word_sentiments(preprocessed_text)
    
    aspect_sentiments = {}
    for category, aspect in aspects:
        category_sentiment = 0
        for word, sentiment in word_sentiments:
            if any(keyword.lower() in word.lower() for keyword in aspect_categories[category]):
                category_sentiment += sentiment
        aspect_sentiments[category] = category_sentiment
    row['aspect_sentiments'] = aspect_sentiments
    return row
"""

def process_review(row):
    review_text = str(row.get("text", ""))
    preprocessed_text = preprocess_text(review_text)
    sentiment_score = analyze_text_sentiment(preprocessed_text)
    aspects = extract_aspects(preprocessed_text)
    word_sentiments = [(word, analyze_text_sentiment(word)) for word in preprocessed_text.split()]
    aspect_sentiments = calculate_aspect_sentiments(aspects, word_sentiments)
    return {
        "sentiment_score": sentiment_score,
        "aspect_sentiments": aspect_sentiments,
    }

def analyze_sentiment(df):
    df = df.copy()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_review, [row for _, row in df.iterrows()]))
    return pd.DataFrame(results)