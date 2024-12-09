import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
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
        "spicy", "bland", "sweet", "salty", "bitter", "sour", "crispy", "juicy", "hearty", "appetizing", "gourmet"
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "attentive", "friendly", "slow", "helpful", "fast", "polite", 
        "rude", "unprofessional", "professional", "efficient", "courteous", "wait time", "smiling", "assistance"
    ],
    "Ambiance": [
        "ambiance", "atmosphere", "vibe", "decor", "music", "lighting", "environment", "comfort", "cozy", "warm", 
        "inviting", "chilly", "modern", "classic", "elegant", "rustic", "lively", "calm", "romantic", "elegant"
    ],
    "Cleanliness": [
        "clean", "dirty", "hygiene", "neat", "messy", "orderly", "spotless", "immaculate", "tidy", "unsanitary", 
        "disorganized", "cleanliness", "filthy", "sanitized", "disgusting", "stinky", "smelly", "fresh", "pristine"
    ],
    "Price": [
        "price", "cost", "expensive", "cheap", "affordable", "value", "inexpensive", "overpriced", "reasonable", 
        "pricy", "value-for-money", "budget", "luxurious", "high-end", "expensive", "costly", "discounted"
    ]
}

def extract_aspect_score(text, keywords):
    sentences = sent_tokenize(text)
    relevant_text = " ".join([sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)])
    if relevant_text.strip():
        return sia.polarity_scores(relevant_text)["compound"]
    return None

def process_review(row):
    row["cleaned_text"] = preprocess_text(row["text"])
    row["sentiment_score"] = sia.polarity_scores(row["cleaned_text"])["compound"]
    
    for aspect, keywords in aspect_categories.items():
        row[f"{aspect}"] = extract_aspect_score(row["cleaned_text"], keywords)
    
    return row

def analyze_sentiment(df): 
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_review, [row for _, row in df.iterrows()]))

    result_df = pd.DataFrame(results)
    return result_df[["name", "longitude", "latitude", "stars_y","sentiment_score"] + [f"{aspect}" for aspect in aspect_categories.keys()]]