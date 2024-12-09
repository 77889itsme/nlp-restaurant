import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

# Infer cuisine based on keywords in review text
def infer_cuisine(text):
    cuisines = {
        'italian': ['pasta', 'pizza', 'risotto'],
        'chinese': ['noodles', 'dumplings', 'dim sum'],
        'japanese': ['sushi', 'ramen', 'tempura'],
        'mexican': ['tacos', 'burrito', 'quesadilla'],
        'indian': ['curry', 'naan', 'masala'],
        'american': ['burger', 'steak', 'fries']
    }
    for cuisine, keywords in cuisines.items():
        if any(keyword in text for keyword in keywords):
            return cuisine
    return 'other'


# wrap up into a function
def build_tfidf_matrix(df):
    df['cuisine'] = df['body_clean'].apply(infer_cuisine)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['body_clean'])
    return vectorizer, tfidf_matrix


# Recommend restaurants based on multi-factor scoring
def recommend_restaurants(user_input, user_city, vectorizer, tfidf_matrix, df, top_n=5):
    # Preprocess user input and infer cuisine
    user_input_cleaned = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input_cleaned])
    user_cuisine = infer_cuisine(user_input)
    
    # Compute cosine similarity for appetite (review match)
    appetite_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Calculate scores for each restaurant
    scores = []
    for idx, row in df.iterrows():
        location_score = 1.0 if row['city'].lower() == user_city.lower() else 0.0
        cuisine_score = 1.0 if row['cuisine'] == user_cuisine else 0.0
        stars_score = row['stars_y'] / 5.0  # Normalize stars (assume max = 5)
        
        # Adjusted weighted aggregate score
        weighted_score = (
            0.55 * location_score +  # City weight
            0.25 * cuisine_score +  # Cuisine weight
            0.2 * stars_score       # Stars weight
        )
        scores.append((idx, weighted_score, appetite_similarities[idx]))
    
    # Sort by weighted score (and appetite similarity as a tiebreaker)
    scores = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)
    
    # Get top recommendations
    recommendations = []
    for idx, score, sim in scores[:top_n]:
        restaurant = df.iloc[idx]['name']
        review = df.iloc[idx]['text']
        city = df.iloc[idx]['city']
        location = df.iloc[idx]['address']  # Get the location
        stars = df.iloc[idx]['stars_y']
        cuisine = df.iloc[idx]['cuisine']
        recommendations.append({
            'restaurant': restaurant,
            'city': city,
            'address': location,  # Include location
            'stars': stars,
            'cuisine': cuisine,
            'matched_review': review,
            'score': score
        })
    
    return recommendations

def recommendation(df, user_input, user_city, top_n=5):
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)

    recommendations = recommend_restaurants(user_input, user_city, vectorizer, tfidf_matrix, df, top_n)
    return recommendations