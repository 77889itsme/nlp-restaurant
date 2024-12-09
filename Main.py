import streamlit as st
import pandas as pd
from sentiment_analysis.app import run_sentiment_analysis
from recommendation.app import run_recommendation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
@st.cache_data
def read_pickle(file_path):
    import pickle
    dat = pickle.load(open(file_path, "rb"))
    df = pd.DataFrame(dat)
    return df
# df = read_pickle("data/dat.pk")

df = pd.read_csv("data/dat.csv")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++ below are streamlit code snippets +++++++++++  #
st.title("Taste & Text")
st.sidebar.title("Navigation")

module = st.sidebar.radio("Go to", ["🏠 Home", "📊 Sentiment Analysis", "🍴 Recommendation"])


if module == "📊 Sentiment Analysis":
    run_sentiment_analysis(df) 
elif module == "🍴 Recommendation":
    run_recommendation(df) 
else:
    st.header("NLP Analysis of Restaurant Reviews in California")

    st.write("Welcome to our **interactive app**, designed as part of our group project!" )
    st.write("Explore restaurant reviews in California with powerful sentiment analysis and recommendation tools we create.")
    st.caption("This is a prototype, and the database is currently limited to Santa Barbara to ensure smooth execution.")
    st.image("images/homepage.png", use_container_width=True)