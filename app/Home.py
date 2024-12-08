import streamlit as st
import pandas as pd
import os
from sentiment_analysis.app import run_sentiment_analysis
from recommendation.app import run_recommendation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++ load data  +++++++++++++++++++++++++ #
@st.cache_data
def read_pickle(file_path):
    return pd.read_pickle(file_path)

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "data/dat_compressed.pkl")
image_path = os.path.join(current_dir, "homepage.png")

df = read_pickle(data_path)

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
    st.header("Analysis of Restaurant Reviews in California")
    st.write("Welcome to our group project's interactive app! Select a module from the sidebar.")
    st.image(image_path, use_container_width=True)