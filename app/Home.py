import streamlit as st
import pandas as pd
import os
import requests
from sentiment_analysis.app import run_sentiment_analysis
from recommendation.app import run_recommendation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++ load data via Google Drive ++++++++++++++++ #
@st.cache_data
def download_data(file_url, file_path):
    if not os.path.exists(file_path):
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.write(f"Downloaded {file_path}")
    return file_path

@st.cache_data
def read_pickle(file_path):
    import pickle
    dat = pickle.load(open(file_path, "rb"))
    df = pd.DataFrame(dat)
    return df

# df = read_pickle("data/dat.pk")

file_url = "https://drive.google.com/uc?id=1zgIA4aMUuT6_t9cWD90BOhVCyXqCHzaZ"
file_path = "data/dat.pk"

file_path_downloaded = download_data(file_url, file_path)

if os.path.exists(file_path_downloaded):
    st.write(f"File is ready at: {file_path_downloaded}")
    df = read_pickle(file_path_downloaded) 
    st.write("Data successfully loaded!")
else:
    st.error("Failed to download the data file.")


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
    st.image("homepage.png", use_column_width=True)