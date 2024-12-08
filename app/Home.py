import streamlit as st
import pandas as pd
import os
import gdown
import pickle
from sentiment_analysis.app import run_sentiment_analysis
from recommendation.app import run_recommendation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++ load data via Google Drive ++++++++++++++++ #
@st.cache_data
def download_data_gdrive(file_id, file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    return file_path

@st.cache_data
def read_pickle(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found!")

        # 加载文件
        with open(file_path, "rb") as f:
            dat = pickle.load(f)
        return pd.DataFrame(dat)
    except Exception as e:
        st.error(f"Failed to load pickle file: {e}")
        return None

# df = read_pickle("data/dat.pk")

file_id = "1zgIA4aMUuT6_t9cWD90BOhVCyXqCHzaZ"
file_path = "data/dat.pk"

file_path_downloaded = download_data_gdrive(file_id, file_path)

if os.path.exists(file_path_downloaded):
    st.write(f"File is ready at: {file_path_downloaded}")
    df = read_pickle(file_path_downloaded)
    if df is not None:
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