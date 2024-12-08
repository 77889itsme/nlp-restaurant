import streamlit as st
import pandas as pd
import os
import requests
import pickle
from sentiment_analysis.app import run_sentiment_analysis
from recommendation.app import run_recommendation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++ load data via Google Drive ++++++++++++++++ #
@st.cache_data
def download_data_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    return destination

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
os.makedirs(os.path.dirname(file_path), exist_ok=True)
file_path_downloaded = download_data_from_google_drive(file_id, file_path)

if os.path.exists(file_path_downloaded):
    st.write(f"File is ready at: {file_path_downloaded}")
    df = read_pickle(file_path_downloaded)
    if df is not None:
        st.write("Data successfully loaded!")
else:
    st.error("Failed to download the data file.")

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "homepage.png")

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