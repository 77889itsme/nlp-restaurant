import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_analysis.code import analyze_sentiment

@st.cache_data
def cached_analyze_sentiment(df):
    return analyze_sentiment(df)

def run_sentiment_analysis(df):
    st.header("Sentiment Analysis")

    user_input = st.text_input("Enter the restaurant you want to look at:")
    
    if st.button("Analyze"):
        df_filtered = df[df['name'] == user_input]

        if df_filtered.empty:
            st.warning(f"No data found for this restaurant: {user_input}")
        else:
            df_sen = cached_analyze_sentiment(df_filtered)

            total_reviews = len(df_filtered)
            avg_rating = round(df_filtered["stars_y"].mean(),2)
            aspect_sentiment_summary = (
                df_sen['aspect_sentiments']
                .apply(lambda x: pd.Series(x))
                .mean()
                .reset_index()
                .rename(columns={0: 'Average Sentiment', 'index': 'Category'})
            )

            st.write(f"**Total Reviews Analyzed:** {total_reviews}")
            st.write(f"**Average Rating of {user_input}:** {avg_rating}")
            st.write("**Sentiment Analysis Table Per Category:**")
            st.dataframe(aspect_sentiment_summary)

            tab1, tab2 = st.tabs(["Ratings by Category", "Map View"])
            
            with tab1:
                st.header("Sentiment Radar")
                
                fig = px.line_polar(
                    aspect_sentiment_summary,
                    r="Average Sentiment",
                    theta="Category",
                    line_close=True
                )
                fig.update_traces(fill='toself', line_color='blue')
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                )
                st.plotly_chart(fig)

            with tab2:
                st.header("Restaurant Map")
                
                if "latitude" in df_filtered.columns and "longitude" in df_filtered.columns:
                    fig = px.scatter_mapbox(
                        df_filtered,
                        lat="latitude",
                        lon="longitude",
                        hover_name="name",
                        zoom=5.5,
                        height=600
                    )
                    fig.update_layout(
                        mapbox_style="carto-positron",
                        margin={"r": 0, "t": 0, "l": 0, "b": 0}
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("No location data available to display the map.")