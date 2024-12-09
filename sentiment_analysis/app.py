import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_analysis.code import analyze_sentiment

def run_sentiment_analysis(df):
    st.header("Sentiment Analysis")

    user_input = st.text_input("Enter the restaurant you want to look at:")
    
    if st.button("Analyze"):
        df_filtered = df[df['name'].str.contains(user_input, case=False, na=False)]

        if df_filtered.empty:
            st.warning(f"No data found for this restaurant: {user_input}")
        else:
            result_df = analyze_sentiment(df_filtered)

            total_reviews = len(df_filtered)
            avg_rating = round(df_filtered["stars_y"].mean(),2)
            overall_avg = result_df["sentiment_score"].mean()
            aspect_averages = (
                result_df[["Food Quality", "Service", "Ambiance", "Cleanliness", "Price"]]
                .mean()
                .reset_index()
                .rename(columns={"index": "Category", 0: "Average Sentiment"})
            ) 

            st.write(f"**Total Reviews Analyzed:** {total_reviews}")
            st.write(f"**Average Rating of Your Search:** {avg_rating}")
            st.write(f"**Average Sentiment Score:** {overall_avg:.2f}")
            st.write("**Sentiment Analysis Table Per Category:**")
            st.dataframe(aspect_averages)


            tab1, tab2 = st.tabs(["Ratings by Category", "Map View"])
            
            with tab1:
                st.header("Sentiment Radar")
                
                fig = px.line_polar(
                    aspect_averages,
                    r="Average Sentiment",
                    theta="Category",
                    line_close=True
                )
                fig.update_traces(fill='toself', line_color='blue')
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
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
                        zoom=12,
                        height=400
                    )
                    fig.update_layout(
                        mapbox_style="carto-positron",
                        margin={"r": 0, "t": 0, "l": 0, "b": 0}
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("No location data available to display the map.")
            
            st.write("Rrocessed Dataset:")
            st.dataframe(result_df)