import streamlit as st
import pandas as pd
import plotly.express as px
from recommendation.code import recommendation


def run_recommendation(df):
    st.header("Restaurant Recommendation System")
    
    user_input = st.text_area("Enter your review:")
    user_city = st.text_input("Enter your city:", value="Santa Barbara")
    st.caption("Currently, our app is limited to searches within Santa Barbara.")
    
    if st.button("Find Restaurants"):
        if user_input and user_city:
            recommendations = recommendation(df, user_input, user_city, top_n=5)
            if recommendations:
                tab1, tab2 = st.tabs(["Recommendations", "Map"])

                with tab1: 
                    st.header("Top Recommendations:")
                    for idx, rec in enumerate(recommendations, start=1):
                        st.subheader(f"{idx}. {rec['restaurant']}")
                        st.write(f"City: {rec['city']}")
                        st.write(f"Address: {rec['address']}")
                        st.write(f"Stars: {rec['stars']}")
                        st.write(f"Cuisine: {rec['cuisine']}")
                        st.write(f"Matched Review: {rec['matched_review']}")
                        st.write(f"Score: {rec['score']:.4f}")
                        st.write("-" * 50)

                with tab2:
                    st.header("Restaurant Locations:")
                    map_data = pd.DataFrame(
                        [{  "Restaurant": rec["restaurant"],
                            "Latitude": rec["latitude"],
                            "Longitude": rec["longitude"],}
                            for rec in recommendations])
                    fig = px.scatter_mapbox(
                        map_data,
                        lat="Latitude",
                        lon="Longitude",
                        text="Restaurant",
                        hover_name="Restaurant",
                        zoom=10,
                        height = 600,
                        mapbox_style="carto-positron",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.write("No recommendations found.")
        else:
            st.warning("Please provide both review and city to get recommendations.")
