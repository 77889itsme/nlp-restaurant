import streamlit as st
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
                st.write("Top Recommendations:")
                for idx, rec in recommendations:
                    st.subheader(f"{idx}. {rec['restaurant']}")
                    st.write(f"City: {rec['city']}")
                    st.write(f"Address: {rec['address']}")
                    st.write(f"Stars: {rec['stars']}")
                    st.write(f"Cuisine: {rec['cuisine']}")
                    st.write(f"Matched Review: {rec['matched_review']}")
                    st.write(f"Score: {rec['score']:.4f}")
                    st.write("-" * 50)
            else:
                st.write("No recommendations found.")
        else:
            st.warning("Please provide both review and city to get recommendations.")
