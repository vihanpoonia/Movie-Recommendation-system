import streamlit as st
from recommend import recommend, new_df

# Streamlit app title
st.title("Movie Recommendation System")

# Dropdown for movie selection
movies_list = new_df['title'].values
selected_movie = st.selectbox("Choose a movie", movies_list)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.write("**Recommended Movies:**")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("No recommendations found. Please try another movie.")
