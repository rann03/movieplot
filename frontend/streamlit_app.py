import streamlit as st
from dotenv import load_dotenv
import requests
import os

load_dotenv()

st.set_page_config(page_title="Movie Recommender Chatbox", page_icon="🎬")
OMDB_API_KEY = os.getenv("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")

st.title("🎬 Movie Recommender Chatbox")
st.write("Give me a movie plot and I'll suggest some 🎥 titles!")

model = st.selectbox(
    "🧠 Choose your movie expert model:",
    options=[
        "qwen/qwen3-4b:free",
        "meta-llama/llama-4-maverick:free",
        "deepseek/deepseek-chat:free",
    ],
    index=0
)

# --- Plot Input ---
plot = st.text_area("📝 Enter a movie plot")
recommend_clicked = st.button("🚀 Recommend")


# --- Utilities ---
def get_movie_data(title: str) -> dict:
    """Queries OMDb for movie metadata and poster."""
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        return data if data.get("Response") == "True" else {}
    except:
        return {}


# --- Recommendation Logic ---
if recommend_clicked:
    with st.spinner("🧠 Talking to the movie expert..."):
        try:
            response = requests.post(
                "http://backend:8000/recommend",
                json={"plot": plot, "model": model}
            )
            response.raise_for_status()
            results = response.json()["suggestions"]
            st.success("✅ Recommendations retrieved!")

            for movie in results:
                score_percent = int(movie["score"] * 100)

                omdb_data = get_movie_data(movie["title"])
                is_real = bool(omdb_data)
                poster_url = omdb_data.get("Poster") if is_real and omdb_data["Poster"] != "N/A" else "https://via.placeholder.com/150x220?text=No+Image"
                imdb_url = f"https://www.imdb.com/title/{omdb_data['imdbID']}" if is_real and "imdbID" in omdb_data else "#"
                description = movie["description"]

                # Determine color label
                if score_percent >= 80:
                    confidence_color = "🟢 High"
                elif score_percent >= 60:
                    confidence_color = "🟠 Medium"
                else:
                    confidence_color = "🔴 Low"

                # --- Card container ---
                with st.container():
                    st.markdown("---")  # optional separator between cards
                    cols = st.columns([1, 3])  # Poster | Confidence + Info

                    with cols[0]:
                        st.image(poster_url, width=120)
                        if is_real:
                            st.link_button("View on IMDb", imdb_url)

                    with cols[1]:
                        st.write(f"**Confidence:** {confidence_color} ({score_percent}%)")
                        st.subheader(f"🎬 {movie['title']}")
                        if not is_real:
                            st.warning("❗ This movie doesn't appear to exist in OMDb.")
                        st.write(f"**Synopsis:** {description}")


        except Exception as e:
            st.error(f"❌ Failed to get recommendations: {e}")
