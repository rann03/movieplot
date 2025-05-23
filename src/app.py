import streamlit as st
from dotenv import load_dotenv
import requests
import os
import socket

load_dotenv()

st.set_page_config(page_title="Movie Recommender Chatbox", page_icon="🎬")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
API_KEY = "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"

st.title("🎬 Movie Recommender Chatbox")
st.write("Give me a movie plot and I'll suggest some 🎥 titles!")

# Hardcoded model
model = "deepseek/deepseek-chat:free"

# --- Plot Input ---
plot = st.text_area("📝 Enter a movie plot")
recommend_clicked = st.button("🚀 Recommend")

# Network Diagnostic Function
def check_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        st.error(f"Port check error: {e}")
        return False

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

# Alternative Recommendation Logic
def mock_recommendations(plot):
    return [
        {
            "title": "Inception",
            "score": 0.85,
            "description": "A mind-bending thriller about dream infiltration"
        },
        {
            "title": "Interstellar",
            "score": 0.75,
            "description": "A space exploration epic about saving humanity"
        }
    ]

# --- Recommendation Logic ---
if recommend_clicked:
    with st.spinner("🧠 Talking to the movie expert..."):
        try:
            # First, check if local backend is running
            if not check_port('localhost', 8000):
                st.warning("Backend service not detected. Using mock recommendations.")
                results = mock_recommendations(plot)
            else:
                # Try to connect to backend
                response = requests.post(
                    "http://localhost:8000/recommend",
                    json={"plot": plot, "model": model, "api_key": API_KEY}
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
            # Add a mock recommendation fallback
            st.warning("Using mock recommendations due to connection error.")
            results = mock_recommendations(plot)