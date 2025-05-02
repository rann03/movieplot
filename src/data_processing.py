import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def load_movie_data(filepath="wiki_movie_plots.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Plot"])
    df["Plot"] = df["Plot"].str.strip().str.lower()
    df["Title"] = df["Title"].str.strip()
    return df