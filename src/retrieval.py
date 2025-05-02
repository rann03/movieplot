from rank_bm25 import BM25Okapi
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from qwen import get_llm_response

nltk.download("punkt", quiet=True)

@st.cache_resource(ttl=3600)
def create_bm25_index(docs):
    tokenized_corpus = [word_tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def retrieve_candidates(bm25, tokenized_corpus, docs_metadata, query, top_k=10):
    tokenized_query = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-top_k:][::-1]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "title": docs_metadata.iloc[idx]["Title"],
            "plot": docs_metadata.iloc[idx]["Plot"],
            "score": doc_scores[idx]
        })
    return candidates

def expand_query(query):
    prompt = "Rewrite this movie plot description to add more related keywords and phrasing, so it improves search recall:\n\n" + query
    expanded = get_llm_response(prompt)
    return expanded if expanded else query

def rerank_results(query, movie_candidates):
    prompt = f"Given the query: \"{query}\", rank the following movies in order of relevance based strictly on their plot descriptions. Reply only with the movie titles separated by commas.\n\n"
    for idx, movie in enumerate(movie_candidates, 1):
        prompt += f"{idx}. {movie['title']}: {movie['plot']}\n"

    ranked_text = get_llm_response(prompt)
    if not ranked_text:
        return [m["title"] for m in movie_candidates]

    ranked_titles = [title.strip() for title in ranked_text.split(",") if title.strip()]
    return ranked_titles