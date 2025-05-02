import streamlit as st
import requests


API_KEY = "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"

def get_llm_response(prompt, model="anthropic/claude-2"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system", 
                "content": "You are a movie expert. ALWAYS provide EXACTLY 5 potential movie titles that match the given plot description. If you cannot find 5, make educated guesses."
            },
            {
                "role": "user", 
                "content": f"MANDATORY: List EXACTLY 5 movie titles that could match this plot:\n{prompt}\n\nRespond with a PRECISE list of 5 movie titles ONLY, even if some are less certain."
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        completion = response.json()
        suggestions = completion["choices"][0]["message"]["content"].strip()
        movie_list = [line.strip() for line in suggestions.split('\n') if line.strip()]
        
        while len(movie_list) < 5:
            movie_list.append("(Additional movie suggestion needed)")
        
        return movie_list[:5]
    
    except Exception as e:
        return [
            "Movie Suggestion 1",
            "Movie Suggestion 2", 
            "Movie Suggestion 3", 
            "Movie Suggestion 4", 
            "Movie Suggestion 5"
        ]

st.title("Movie Plot Retriever - LLM Chatbox")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter a movie plot description")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner('Finding potential movie matches...'):
        bot_responses = get_llm_response(user_input)
    
    formatted_response = "Potential movie matches:\n\n" + "\n".join([f"• {movie}" for movie in bot_responses])
    
    st.chat_message("assistant").markdown(formatted_response)
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
