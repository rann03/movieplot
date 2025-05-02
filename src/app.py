import streamlit as st
from qwen import get_llm_response

st.title("Movie Plot Retriever - LLM Chatbox")

if "messages" not in st.session_state:
    st.session_state.messages = []

def submit():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = user_input

    bot_reply = get_llm_response(prompt)
    if not bot_reply:
        bot_reply = "Sorry, I couldn't generate a response. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.session_state.user_input = ""

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

st.text_area("Enter a movie plot description or question:", key="user_input", on_change=submit, placeholder="Type here and press Enter")