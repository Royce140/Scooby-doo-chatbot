import streamlit as st
from initial import Chatbot, generate_response
import os
# Streamlit app
st.title("Scooby-Doo Chatbot")
# Setup chatbot
chatbot = Chatbot()

#Setup sidebar
#sidebar = st.sidebar
#check = sidebar.button("Setup chatbot")
if check:
    chatbot.setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
