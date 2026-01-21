import streamlit as st

def init_state():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_pdf" not in st.session_state:
        st.session_state.processed_pdf = None
