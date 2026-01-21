import streamlit as st
from config import AppConfig
from state.session_state import init_state
from ui.sidebar import sidebar_ui
from ui.summary_tab import summary_tab
from ui.chat_tab import chat_tab

st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT
)

init_state()

st.title("📸 SummarySnap")

uploaded = sidebar_ui(AppConfig)

if st.session_state.vectorstore:
    tab1, tab2 = st.tabs(["📝 Summary", "💬 Chat"])
    with tab1:
        summary_tab(AppConfig)
    with tab2:
        chat_tab(AppConfig)
