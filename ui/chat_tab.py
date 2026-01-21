import streamlit as st
from services.llm_services import LLMService
from services.chat_services import ChatService

def chat_tab(config):
    if st.session_state.vectorstore:
        llm = LLMService(
            config.GOOGLE_API_KEY,
            config.LLM_MODEL
        ).get_llm()

        chat_service = ChatService(llm, st.session_state.vectorstore)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask about PDF")

        if question:
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )

            answer = chat_service.ask(question)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            st.rerun()
