import streamlit as st
from services.pdf_services import PDFService
from services.vector_services import VectorService

def sidebar_ui(config):
    st.image("https://img.icons8.com/fluency/96/000000/pdf-2.png", width=80)
    st.markdown("### 📤 Upload PDF")

    uploaded = st.file_uploader("Choose PDF", type="pdf")

    if uploaded and st.button("🚀 Process PDF"):
        pdf_service = PDFService()
        docs = pdf_service.load_and_split(uploaded)

        vector_service = VectorService(
            config.GOOGLE_API_KEY,
            config.EMBEDDING_MODEL
        )

        st.session_state.documents = docs
        st.session_state.vectorstore = vector_service.create_vectorstore(docs)
        st.session_state.processed_pdf = uploaded.name
        st.success("PDF processed successfully")

    return uploaded
