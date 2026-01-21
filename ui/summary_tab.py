import streamlit as st
from services.llm_services  import LLMService
from services.Long_summary_services import LongSummaryService
from services.short_summary_services import ShortSummaryService

def summary_tab(config):

    if not st.session_state.vectorstore:
        st.info("Upload and process a PDF first")
        return

    st.subheader("Generate Summary")

    summary_type = st.selectbox(
        "Select summary type",
        ["Long Summary", "Short Summary"]
    )

    if st.button("✨ Generate Summary"):

        llm = LLMService(
            config.GOOGLE_API_KEY,
            config.LLM_MODEL
        ).get_llm()

        # 🔑 OCP in action
        if summary_type == "Long Summary":
            summary_service = LongSummaryService(llm)
        else:
            summary_service = ShortSummaryService(llm)

        summary = summary_service.generate(st.session_state.documents)

        st.markdown("### 📄 Summary Result")
        st.write(summary)

        st.download_button(
            "📥 Download Summary",
            summary,
            file_name="summary.txt"
        )
