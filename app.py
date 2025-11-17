import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="SummarySnap - AI PDF Summarizer",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def load_and_process_pdf(pdf_file):
    """Load and split PDF into chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vectorstore(splits, api_key):
    """Create Chroma vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore

def initialize_llm(api_key):
    """Initialize Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
        google_api_key=api_key
    )

def generate_summary(documents, api_key):
    """Generate summary using map_reduce"""
    llm = initialize_llm(api_key)
    
    prompt_template = """
    Write a comprehensive summary of the following text.
    Include main topics, key points, and important details.
    
    Text: {text}
    
    SUMMARY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        combine_prompt=prompt
    )
    
    result = chain.invoke(documents)
    return result["output_text"]

def handle_chat(question, vectorstore, api_key):
    """Handle chat question with context retrieval"""
    llm = initialize_llm(api_key)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    response = chain.invoke({"question": question})
    return response["answer"]

# ==================== SESSION STATE ====================

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = None

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# ==================== MAIN UI ====================

# Header
st.markdown('<h1 class="main-header">📸 SummarySnap</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Snap the essence of your PDFs instantly with AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/pdf-2.png", width=80)
    st.markdown("### 📤 Upload Your PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF document to analyze"
    )
    
    if uploaded_file:
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        file_size = uploaded_file.size / 1024
        st.metric("File Size", f"{file_size:.2f} KB")
        
        if api_key:
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("⚡ Processing your document..."):
                    try:
                        splits = load_and_process_pdf(uploaded_file)
                        st.session_state.vectorstore = create_vectorstore(splits, api_key)
                        st.session_state.processed_pdf = uploaded_file.name
                        st.session_state.documents = splits
                        st.balloons()
                        st.success(f"✨ Successfully processed {len(splits)} chunks!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.error("⚠️ API Key not found in .env file")
    
    st.markdown("---")
    
    if st.session_state.vectorstore:
        st.markdown("### 📊 Document Stats")
        st.info(f"""
        **Processed:** {st.session_state.processed_pdf}
        
        **Chunks:** {len(st.session_state.documents)}
        
        **Status:** Ready for chat!
        """)
    
    st.markdown("---")
    
    st.markdown("### 🔧 Powered By")
    st.markdown("""
    - 🤖 Google Gemini AI
    - 🦜 LangChain Framework
    - 🔍 ChromaDB Vector Store
    - ⚡ Streamlit UI
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ by [Your Name]")

# Main content
if not api_key:
    st.error("⚠️ **Google API Key not found!**")
    st.info("""
    Please add your Google Gemini API key to the `.env` file:
    
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```
    
    Get your free API key from: https://ai.google.dev/
    """)
    st.stop()

if not uploaded_file:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📝 Smart Summaries</h3>
            <p>Generate comprehensive summaries of any PDF document using advanced AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>💬 Interactive Chat</h3>
            <p>Ask questions and get instant answers from your documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🔍 Context Aware</h3>
            <p>Maintains conversation history for intelligent responses</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 How to Use SummarySnap")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        **Step 1: Upload**
        - Click on the sidebar
        - Choose your PDF file
        - Click "Process PDF"
        
        **Step 2: Summarize**
        - Go to Summary tab
        - Click "Generate Summary"
        - Download if needed
        """)
    
    with steps_col2:
        st.markdown("""
        **Step 3: Chat**
        - Switch to Chat tab
        - Ask any question
        - Get AI-powered answers
        
        **Step 4: Explore**
        - View source references
        - Continue conversations
        - Clear chat anytime
        """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Perfect For")
    use_cases = st.columns(4)
    
    with use_cases[0]:
        st.info("📚 **Research**\n\nAcademic papers & studies")
    with use_cases[1]:
        st.info("⚖️ **Legal**\n\nContracts & documents")
    with use_cases[2]:
        st.info("💼 **Business**\n\nReports & presentations")
    with use_cases[3]:
        st.info("🎓 **Education**\n\nTextbooks & materials")
    
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📝 Summary", "💬 Chat with PDF"])

# ==================== SUMMARY TAB ====================
with tab1:
    if st.session_state.vectorstore is not None:
        st.markdown("### Generate Document Summary")
        st.markdown("Click below to generate a comprehensive AI-powered summary of your document.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Generate Summary", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your document..."):
                    try:
                        summary = generate_summary(st.session_state.documents, api_key)
                        
                        st.markdown("---")
                        st.markdown("### 📄 Summary Result")
                        
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 4px solid #667eea;">
                        {summary}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary,
                                file_name=f"{st.session_state.processed_pdf}_summary.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
    else:
        st.info("👈 Please upload and process a PDF first from the sidebar")

# ==================== CHAT TAB ====================
with tab2:
    if st.session_state.vectorstore is not None:
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_question = st.chat_input("💭 Ask anything about your PDF...")
        
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = handle_chat(
                            user_question,
                            st.session_state.vectorstore,
                            api_key
                        )
                        st.markdown(answer)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.chat_history:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
    
    else:
        st.info("👈 Please upload and process a PDF first from the sidebar")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Made with ❤️ using LangChain, Google Gemini, and Streamlit</p>
    <p>📸 <strong>SummarySnap</strong> - Your AI Document Assistant</p>
</div>
""", unsafe_allow_html=True)
