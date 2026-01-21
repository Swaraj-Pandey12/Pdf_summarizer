import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    PAGE_TITLE = "SummarySnap - AI PDF Summarizer"
    PAGE_ICON = "📸"
    LAYOUT = "wide"

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-2.0-flash-exp"

