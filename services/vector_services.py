from langchain_google_genai  import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

class VectorService:
    def __init__(self, api_key, model):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )

    def create_vectorstore(self, documents):
        return Chroma.from_documents(documents, self.embeddings)
