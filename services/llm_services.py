from langchain_google_genai import ChatGoogleGenerativeAI

class LLMService:
    def __init__(self, api_key, model, temperature=0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    def get_llm(self):
        return self.llm
