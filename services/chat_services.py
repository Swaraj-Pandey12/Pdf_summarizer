from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ChatService:
    def __init__(self, llm, vectorstore):
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            return_source_documents=True
        )

    def ask(self, question):
        return self.chain.invoke({"question": question})["answer"]
