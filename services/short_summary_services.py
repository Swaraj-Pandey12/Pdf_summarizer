from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from services.summary_services import SummaryBase

class ShortSummaryService(SummaryBase):

    def __init__(self, llm):
        self.llm = llm

    def generate(self, documents):
        prompt = PromptTemplate(
            template="""
            Write a VERY SHORT summary (5–6 lines max) of the following text.
            Focus only on the most important points.

            Text: {text}

            SHORT SUMMARY:
            """,
            input_variables=["text"]
        )

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt
        )

        result = chain.invoke(documents)
        return result["output_text"]
