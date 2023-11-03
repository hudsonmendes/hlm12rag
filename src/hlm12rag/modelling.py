from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


class RagQA:
    """
    Coordinates Document Retrieval, Question Answering and Response Generation.
    """

    def __init__(self, llm: HuggingFacePipeline) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            "\n".join(
                [
                    "You are an assistant for question-answering tasks.",
                    "If you don't know the answer, just say that you don't know.",
                    "Provide the shortest possible answer.",
                    "",
                    "Question: ```{question}```",
                    "Answer: ",
                ]
            )
        )
        self.qa = LLMChain(llm=llm, prompt=self.prompt)

    def ask(self, question: str) -> str:
        x = dict(question=question)
        y = self.qa(x)
        return y["text"]
