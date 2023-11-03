# Python Built-in Modules
import pathlib
from dataclasses import dataclass, field

# Third-Party Libraries
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.docarray import DocArrayInMemorySearch

QA_PROMPT = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the context is unrelated to the question, ignore it.
If you don't know the answer, just say that you don't know.
Provide the shortest possible answer.

Question: ```{question}```
Context: ```{context}```
Answer: """


class RagQA:
    """
    Coordinates Document Retrieval, Question Answering and Response Generation.
    """

    def __init__(
        self,
        llm: HuggingFacePipeline,
        retriever: VectorStoreRetriever,
    ) -> None:
        """
        Constructs a new instane of RagQA.

        It's recommended that you use the RagQABuilder class rather than
        construct this class directly, to avoid having to deal with the
        construction complexity.

        Example:
            >>> from hlm12rag.modelling import RagQABuilder
            ... rag_qa = RagQABuilder(dir_docs).build()
            ... rag_qa.ask("What are the recommendations for building new microservices?")

        :param llm: The Language Model to use for response generation.
        :param retriever: The Document Retriever to use for document retrieval.
        """
        self.llm = llm
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs=dict(prompt=ChatPromptTemplate.from_template(QA_PROMPT), verbose=True),
            verbose=True,
        )

    def ask(self, question: str) -> str:
        x = dict(query=question)
        y = self.qa(x)
        return y["result"]


@dataclass(frozen=True)
class RagQABuilder:
    """
    A builder for RagQA instances.

    :attr dirpath: The directory containing the documents to be used for retrieval.
    :attr pipeline_task: The task to use for response generation, defaults to `text2text-generation`
    :attr pipeline_model_id: The model to use for response generation, defaults to `google/flan-t5-small`
    :attr pipeline_model_kwargs: The kwargs to pass to the model for response generation, defaults to dict(temperature=10e-16, max_length=64, do_sample=True)
    """

    dirpath: pathlib.Path
    retrieval_embedding_model_id: str = field(default="multi-qa-MiniLM-L6-cos-v1")
    retrieval_search_threshold: float = field(default=0.5)
    pipeline_task: str = field(default="text2text-generation")
    pipeline_model_id: str = field(default="google/flan-t5-small")
    pipeline_model_kwargs: dict = field(
        default_factory=lambda: dict(
            temperature=10e-16,
            max_length=64,
            do_sample=True,
        )
    )

    def build(self) -> RagQA:
        """
        Builds a RagQA instance using the attributes set on the builder.

        :return: A RagQA instance.
        """
        llm = HuggingFacePipeline.from_model_id(
            task=self.pipeline_task,
            model_id=self.pipeline_model_id,
            model_kwargs=self.pipeline_model_kwargs,
            verbose=True,
        )
        document_loader = DirectoryLoader(self.dirpath)
        documents = document_loader.load()
        document_chunker = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
        document_chunks = document_chunker.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=self.retrieval_embedding_model_id)
        vector_store = DocArrayInMemorySearch.from_documents(document_chunks, embeddings)
        vector_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=dict(score_threshold=self.retrieval_search_threshold),
        )
        return RagQA(llm=llm, retriever=vector_retriever)
