import unittest

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from hlm12rag.modelling import RagQA


class TestRagQA(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs_generic = {
            "Berlin or London": "London",
            "In which continent is Brazil located?": "South America",
            "What's the largest ocean in the world?": "Pacific Ocean",
        }
        self.pairs_specific = {
            "What's arguslweruna role?": "king",
            "What is the bog near ag45i4nt like?": "dry",
            "What is 4831asx capable of?": "shooting lasers",
        }

        llm = HuggingFacePipeline.from_model_id(
            task="text2text-generation",
            model_id="google/flan-t5-small",
            model_kwargs=dict(temperature=10e-16, max_length=64, do_sample=True),
        )
        self.qa = RagQA(llm=llm)

    def test_ask_answers_from_knowledge(self):
        for question, expected in self.pairs_generic.items():
            actual = self.qa.ask(question)
            print((question, expected, actual))
            self.assertEqual(expected, actual)

    def test_ask_answers_from_docs(self):
        for question, expected in self.pairs_specific.items():
            actual = self.qa.ask(question)
            print((question, expected, actual))
            self.assertEqual(expected, actual)
