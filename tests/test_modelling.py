# Python Built-in Modules
import pathlib
import unittest

# My Packages and Modules
from hlm12rag.modelling import RagQABuilder


class TestRagQA(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs_generic = {
            "What is the currency in England": None,
            "Where is Brazil located? South America or North America?": None,
            "What's the largest ocean in the world?": None,
        }
        self.pairs_specific = {
            "What's arguslweruna role?": "king",
            "What is the bog near ag45i4nt like?": "dry",
            "What is 4831asx capable of?": "shooting lasers",
        }

        document_dir = pathlib.Path("./data_sample")
        self.qa = RagQABuilder(dirpath=document_dir).build()

    def test_ask_does_not_answers_from_knowledge(self):
        for question, expected in self.pairs_generic.items():
            actual = self.qa.ask(question)
            print((question, expected, actual))
            self.assertEqual(expected, actual)

    def test_ask_answers_from_docs(self):
        for question, expected in self.pairs_specific.items():
            actual = self.qa.ask(question)
            print((question, expected, actual))
            self.assertEqual(expected, actual)
