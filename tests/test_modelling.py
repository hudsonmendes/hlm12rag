import unittest

from hlm12rag.modelling import RagQA


class TestRagQA(unittest.TestCase):
    def setUp(self) -> None:
        self.qa = RagQA()

    def test_ask_produces_response(self):
        answer = self.qa.ask("What's arguslweruna role?")
        self.assertEqual("king", answer)
