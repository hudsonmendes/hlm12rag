# Python Built-in Modules
from typing import Callable, List
from dataclasses import dataclass, field

# Third-Party Libraries
import pandas as pd
from sklearn.model_selection import KFold

# Local Folders
from tqdm.auto import tqdm
from .modelling import RagQA


@dataclass(frozen=True)
class QATrainingResult:
    tp: int = field()
    count: int = field()

    @property
    def score(self) -> float:
        return self.tp / self.count if self.count != 0 else 0.0


@dataclass(frozen=True)
class QATraining:
    results: List[QATrainingResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum([r.score for r in self.results]) / len(self.results)


class QATrainer:
    dataset: pd.DataFrame
    k: int

    def __init__(
        self,
        dataset: pd.DataFrame,
        correctness_fn: Callable[[str, str], bool],
        k: int = 5,
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        self.correctness_fn = correctness_fn

    def train(self, model: RagQA) -> QATraining:
        results: List[QATrainingResult] = []
        for train_index, test_index in self.kfold.split(self.dataset):
            _, df_valid = self.dataset[train_index], self.dataset[test_index]
            y_pred = [model.ask(x) for x in df_valid["question"]]
            y_true = df_valid["answer"].values
            y_pairs = tqdm(list(zip(y_pred, y_true)))
            corrects = [True for (yp, yt) in y_pairs if self.correctness_fn(yp, yt)]
            results.append(len(corrects), len(y_true))
        return QATraining(results=results)
