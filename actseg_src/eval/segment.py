import typing as t

import numpy as np

from . import Metric
from .external.mstcn_code import edit_score


class Edit(Metric):
    def __init__(self, ignore_ids: t.Sequence[int] = ()):
        self.ignore_ids = ignore_ids
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self) -> None:
        self.values = []

    def add(self, targets: t.Sequence[int], predictions: t.Sequence[int]) -> float:
        current_score = edit_score(
            recognized=predictions,
            ground_truth=targets,
            bg_class=self.ignore_ids,
        )

        self.values.append(current_score)
        return current_score

    def summary(self) -> float:
        if len(self.values) > 0:
            return np.array(self.values).mean()
        else:
            return 0.0
