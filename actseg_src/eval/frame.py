import typing as t

import numpy as np

from . import Metric
from .external.isba_code import IoD as IoDExternal, IoU as IoUExternal
from .external.mstcn_code import f_score


def careful_divide(correct: int, total: int, zero_value: float = 0.0) -> float:
    if total == 0:
        return zero_value
    else:
        return correct / total


class F1Score(Metric):
    def __init__(
        self,
        overlaps: t.Sequence[float] = (0.1, 0.25, 0.5),
        ignore_ids: t.Sequence[int] = (),
    ):
        self.overlaps = overlaps
        self.ignore_ids = ignore_ids
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self) -> None:
        self.tp = [0.0] * len(self.overlaps)
        self.fp = [0.0] * len(self.overlaps)
        self.fn = [0.0] * len(self.overlaps)

    def add(
        self, targets: t.Sequence[int], predictions: t.Sequence[int]
    ) -> t.List[float]:
        current_result = []

        for s in range(len(self.overlaps)):
            tp1, fp1, fn1 = f_score(
                predictions,
                targets,
                self.overlaps[s],
                bg_class=self.ignore_ids,
            )
            self.tp[s] += tp1
            self.fp[s] += fp1
            self.fn[s] += fn1

            current_f1 = self.get_f1_score(tp1, fp1, fn1)
            current_result.append(current_f1)

        return current_result

    def summary(self) -> t.List[float]:
        result = []

        for s in range(len(self.overlaps)):
            f1 = self.get_f1_score(tp=self.tp[s], fp=self.fp[s], fn=self.fn[s])
            result.append(f1)

        return result

    @staticmethod
    def get_f1_score(tp: float, fp: float, fn: float) -> float:
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0

        if precision + recall != 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = f1 * 100
        else:
            f1 = 0.0

        return f1


class MoFAccuracy(Metric):
    def __init__(self, ignore_ids: t.Sequence[int] = ()):
        self.ignore_ids = ignore_ids

        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self) -> None:
        self.total = 0
        self.correct = 0

    def add(self, targets: t.Sequence[int], predictions: t.Sequence[int]) -> float:
        assert len(targets) == len(predictions)
        targets, predictions = np.array(targets), np.array(predictions)

        mask = np.logical_not(np.isin(targets, self.ignore_ids))
        targets, predictions = targets[mask], predictions[mask]

        current_total = len(targets)
        # noinspection PyUnresolvedReferences
        current_correct = (targets == predictions).sum()
        current_result = careful_divide(current_correct, current_total)

        self.correct += current_correct
        self.total += current_total

        return current_result

    def summary(self) -> float:
        return careful_divide(self.correct, self.total)

    def name(self) -> str:
        if self.ignore_ids:
            return "MoF-BG"
        else:
            return "MoF"


class MoFAccuracyFromLogits(MoFAccuracy):
    def add(self, targets: t.Sequence[int], predictions: np.ndarray) -> float:
        """
        Here we assume the predictions are logits of shape [N x C]
        It can be torch or numpy array.

        N: number of predictions
        C: number of classes

        Implementation is simple, first convert logits to classes,
        then call parent class.
        """

        prediction = predictions.argmax(-1)
        return super().add(targets, prediction)


class IoD(Metric):
    def __init__(self, ignore_ids: t.Sequence[int] = ()):
        self.ignore_ids = ignore_ids
        self.calculation_function = IoDExternal
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self) -> None:
        self.values = []

    def add(self, targets, predictions) -> float:
        assert len(targets) == len(predictions)
        targets, predictions = np.array(targets), np.array(predictions)
        result = self.calculation_function(predictions, targets, self.ignore_ids)
        self.values.append(result)
        return result

    def summary(self) -> float:
        if len(self.values) > 0:
            return sum(self.values) / len(self.values)
        else:
            return 0.0

    def name(self) -> str:
        if not self.ignore_ids:
            return "IoD"
        else:
            return "IoD-BG"


class IoU(IoD):
    def __init__(self, ignore_ids: t.Sequence[int] = ()):
        super().__init__(ignore_ids=ignore_ids)
        self.calculation_function = IoUExternal

    def name(self) -> str:
        if not self.ignore_ids:
            return "IoU"
        else:
            return "IoU-BG"
