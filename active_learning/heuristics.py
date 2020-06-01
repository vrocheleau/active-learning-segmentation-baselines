import numpy as np
import random
from scipy.ndimage.morphology import distance_transform_edt
from sklearn.preprocessing import MinMaxScaler
import cv2
import torch

class AbstractHeuristic:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_to_label(self, predictions, model, n_to_label):
        raise NotImplementedError


class MCDropoutUncertainty(AbstractHeuristic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_to_label(self, predictions, model, n_to_label):

        if n_to_label > len(predictions):
            return range(len(predictions))

        uncertainties = np.zeros(len(predictions))

        for i, (stack, _) in enumerate(predictions):
            std = np.std(stack, axis=0)
            if self.kwargs['edt']:
                transform = distance_transform_edt((1 - np.mean(stack, axis=0)))
                uncert = np.sum(np.multiply(std, transform))
                uncertainties[i] = uncert / std.size
            else:
                sum_std = np.sum(std)
                uncertainties[i] = sum_std / std.size

        return np.argpartition(uncertainties, -n_to_label)[-n_to_label:]


class Random(AbstractHeuristic):

    def get_to_label(self, predictions, model, n_to_label):

        if n_to_label > len(predictions):
            return range(len(predictions))

        idx = range(len(predictions))
        return random.sample(idx, k=n_to_label)


if __name__ == "__main__":
    heuristic = Random()
    samples = ['a', 'b', 'c', 'd', 'dsadas', 'qasdasdassd', '555555555555555', '12', 9]

    samples = [100,90,80,70,60,50,40,30,20,10,0]

    n_to_label = 5
    print(heuristic.get_to_label(samples, None, n_to_label))
