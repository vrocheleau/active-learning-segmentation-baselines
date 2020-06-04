import numpy as np
import random
from scipy.ndimage.morphology import distance_transform_edt
from scipy.special import softmax, xlogy
from tqdm import tqdm

def get_top_scores(scores, n_to_label):
    sorted_idx = np.argsort(scores)[::-1]
    return sorted_idx[:n_to_label]

class AbstractHeuristic:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_to_label(self, predictions, model, n_to_label):
        """
        Computes the scores and returns the indexes of samples to label
        :param predictions: Prediction logits [#batch_size, #classes, ..., #mc_iterations]
        :param model: Torch module learner
        :param n_to_label: Number of samples to label
        :return:
        """
        raise NotImplementedError


class MCDropoutUncertainty(AbstractHeuristic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_to_label(self, predictions, model, n_to_label):

        if n_to_label > len(predictions):
            return range(len(predictions))

        uncertainties = np.zeros(len(predictions))

        for i, (stack, _, _, _, _) in enumerate(predictions):
            pred_probs = softmax(stack, 0)
            pred_classes = np.argmax(pred_probs, 0)
            std = np.std(pred_classes, axis=-1)
            if self.kwargs['edt']:
                transform = distance_transform_edt((1 - np.mean(pred_classes, axis=-1)))
                uncert = np.sum(np.multiply(std, transform))
                uncertainties[i] = uncert / std.size
            else:
                sum_std = np.sum(std)
                uncertainties[i] = sum_std / std.size

        return get_top_scores(uncertainties, n_to_label)


class Random(AbstractHeuristic):

    def get_to_label(self, predictions, model, n_to_label):

        if n_to_label > len(predictions):
            return range(len(predictions))

        idx = range(len(predictions))
        return random.sample(idx, k=n_to_label)


class BALD(AbstractHeuristic):
    """
        Computes the BALD acquisition function for a set of MC dropout predictions

        References:
            https://arxiv.org/abs/1703.02910
            https://github.com/ElementAI/baal/blob/master/src/baal/active/heuristics/heuristics.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_to_label(self, predictions, model, n_to_label):

        if n_to_label > len(predictions):
            return range(len(predictions))

        scores = np.zeros(len(predictions))

        pbar = tqdm(predictions, ncols=80, desc='BALD scoring')
        for i, (stack, _, _, _, _) in enumerate(pbar):

            # [n_sample, n_class, ..., n_iterations]
            assert stack.ndim >= 3
            stack = np.expand_dims(stack, axis=0)
            stack = softmax(stack, 1)

            expected_entropy = - np.mean(np.sum(xlogy(stack, stack), axis=1), axis=-1)
            expected_p = np.mean(stack, axis=-1)
            entropy_expected_p = - np.sum(xlogy(expected_p, expected_p), axis=1)
            bald_acq = entropy_expected_p - expected_entropy

            reduced_bald_acq = np.mean(bald_acq)
            scores[i] = reduced_bald_acq

        return get_top_scores(scores, n_to_label)


class MaxEntropy(AbstractHeuristic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_to_label(self, predictions, model, n_to_label):

        if n_to_label > len(predictions):
            return range(len(predictions))

        scores = np.zeros(len(predictions))

        for i, (stack, _, _, _, _) in enumerate(predictions):
            scores[i] = - np.mean(np.sum(stack * np.log(stack + 1e-10), axis=-1))

        return get_top_scores(scores, n_to_label)


if __name__ == "__main__":

    samples = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 1000])

    print(get_top_scores(np.array(samples), 5))
