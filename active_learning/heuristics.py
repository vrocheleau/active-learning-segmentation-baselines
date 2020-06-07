import numpy as np
import random
from scipy.ndimage.morphology import distance_transform_edt
from scipy.special import softmax, xlogy
from tqdm import tqdm


def get_ranked_indexes(scores, n_to_label):
    sorted_idx = np.argsort(scores)[::-1]
    return sorted_idx[:n_to_label]


def get_balanced_idx(ranked_indexes, predictions, n_to_label, num_classes):

    idx = []
    for c in range(num_classes):

        class_count = 0
        i = 0
        while class_count < n_to_label:
            stack, _, _, _, lbl = predictions[ranked_indexes[i]]
            if c == lbl:
                idx.append(ranked_indexes[i])
                class_count += 1
            i += 1
    return idx


class AbstractHeuristic:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_to_label(self, predictions, model, n_to_label, num_classes, balance_al):
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

    def get_to_label(self, predictions, model, n_to_label, num_classes, balance_al):

        if n_to_label > len(predictions):
            return range(len(predictions))

        uncertainties = np.zeros(len(predictions))

        for i, (stack, _, _, _, lbl) in enumerate(predictions):
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

        if balance_al:
            ranked_indexes = get_ranked_indexes(uncertainties, len(predictions))
            return get_balanced_idx(ranked_indexes, predictions, n_to_label, num_classes)
        else:
            return get_ranked_indexes(uncertainties, n_to_label)


class Random(AbstractHeuristic):

    def get_to_label(self, predictions, model, n_to_label, num_classes, balance_al):

        shuffled_indexes = np.arange(len(predictions))
        for i in range(1000):
            np.random.shuffle(shuffled_indexes)

        if balance_al:
            return get_balanced_idx(shuffled_indexes, predictions, n_to_label, num_classes)
        else:
            return shuffled_indexes[:n_to_label]


class BALD(AbstractHeuristic):
    """
        Computes the BALD acquisition function for a set of MC dropout predictions

        References:
            https://arxiv.org/abs/1703.02910
            https://github.com/ElementAI/baal/blob/master/src/baal/active/heuristics/heuristics.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_to_label(self, predictions, model, n_to_label, num_classes, balance_al):

        if n_to_label > len(predictions):
            return range(len(predictions))

        scores = np.zeros(len(predictions))

        pbar = tqdm(predictions, ncols=80, desc='BALD scoring')
        for i, (stack, _, _, _, lbl) in enumerate(pbar):

            # [n_sample, n_class, ..., n_iterations]
            assert stack.ndim >= 3
            stack = np.expand_dims(stack, axis=0)

            # requires probabilities
            stack = softmax(stack, 1)

            # BALD score
            expected_entropy = - np.mean(np.sum(xlogy(stack, stack), axis=1), axis=-1)
            expected_p = np.mean(stack, axis=-1)
            entropy_expected_p = - np.sum(xlogy(expected_p, expected_p), axis=1)
            bald_acq = entropy_expected_p - expected_entropy

            reduced_bald_acq = np.mean(bald_acq)
            scores[i] = reduced_bald_acq

        if balance_al:
            ranked_indexes = get_ranked_indexes(scores, len(predictions))
            return get_balanced_idx(ranked_indexes, predictions, n_to_label, num_classes)
        else:
            return get_ranked_indexes(scores, n_to_label)


class MaxEntropy(AbstractHeuristic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_to_label(self, predictions, model, n_to_label, num_classes, balance_al):

        if n_to_label > len(predictions):
            return range(len(predictions))

        scores = np.zeros(len(predictions))

        for i, (stack, _, _, _, lbl) in enumerate(predictions):
            # [n_sample, n_class, ..., n_iterations]
            assert stack.ndim >= 3
            stack = np.expand_dims(stack, axis=0)

            # requires probabilities
            stack = softmax(stack, 1)

            pixel_wise_entropy = - np.mean(np.sum(xlogy(stack, stack), axis=1), axis=-1)
            scores[i] = np.mean(pixel_wise_entropy)

        if balance_al:
            ranked_indexes = get_ranked_indexes(scores, len(predictions))
            return get_balanced_idx(ranked_indexes, predictions, n_to_label, num_classes)
        else:
            return get_ranked_indexes(scores, n_to_label)


if __name__ == "__main__":
    # np.random.seed(1337)

    samples = np.array(range(100))

    heur = Random()

