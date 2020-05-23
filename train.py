from datasets import dataset_loaders
from sacred import Experiment

ex = Experiment('al_training', ingredients=[data])

if __name__ == "__main__":
