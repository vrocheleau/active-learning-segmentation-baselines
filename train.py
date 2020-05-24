from datasets import dataset_loaders
from sacred import Experiment
from datasets.dataset_loaders import dataset_ingredient, load_glas
from baal.active import ActiveLearningDataset, ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule
from baal.modelwrapper import ModelWrapper
from baal.active.heuristics import BALD
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
from models.unet import UNet
from tqdm import tqdm

ex = Experiment('al_training', ingredients=[dataset_ingredient])

@ex.config
def conf():
    data_path = 'data/GlaS'
    splits_path = 'data/splits/glas'
    preload = False
    patch_size = None
    batch_size = 16
    shuffle = True
    n_label_start = 5
    manual_seed = 0
    epochs = 30
    al_iters = 10
    n_data_to_label = 5

@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, shuffle, n_label_start, manual_seed, epochs, al_iters, n_data_to_label):

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload, patch_size, batch_size, shuffle)
    active_set = ActiveLearningDataset(train_ds)
    active_set.label_randomly(n_label_start)

    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    model = UNet()
    model = MCDropoutModule(model)

    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    # criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    model = ModelWrapper(model, criterion)

    heuristic = BALD(shuffle_prop=0.1)

    active_loop = ActiveLearningLoop(active_set, model.predict_on_dataset, heuristic, n_data_to_label)

    for al_it in tqdm(range(al_iters)):
        # # train
        model.train_on_dataset(active_set, optimizer, batch_size, epochs, use_cuda)

        # test
        model.test_on_dataset(test_ds, batch_size, use_cuda)
        metrics = model.metrics

        should_continue = active_loop.step()
        model.reset_fcs()
        if not should_continue:
            break

        val_loss = metrics['test_loss'].value
        logs = {
            "val": val_loss,
            "epoch": al_it,
            "train": metrics['train_loss'].value,
            "labeled_data": active_set._labelled,
            "Next Training set size": len(active_set)
        }
        print(logs)