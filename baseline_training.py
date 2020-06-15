from torch.utils.data import DataLoader
from sacred import Experiment
from datasets.dataset_loaders import dataset_ingredient, load_glas
import torch
from torch.backends import cudnn
from torch import nn, optim
import random
from models.unet import UNet
from active_learning.method_wrapper import MCDropoutUncert
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
from torch.optim import SGD, lr_scheduler
import numpy as np
from active_learning.heuristics import Random, MCDropoutUncertainty, MaxEntropy, BALD
import scipy

ex = Experiment('baseline_training', ingredients=[dataset_ingredient])

@ex.config
def conf():
    data_path = 'data/GlaS'
    splits_path = 'data/splits/glas'
    preload = True
    batch_size = 16
    shuffle = True
    manual_seed = 0
    epochs = 50
    n_classes = 2
    patch_size = (416, 416)

def get_optimizer_scheduler(model):
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40)
    return optimizer, scheduler

@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, shuffle, manual_seed, epochs, n_classes):

    torch.manual_seed(manual_seed)
    random.seed(manual_seed)

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload, patch_size=patch_size)

    # Load model
    model = UNet(in_channels=3, n_classes=n_classes, dropout=True)

    print(model)

    method_wrapper = MCDropoutUncert(base_model=model, n_classes=n_classes, state_dict_path='state_dicts/baseline_model.pt')

    method_wrapper.train(train_ds=train_ds,
                         val_ds=val_ds,
                         test_ds=test_ds,
                         epochs=epochs,
                         batch_size=batch_size,
                         opt_sch_callable=get_optimizer_scheduler)

    test_metrics = method_wrapper.evaluate(DataLoader(dataset=test_ds, batch_size=1, shuffle=True), test=True)

    print(test_metrics['mean_dice'])