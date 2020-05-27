from torch.utils.data import DataLoader
from sacred import Experiment
from datasets.dataset_loaders import dataset_ingredient, load_glas
import torch
from torch import nn, optim
import random
from models.unet import UNet
from active_learning.method_wrapper import MCDropout_Uncert
import torch.nn.functional as F

ex = Experiment('baseline_training', ingredients=[dataset_ingredient])

@ex.config
def conf():
    data_path = 'data/GlaS'
    splits_path = 'data/splits/glas'
    preload = True
    patch_size = None
    batch_size = 16
    shuffle = True
    manual_seed = 0
    epochs = 100
    n_classes = 2

@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, shuffle, manual_seed, epochs, n_classes):

    torch.backends.cudnn.benchmark = True
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload)

    [train_ds.__getitem__(i)[0].numpy().max() for i in range(train_ds.n)]

    # Load model
    model = UNet(in_channels=3, n_classes=n_classes)

    print(model)

    method_wrapper = MCDropout_Uncert(base_model=model, n_classes=n_classes)

    method_wrapper.train(train_ds=train_ds, val_ds=val_ds, epochs=epochs, batch_size=batch_size)

    test_metrics = method_wrapper.evaluate(DataLoader(dataset=test_ds, batch_size=1, shuffle=True), test=True)

    print(test_metrics['mean_dice'])