from torch.utils.data import DataLoader
from sacred import Experiment
from datasets.dataset_loaders import dataset_ingredient, load_glas
from baal.active import ActiveLearningDataset, ActiveLearningLoop
import torch
from torch import nn, optim
from torch.optim import SGD, lr_scheduler
import random
from models.unet import UNet
from tqdm import tqdm
from copy import deepcopy

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
    al_iters = 1
    n_data_to_label = 5


def train(model, train_ds, valid_ds, epochs, criterion, optimizer):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model)
    model.to(device)
    lr_step = 10
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step)

    best_valid_dice = 0
    best_valid_loss = float('inf')
    best_model_dict = deepcopy(model.module.state_dict())

    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
    val_loader = DataLoader(valid_ds, batch_size=1, shuffle=True)

    for epoch in range(epochs): #itworks
        model.train()

        for images, masks in tqdm(train_loader, ncols=100, desc='Training'):
            images, masks = images.to(device), masks.squeeze(1).to(device, non_blocking=True)

            out = model(images)

            loss = criterion(out, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, shuffle, n_label_start, manual_seed, epochs, al_iters, n_data_to_label):

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload, patch_size, batch_size, shuffle)
    active_set = ActiveLearningDataset(train_ds)
    active_set.label_randomly(n_label_start)

    torch.backends.cudnn.benchmark = True
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Load model
    model = UNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for al_it in tqdm(range(al_iters)):

        # train
        train(model, active_set, val_ds, epochs, criterion, optimizer)

