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
import numpy as np
from matplotlib import pyplot
from baal.bayesian.dropout import MCDropoutModule
import os, shutil
from sklearn.preprocessing import MinMaxScaler

ex = Experiment('al_training', ingredients=[dataset_ingredient])

@ex.config
def conf():
    data_path = 'data/GlaS'
    splits_path = 'data/splits/glas'
    preload = False
    patch_size = None
    batch_size = 32
    shuffle = True
    n_label_start = 50
    manual_seed = 0
    epochs = 30
    al_cycles = 100
    n_data_to_label = 5
    mc_iters = 50


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

    for epoch in tqdm(range(epochs), ncols=100, desc='Training'):
        model.train()

        for images, masks, _, _ in train_loader:
            images, masks = images.to(device), masks.squeeze(1).to(device, non_blocking=True)
            class_masks = (masks != 0).long()

            out = model(images)

            loss = criterion(out, class_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def rank_mc_dropout_euclidean(model, train_ds, n_iters, n_to_label, al_cycle):
    # If the pool is smaller than n_to_label simply return every index
    if n_to_label > train_ds.n_unlabelled:
        return range(train_ds.n_unlabelled)

    # Check if map dirs exist
    dir = 'maps/run_{}/'.format(al_cycle)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model)
    model.to(device)

    pool_loader = DataLoader(train_ds.pool, batch_size=1, shuffle=False)

    uncerts = np.zeros(train_ds.n_unlabelled)
    with torch.no_grad():
        model.eval()

        for i, (image, _, _, name) in enumerate(tqdm(pool_loader, desc='Pool ranking')):

            stack = []
            for _ in range(n_iters):
                out = model(image)

                pred = out.cpu().numpy().squeeze(0).argmax(0)
                stack.append(pred)

            scaler = MinMaxScaler()
            stack = np.array(stack)
            map = np.var(stack, axis=0)

            map = scaler.fit_transform(map)

            uncerts[i] = np.sum(np.std(stack, axis=0))

            fig_path = os.path.join(dir, 'map_{}.png'.format(i))
            fig = pyplot.figure()
            pyplot.imshow(map, cmap='viridis')
            pyplot.colorbar()
            pyplot.title('{} uncertainty map'.format(name[0]))
            pyplot.savefig(fig_path)
            pyplot.close(fig)

    # Get ids of n_to_label most uncertain samples in the pool
    idx = np.argpartition(uncerts, -n_to_label)[-n_to_label:]
    return idx


@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, shuffle, n_label_start, manual_seed, epochs, al_cycles, n_data_to_label, mc_iters):

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload)
    active_set = ActiveLearningDataset(train_ds)

    active_set.label_randomly(n_label_start)

    torch.backends.cudnn.benchmark = True
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Load model
    model = UNet(in_channels=3, n_classes=2)
    model = MCDropoutModule(model)

    print(model)

    # Save state dict to reset the model
    base_state_dict = model.state_dict()

    last_cycle = False
    for al_it in range(al_cycles):

        print("##### ACTIVE LEARNING ITERATION {}/{}#####".format(al_it, al_cycles))

        # Reset model
        model.load_state_dict(base_state_dict)
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        # train
        train(model, active_set, val_ds, epochs, criterion, optimizer)

        if last_cycle:
            break

        # Uncertainty ranking
        label_idx = rank_mc_dropout_euclidean(model=model,
                                              train_ds=active_set,
                                              n_iters=mc_iters,
                                              n_to_label=n_data_to_label,
                                              al_cycle=al_it)

        # Label new samples
        active_set.label(label_idx)

        if active_set.n_unlabelled == 0:
            print("Every sample from the pool has been labeled, closing AL loop.")
            last_cycle = True
