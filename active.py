import os
import random
import shutil

import numpy as np
import torch
from baal.active import ActiveLearningDataset
from sacred import Experiment
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from active_learning import heuristics
from active_learning.maps import ComparativeVarianceMap
from active_learning.method_wrapper import MCDropoutUncert
from datasets.dataset_loaders import dataset_ingredient, load_glas
from models.unet import UNet

ex = Experiment('al_training', ingredients=[dataset_ingredient])

@ex.config
def conf():
    data_path = 'data/GlaS'
    splits_path = 'data/splits/glas'
    preload = True
    patch_size = (416, 416)
    batch_size = 16
    n_label_start = 5
    manual_seed = None
    epochs = 50
    al_cycles = 20
    n_data_to_label = 1
    mc_iters = 50
    base_state_dict_path = 'state_dicts/al_model.pt'
    heuristic = 'mc'
    run = 0
    results_dir = 'results/exp_5/'


def save_prediction_maps(stacks_list, al_cycle, map_processor):
    # Check if map dirs exist
    dir = 'maps/run_{}/'.format(al_cycle)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)

    for i, (stack, image, mask, name, lbl) in enumerate(tqdm(stacks_list, ncols=80, desc="Save pred maps")):
        map_processor.process_maps(predictions=stack,
                                   sample_name=name,
                                   sample_num=i,
                                   save_dir=dir,
                                   input_img=image,
                                   gt_img=mask)


def get_optimizer_scheduler(model):
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40)
    return optimizer, scheduler


pool_specifics = {
    'data_transform': transforms.ToTensor(),
    'mask_transform': transforms.ToTensor(),
    'augment': False,
    'patch_size': None
}

heuristics_dict = {
    'mc': heuristics.MCDropoutUncertainty(edt=True),
    'rand': heuristics.Random(),
    'bald': heuristics.BALD()
}

@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, n_label_start, manual_seed, epochs, al_cycles,
         n_data_to_label, mc_iters, base_state_dict_path, heuristic, run, results_dir):

    print("Start of AL experiment using {} heuristic".format(heuristic))
    torch.backends.cudnn.benchmark = True

    if manual_seed:
        torch.manual_seed(manual_seed)
        random.seed(manual_seed)
        np.random.seed(manual_seed)

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload, patch_size=patch_size)
    active_set = ActiveLearningDataset(train_ds, pool_specifics=pool_specifics)

    # active_set.label_randomly(n_label_start)
    active_set.label(range(n_label_start))

    # Load model
    model = UNet(in_channels=3, n_classes=2, dropout=True)
    print(model)

    method_wrapper = MCDropoutUncert(base_model=model, n_classes=2, state_dict_path=base_state_dict_path)

    mean_dices = []
    last_cycle = False
    for al_it in range(al_cycles):

        print("##### ACTIVE LEARNING ITERATION {}/{}#####".format(al_it, al_cycles))

        # Reset model
        method_wrapper.reset_params()

        # train
        method_wrapper.train(train_ds=active_set,
                             val_ds=val_ds,
                             epochs=epochs,
                             batch_size=batch_size,
                             opt_sch_callable=get_optimizer_scheduler)

        test_metrics = method_wrapper.evaluate(DataLoader(dataset=test_ds, batch_size=1, shuffle=False), test=True)
        mean_dices.append(test_metrics['mean_dice'])

        if last_cycle:
            print("Every sample from the pool has been labeled, closing AL loop.")
            break

        # Make predictions on unlabeled pool
        predictions = method_wrapper.predict(active_set.pool, n_predictions=mc_iters)

        # save_prediction_maps(predictions, al_it, map_processor=ComparativeVarianceMap())

        heur = heuristics_dict[heuristic]
        to_label = heur.get_to_label(predictions=predictions, model=None, n_to_label=n_data_to_label)

        del predictions

        # Label new samples
        active_set.label(to_label)

        if active_set.n_unlabelled == 0:
            last_cycle = True

    print(mean_dices)
    np.save(results_dir + '{}_{}.npy'.format(heuristic, run), np.array(mean_dices))

