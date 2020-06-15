import os
import random
import shutil

import numpy as np
import torch
import seaborn as sns
from baal.active import ActiveLearningDataset
from sacred import Experiment
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from os.path import join

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
    al_cycles = 10
    n_data_to_label = 1
    mc_iters = 20
    base_state_dict_path = 'state_dicts/al_model.pt'
    heuristic = 'mc'
    run = 0
    results_dir = 'results/exp_7/'
    save_maps = True
    balance_al = True
    num_classes = 2
    save_uncerts = False


def save_prediction_maps(stacks_list, al_cycle, map_processor, heuristic, uncertainties=None):
    # Check if map dirs exist
    # dir = 'maps/run_{}/'.format(al_cycle)

    heur_dir = 'maps/{}/'.format(heuristic)
    run_dir = join(heur_dir, 'run_{}/'.format(al_cycle))

    if not os.path.isdir(heur_dir):
        os.mkdir(heur_dir)

    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)

    for i, (stack, image, mask, name, lbl) in enumerate(tqdm(stacks_list, ncols=80, desc="Save pred maps")):
        map_processor.process_maps(predictions=stack,
                                   sample_name=name,
                                   sample_num=i,
                                   save_dir=run_dir,
                                   input_img=image,
                                   gt_img=mask,
                                   scores=uncertainties)


def save_uncert_histogram(uncertainties, heuristic_name):

    dir = 'uncerts/{}/'.format(heuristic_name)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)

    for i, scores in enumerate(uncertainties):

        fig = plt.figure()
        sns.distplot(scores, hist=False, rug=True)
        plt.title('Uncertainty density ({} samples)'.format(len(scores)))
        plt.savefig(os.path.join(dir, 'run_{}.png'.format(i)))
        plt.close(fig)


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
    'bald': heuristics.BALD(),
    'max': heuristics.MaxEntropy()
}

@ex.automain
def main(data_path, splits_path, preload, patch_size, batch_size, n_label_start, manual_seed, epochs, al_cycles,
         n_data_to_label, mc_iters, base_state_dict_path, heuristic, run, results_dir, balance_al, num_classes, save_maps, save_uncerts):

    print("Start of AL experiment using {} heuristic".format(heuristic))
    torch.backends.cudnn.benchmark = True

    if manual_seed:
        torch.manual_seed(manual_seed)
        random.seed(manual_seed)
        np.random.seed(manual_seed)

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload, patch_size=patch_size)
    active_set = ActiveLearningDataset(train_ds, pool_specifics=pool_specifics)

    # Label 4 images, 2 for each class
    active_set.label([0, 1, 2, 3])

    # Load model
    model = UNet(in_channels=3, n_classes=num_classes, dropout=True)
    print(model)

    method_wrapper = MCDropoutUncert(base_model=model, n_classes=num_classes, state_dict_path=base_state_dict_path)

    acq_scores = []
    mean_dices = []
    last_cycle = False
    for al_it in range(al_cycles + 1):

        print("##### ACTIVE LEARNING ITERATION {}/{}#####".format(al_it, al_cycles))
        print("Labeled pool size: {}".format(active_set.n_labelled))
        print("Unlabeled pool size: {}".format(active_set.n_unlabelled))

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

        heur = heuristics_dict[heuristic]
        to_label, scores = heur.get_to_label(predictions=predictions,
                                     model=None,
                                     n_to_label=n_data_to_label,
                                     num_classes=num_classes,
                                     balance_al=balance_al)

        if save_maps:
            save_prediction_maps(predictions, al_it, map_processor=ComparativeVarianceMap(), heuristic=heuristic, uncertainties=scores)

        del predictions
        acq_scores.append(scores)

        # Label new samples
        active_set.label(to_label)

        if active_set.n_unlabelled == 0:
            last_cycle = True

    print(mean_dices)
    np.save(results_dir + '{}_{}.npy'.format(heuristic, run), np.array(mean_dices))
    if save_uncerts:
        save_uncert_histogram(acq_scores, heuristic)
