from sacred import Ingredient
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import SegDataset
from utils import *

from baal.active import get_heuristic, ActiveLearningDataset

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def config():
    data_path = 'data'
    preload = True
    batch_size = 64
    shuffle = True
    num_workers = 8


@dataset_ingredient.named_config
def glas():
    data_path = 'data/GlaS'
    name = 'glas'
    folds_dir = 'folds/glas'
    batch_size = 16
    patch_size = 416
    sampler_mul = 8


@dataset_ingredient.capture
def load_glas(data_path, splits_path, preload, patch_size, batch_size, shuffle):

    train_split, test_split, val_split = get_paths(splits_path, 'csv')

    train_rows = csv_reader(train_split)
    test_rows = csv_reader(test_split)
    val_rows = csv_reader(val_split)

    train_trans = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor(),
            ])

    train_ds = SegDataset(data_dir=data_path,
                          rows=train_rows,
                          data_transform=train_trans,
                          mask_transform=transforms.ToTensor(),
                          preload=preload)

    test_ds = SegDataset(data_dir=data_path,
                         rows=test_rows,
                         data_transform=transforms.ToTensor(),
                         mask_transform=transforms.ToTensor(),
                         preload=preload)

    val_ds = SegDataset(data_dir=data_path,
                        rows=val_rows,
                        data_transform=transforms.ToTensor(),
                        mask_transform=transforms.ToTensor(),
                        preload=preload)

    return train_ds, test_ds, val_ds


def get_splits(splits_path):
    splits_files = get_paths(splits_path, 'csv')
    train = list(filter(lambda f: 'train' in f, splits_files))[0]
    test = list(filter(lambda f: 'train' in f, splits_files))[0]
    val = list(filter(lambda f: 'train' in f, splits_files))[0]
    return train, test, val


if __name__ == '__main__':
    data_path = '../data/GlaS'
    splits_path = '../data/splits/glas'
    preload = False
    patch_size = None
    batch_size = 16
    shuffle = True

    train_ds, test_ds, val_ds = load_glas(data_path, splits_path, preload, patch_size, batch_size, shuffle)
    active_set = ActiveLearningDataset(train_ds)
    active_set.label_randomly(5)

    img, mask = train_ds.__getitem__(0)

    print(active_set.n_labelled)