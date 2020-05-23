from sacred import Ingredient
from torchvision import transforms
from torch.utils.data import DataLoader

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
def load_glas(data_path, preload, patch_size, batch_size, shuffle, sampler_mul, num_workers, drop_last, pin_memory):

