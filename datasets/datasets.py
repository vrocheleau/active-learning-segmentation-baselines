from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from random import random, randint
from os.path import join
import torch

class SegDataset(Dataset):

    def __init__(self, data_dir, rows, data_transform, mask_transform, preload, resize=None):
        self.data_dir = data_dir
        self.rows = rows
        self.data_transform = data_transform
        self.mask_transform = mask_transform
        self.preload = preload

        self.n = len(rows)
        self.joined_rows = [[join(data_dir, file), join(data_dir, mask), lbl] for file, mask, lbl in rows]

        if preload:
            self.images = self.load_images(data_dir, [file for file, _, _ in rows])
            self.masks = self.load_images(data_dir, [mask for _, mask, _ in rows])
            self.labels = [lbl for _, _, lbl in rows]


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):

        image_path, mask_path, label = self.joined_rows[item]
        if self.preload:
            image = self.images[image_path]
            mask = self.masks[mask_path]
        else:
            image = Image.open(image_path)
            mask = Image.open(mask_path)

        image = image.convert('RGB')
        mask = mask.convert('L')

        if self.data_transform:
            image = self.data_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        file_name = self.rows[item][0]
        return image, mask, label, file_name


    def load_images(self, data_dir, files):
        images = {}
        for f in files:
            path = join(data_dir, f)
            images[str(path)] = Image.open(path)
        return images


