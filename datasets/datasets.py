from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from random import random, randint
from os.path import join
import numpy as np
from torchvision import transforms


class SegDataset(Dataset):

    def __init__(self, data_dir, rows, data_transform, mask_transform, preload, patch_size=None, augment=False,
                 resize=None):
        self.data_dir = data_dir
        self.rows = rows
        self.data_transform = data_transform
        self.mask_transform = mask_transform
        self.preload = preload
        self.augment = augment
        self.patch_size = patch_size
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

        if self.augment:

            # random rotate
            angle = randint(0, 3) * 90
            image = image.rotate(angle)
            mask = mask.rotate(angle)

            # random crop
            if self.patch_size is not None:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=self.patch_size)

                image = transforms.functional.crop(image, i, j, h, w)
                mask = transforms.functional.crop(mask, i, j, h, w)

            # flip
            if random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.patch_size is not None:
            center_crop = transforms.CenterCrop(self.patch_size)
            image = center_crop(image)
            mask = center_crop(mask)

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