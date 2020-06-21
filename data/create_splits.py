from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
from utils import *


class GlasSplit():

    def get_mask(self, file, files):
        name = file.replace('.bmp', '') + '_anno'
        return list(filter(lambda f: name in f, files))[0]

    def get_label(self, file, csv):
        name = file.replace('.bmp', '')
        lbl = None
        for line in csv:
            if line[0] == name:
                lbl = line[2].strip()
                break
        return 0 if lbl == 'benign' else 1

    def write_splits_csv(self, name, files, masks, labels):
        with open(name, 'w') as out:
            filewriter = csv.writer(out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for file, mask, label in zip(files, masks, labels):
                filewriter.writerow([file, mask, label])

    def create_glas_split(self, val_size=0.2):

        files = get_files('../data/GlaS', 'bmp')
        csv = csv_reader('../data/GlaS/Grade.csv')

        train_files = list(filter(lambda f: 'train' in f and 'anno' not in f, files))
        test_files = list(filter(lambda f: 'test' in f and 'anno' not in f, files))
        train_files, val_files = train_test_split(train_files, test_size=val_size)

        train_masks = [self.get_mask(f, files) for f in train_files]
        test_masks = [self.get_mask(f, files) for f in test_files]
        val_masks = [self.get_mask(f, files) for f in val_files]

        train_labels = [self.get_label(f, csv) for f in train_files]
        test_labels = [self.get_label(f, csv) for f in test_files]
        val_labels = [self.get_label(f, csv) for f in val_files]

        base = '../data/splits/glas/{}'

        # self.write_splits_csv(base.format('train.csv'), train_files, train_masks, train_labels)
        # self.write_splits_csv(base.format('test.csv'), test_files, test_masks, test_labels)
        # self.write_splits_csv(base.format('val.csv'), val_files, val_masks, val_labels)


if __name__ == '__main__':

    GlasSplit().create_glas_split()