from pathlib import Path
import csv


def get_files(dir_path, ext):
    return [str(path.name) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def get_paths(dir_path, ext):
    return [str(path) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def parse_split_csv(csv_path):
    rows = []
    with open(csv_path, 'rb') as file:
        spamreader = csv.reader(file, delimiter=', ', quotechar='|')
        for row in spamreader:
            rows.append(row)
    return rows

def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out