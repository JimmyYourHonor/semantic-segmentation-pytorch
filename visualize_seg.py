# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate #, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_seg(seg):
    pixs = seg.size
    uniques, counts = np.unique(seg, return_counts=True)
    print(uniques)
    for idx in np.argsort(counts)[::-1]:
        if uniques[idx] == -1:
            continue
        name = names[uniques[idx]+1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
    seg_color = colorEncode(seg, colors).astype(np.uint8)
    Image.fromarray(seg_color).save('seg.png')

if __name__ == '__main__':
    segm = Image.open('ADE_val_00000001.png')
    segm = torch.from_numpy(np.array(segm)).long() - 1
    segm = as_numpy(segm)
    visualize_seg(segm)