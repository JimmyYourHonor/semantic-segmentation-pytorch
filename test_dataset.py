import argparse
from scipy.io import loadmat
import csv
import numpy as np
from mit_semseg.config import cfg
from mit_semseg.lib.nn import train_collate
from mit_semseg.utils import colorEncode
import matplotlib.pyplot as plt
import torch
from mit_semseg.dataset import TrainDataset

if __name__ == '__main__':

    # Dataset and Loader
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-segformer-test.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.batch_size_per_gpu,  # we have modified data_parallel
        shuffle=True,  # we do not use this param
        collate_fn=train_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    
    colors = loadmat('data/color150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    def visualize_result(data, pred):
        img = np.float32(data)
        img = img*np.array([[[0.229]], [[0.224]], [[0.225]]]) + np.array([[[0.485]], [[0.456]], [[0.406]]])

        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        for idx in np.argsort(counts)[::-1]:
            if uniques[idx] == -1:
                continue
            name = names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

        # colorize prediction
        pred_color = colorEncode(pred, colors).astype(np.uint8)

        # aggregate images and save
        im_vis = np.concatenate((img.transpose((1,2,0)), pred_color/255.), axis=1)
        plt.imshow(im_vis)
        plt.show()

    torch.manual_seed(0)
    for i, batch in enumerate(loader_train):
        for j in range(cfg.TRAIN.batch_size_per_gpu):
            img = batch['img_data'][j, ...]
            segm = batch['seg_label'][j, ...]
            visualize_result(data=img, pred=segm)