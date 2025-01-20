import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mit_semseg.lib.nn import train_collate

from mit_semseg.dataset import TrainDataset
from mit_semseg.models.segformer import Block
from mit_semseg.config import cfg
from mit_semseg.logging import *

class TestLogActivations(unittest.TestCase):
    def setUp(self):
        self.model = Block(
            dim=3, num_heads=1, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=1, sliding=False, kernel=16
        ).cuda()
        cfg.DATASET.list_train = "./data/training_10.odgt"
        cfg.DIR = "ckpt_test"
        self.cfg = cfg
        self.dataset = TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET)

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=5,
            shuffle=True,
            collate_fn=train_collate,
            num_workers=1,
            drop_last=True,
            pin_memory=True)
        
        self.activation_log = LogActivationGrad()
    def test(self):
        for epoch in range(2):
            self.activation_log.on_train_epoch_start(self.model)
            for i, batch_data in enumerate(self.loader):
                for k, v in batch_data.items():
                    batch_data[k] = batch_data[k].cuda()
                _, _, H, W = batch_data['img_data'].shape
                x = batch_data['img_data'].flatten(2).transpose(1, 2)
                output = self.model(x, H, W)
                loss = output.mean()
                self.activation_log.before_backward(
                    batch_data['img_data'],
                    batch_data['seg_label'],
                    loss,
                    f"epoch_{epoch}"
                )
            self.activation_log.on_train_epoch_end(self.model)
        self.activation_log.save_results(self.cfg)

if __name__ == 'main':
    unittest.main()