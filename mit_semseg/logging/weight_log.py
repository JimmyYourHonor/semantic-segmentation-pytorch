import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from mit_semseg.logging.base_log import Log
from mit_semseg.utils import AverageMeter
from mit_semseg.models.lib.pos_emb import Image2DPositionalEncoding, RelativePositionalEncoding

class LogWeight(Log):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.params_before = {}
        self.params_after = {}
        self.grads = {}
        self.update_ratios_avg = {}
        self.grad_ratios_avg = {}
        self.update_ratios_avgs = []
        self.grad_ratios_avgs = []

    def on_train_epoch_start(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.conv._ConvNd):
                meta_name = ".".join(name.split(".")[:3])
                self.update_ratios_avg[meta_name] = AverageMeter()
                self.grad_ratios_avg[meta_name] = AverageMeter()

    def before_optim(self, **kwargs):
        self.params_before = {}
        self.grads = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.conv._ConvNd):
                meta_name = ".".join(name.split(".")[:3])
                if meta_name not in self.params_before:
                    self.params_before[meta_name] = []
                if meta_name not in self.grads:
                    self.grads[meta_name] = []
                self.params_before[meta_name].append(m.weight.detach().cpu())
                self.grads[meta_name].append(m.weight.grad.detach().cpu())

    def after_optim(self):
        self.params_after = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.conv._ConvNd):
                meta_name = ".".join(name.split(".")[:3])
                if meta_name not in self.params_after:
                    self.params_after[meta_name] = []
                self.params_after[meta_name].append(m.weight.detach().cpu())
            elif isinstance(m, Image2DPositionalEncoding):
                meta_name = ".".join(name.split(".")[:3])
                if meta_name not in self.params_after:
                    self.params_after[meta_name] = []
                self.params_after[meta_name].append(m.h_embedding.detach().cpu())
                self.params_after[meta_name].append(m.w_embedding.detach().cpu())
            elif isinstance(m, RelativePositionalEncoding):
                meta_name = ".".join(name.split(".")[:3])
                if meta_name not in self.params_after:
                    self.params_after[meta_name] = []
                self.params_after[meta_name].append(m.relative_position_bias_table.detach().cpu())

        for name in self.params_before.keys():
            param_before = torch.cat([param.flatten() for param in self.params_before[name]])
            param_after = torch.cat([param.flatten() for param in self.params_after[name]])
            grad = torch.cat([param.flatten() for param in self.grads[name]])
            update = param_after - param_before
            update_ratio = (update.std() / param_after.std()).log10().data.item()
            grad_ratio = nn.functional.cosine_similarity(grad, update, dim=0).data.item()
            self.grad_ratios_avg[name].update(grad_ratio)
            self.update_ratios_avg[name].update(update_ratio)

    def on_train_epoch_end(self):
        self.grad_ratios_avgs.append({})
        self.update_ratios_avgs.append({})
        for name in self.grad_ratios_avg.keys():
            self.grad_ratios_avgs[-1][name] = self.grad_ratios_avg[name].average()
            self.update_ratios_avgs[-1][name] = self.update_ratios_avg[name].average()

    def save_checkpoint(self):
        return {
            "grad_ratio_avgs" : self.grad_ratios_avgs,
            "update_ratio_avgs": self.update_ratios_avgs
        }
    
    def load_checkpoint(self, checkpoint):
        self.grad_ratios_avgs = checkpoint["grad_ratio_avgs"]
        self.update_ratios_avgs = checkpoint["update_ratio_avgs"]

    def save_results(self, cfg):
        colors = sns.color_palette('hls', len(self.update_ratios_avgs[0]))
        plt.gca().set_prop_cycle('color', colors)
        legends = []
        for name in self.update_ratios_avgs[0].keys():
            plt.plot([self.update_ratios_avgs[j][name] for j in range(len(self.update_ratios_avgs))])
            legends.append(name)
        plt.plot([0, len(self.update_ratios_avgs)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
        plt.legend(legends, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(os.path.join(cfg.DIR ,'update_ratios.png'), bbox_inches='tight')
        plt.clf()

        colors = sns.color_palette('hls', len(self.grad_ratios_avgs[0]))
        plt.gca().set_prop_cycle('color', colors)
        legends = []
        for name in self.grad_ratios_avgs[0].keys():
            plt.plot([self.grad_ratios_avgs[j][name] for j in range(len(self.grad_ratios_avgs))])
            legends.append(name)
        plt.legend(legends, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(os.path.join(cfg.DIR ,'grad_ratios.png'), bbox_inches='tight')
        plt.clf()
        plt.close()