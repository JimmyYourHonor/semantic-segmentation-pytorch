import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
import seaborn as sns
import os

from mit_semseg.logging.base_log import Log
from mit_semseg.logging.utils import get_activation_hook, min_max_normalize
from mit_semseg.utils import colorEncode
colors = loadmat('data/color150.mat')['colors']

class LogActivationGrad(Log):
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold
        self.hooks = []
        self.features = {}
        self.vis_set = {}

    def on_train_epoch_start(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.conv._ConvNd):
                hook_fn = get_activation_hook(self.features, name)
                self.hooks.append(m.register_forward_pre_hook(hook_fn))
    
    def on_train_epoch_end(self, model):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
        
    def before_backward(self, input, target, loss, epoch):
        idx = np.random.rand(input.shape[0]) > self.threshold
        if epoch not in self.vis_set:
            self.vis_set[epoch] = {}
            self.vis_set[epoch]['input'] = input[idx].detach().cpu().numpy()
            self.vis_set[epoch]['target'] = target[idx].detach().cpu().numpy()
        else:
            self.vis_set[epoch]['input'] = np.concatenate(
                [self.vis_set[epoch]['input'],
                input[idx].detach().cpu().numpy()],
                axis=0
            )
            self.vis_set[epoch]['target'] = np.concatenate(
                [self.vis_set[epoch]['target'],
                target[idx].detach().cpu().numpy()],
                axis=0
            )
        for name, activations in self.features.items():
            if name not in self.vis_set[epoch]:
                self.vis_set[epoch][name] = []
            for i, act in enumerate(activations):
                grad = torch.autograd.grad(loss, act, retain_graph=True)[0]
                if len(grad.shape) == 3:
                    grad = torch.mean(grad, dim=-1).detach().cpu().numpy()
                elif len(grad.shape) == 4:
                    grad = torch.mean(grad, dim=1).detach().cpu().numpy()
                if i == len(self.vis_set[epoch][name]):
                    self.vis_set[epoch][name].append(grad[idx])
                else:
                    self.vis_set[epoch][name][i] = np.concatenate(
                        [self.vis_set[epoch][name][i], grad[idx]],
                        axis=0
                    )
    
    def save_checkpoint(self):
        return {
            "vis_set": self.vis_set,
        }
    
    def save_results(self, cfg):
        with PdfPages(os.path.join(cfg.DIR ,'activation_grad.pdf')) as pdf:
            plt.rcParams["figure.figsize"] = [7.00, 3.50] 
            plt.rcParams["figure.autolayout"] = True
            for epoch in self.vis_set:
                image_set = min_max_normalize(self.vis_set[epoch]['input'])
                target_set = self.vis_set[epoch]['target']
                for i in range(image_set.shape[0]):
                    image = (image_set[i] * 255).astype(np.uint8)
                    image = image.transpose(1,2,0)
                    target = colorEncode(target_set[i], colors)
                    fig1 = plt.figure()
                    plt.imshow(np.concatenate((image, target),
                                            axis=1).astype(np.uint8))
                    plt.show()
                    pdf.savefig(fig1)
                    total_act = 0
                    for name in self.vis_set[epoch]:
                        if name in ['input', 'target']:
                          continue
                        total_act += len(self.vis_set[epoch][name])
                    fig, ax = plt.subplots((total_act + 3) // 4, 4)
                    fig.suptitle(epoch)
                    idx = 0
                    for name in self.vis_set[epoch]:
                        if name in ['input', 'target']:
                          continue
                        for j, activations in enumerate(self.vis_set[epoch][name]):
                            act = activations[i]
                            x = idx // 4
                            y = idx % 4
                            idx += 1
                            if len(act.shape) == 1:
                                act = act.reshape(
                                int(np.sqrt(act.shape[0])),
                                int(np.sqrt(act.shape[0]))
                                )
                            ax[x,y].imshow(min_max_normalize(act))
                            ax[x,y].set_title(name + f"_{j}")
                    pdf.savefig(fig)

    def load_checkpoint(self, checkpoint):
        self.vis_set = checkpoint["vis_set"]