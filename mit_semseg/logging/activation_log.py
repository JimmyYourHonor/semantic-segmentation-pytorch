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
    def __init__(self):
        super().__init__()
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
        temp_idx = []
        if epoch == 'epoch_0':
            input_bytes = input[0].detach().cpu().numpy().tobytes()
            target_bytes = target[0].detach().cpu().numpy().tobytes()
            if (input_bytes, target_bytes) not in self.vis_set:
                self.vis_set[(input_bytes, target_bytes)] = {'epoch_0':{}}
            temp_idx.append(0)
        else:
            for i in range(input.shape[0]):
                input_bytes = input[i].detach().cpu().numpy().tobytes()
                target_bytes = target[i].detach().cpu().numpy().tobytes()
                if (input_bytes, target_bytes) in self.vis_set:
                    self.vis_set[(input_bytes, target_bytes)][epoch] = {}
                    temp_idx.append(i)
        for name, activations in self.features.items():
            for i in temp_idx:
                input_bytes = input[i].detach().cpu().numpy().tobytes()
                target_bytes = target[i].detach().cpu().numpy().tobytes()
                if name not in self.vis_set[(input_bytes,target_bytes)][epoch]:
                    self.vis_set[(input_bytes,target_bytes)][epoch][name] = []
            for act in activations:
                grad = torch.autograd.grad(loss, act, retain_graph=True)[0].detach().cpu().numpy()
                for i in temp_idx:
                    input_bytes = input[i].detach().cpu().numpy().tobytes()
                    target_bytes = target[i].detach().cpu().numpy().tobytes()
                    self.vis_set[(input_bytes,target_bytes)][epoch][name].append(grad[i])
    
    def save_checkpoint(self):
        return {
            "vis_set": self.vis_set,
        }
    
    def save_results(self, cfg):
        with PdfPages(os.path.join(cfg.DIR ,'activation_grad.pdf')) as pdf:
            plt.rcParams["figure.figsize"] = [7.00, 3.50] 
            plt.rcParams["figure.autolayout"] = True
            for key, value in self.vis_set.items():
                image_bytes, target_bytes = key
                image = np.frombuffer(image_bytes, dtype=np.float32).reshape([512, 512, 3])
                target = np.frombuffer(target_bytes, dtype=np.int64).reshape([512, 512])
                image = (min_max_normalize(image) * 255).astype(np.uint8)
                target = colorEncode(target, colors)
                fig1 = plt.figure()
                plt.imshow(np.concatenate((image, target),
                                        axis=1).astype(np.uint8))
                plt.show()
                pdf.savefig(fig1)

                for epoch, names in value.items():
                    total_act = 0
                    for _, activations in names.items():
                        total_act += len(activations)
                    fig, ax = plt.subplots((total_act + 3) // 4, 4)
                    fig.suptitle(epoch)
                    idx = 0
                    for name, activations in names.items():
                        for i, act in enumerate(activations):
                            x = idx // 4
                            y = idx % 4
                            idx += 1
                            ax[x,y].imshow(act)
                            ax[x,y].set_title(name + f"_{i}")
                    pdf.savefig(fig)

    def load_checkpoint(self, checkpoint):
        self.vis_set = checkpoint["vis_set"]