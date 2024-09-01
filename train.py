# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from PIL import Image
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset, ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger, accuracy, intersectionAndUnion, colorEncode
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback, train_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
import matplotlib.pyplot as plt
import seaborn as sns

colors = loadmat('data/color150.mat')['colors']
def visualize_result(data, pred, savepath):
    (img, seg) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    Image.fromarray(im_vis).save(savepath)

def evaluate(segmentation_module, loader, cfg, epoch, gpu=0):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    for i, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if i%200 == 0:
            if not os.path.isdir(os.path.join(cfg.DIR, f"epoch_{epoch}")):
                    os.makedirs(os.path.join(cfg.DIR, f"epoch_{epoch}"))
            visualize_result(
                (batch_data['img_ori'], seg_label),
                pred,
                os.path.join(cfg.DIR, f"epoch_{epoch}", f'val_{i}.png')
            )
    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    

# train one epoch
def train(segmentation_module, loader, optimizers, ave_total_loss, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i, batch_data in enumerate(loader):
        # load a batch of data
        for k, v in batch_data.items():
          batch_data[k] = batch_data[k].cuda()
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * len(loader)
        adjust_learning_rate(optimizers, cur_iter, cfg)
        momentum_decay(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc, miou = segmentation_module(batch_data)
        loss = loss.mean()

        # Backward
        loss.backward()

        # Save gradients and weights before update to compute stats
        params_before = {}
        grads = {}
        for name, m in segmentation_module.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.conv._ConvNd):
                meta_name = ".".join(name.split(".")[:3])
                if meta_name not in params_before:
                    params_before[meta_name] = []
                if meta_name not in grads:
                    grads[meta_name] = []
                params_before[meta_name].append(m.weight.detach().cpu())
                grads[meta_name].append(m.weight.grad.detach().cpu())

        for optimizer in optimizers:
            optimizer.step()

        params_after = {}
        for name, m in segmentation_module.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.modules.conv._ConvNd):
                meta_name = ".".join(name.split(".")[:3])
                if meta_name not in params_after:
                    params_after[meta_name] = []
                params_after[meta_name].append(m.weight.detach().cpu())
        # Get weights after update to compute stats
        update_ratios = {}
        grads_ratios = {}
        for name in params_before.keys():
            param_before = torch.cat([param.flatten() for param in params_before[name]])
            param_after = torch.cat([param.flatten() for param in params_after[name]])
            grad = torch.cat([param.flatten() for param in grads[name]])
            update = param_after - param_before
            update_ratios[name] = (update.std() / param_after.std()).log10().data.item()
            grads_ratios[name] = nn.functional.cosine_similarity(grad, update, dim=0).data.item()


        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())

        # update union and intersection

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, momentum: {:.6f}, '
                  'Accuracy: {:4.2f}, miou: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, len(loader),
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder, cfg.TRAIN.running_beta1,
                          acc.data.item()*100, miou.data.item()*100, ave_total_loss.average()))

        fractional_epoch = epoch - 1 + 1. * i / len(loader)
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss.data.item())
        history['train']['acc'].append(acc.data.item())
        history['train']['miou'].append(miou.data.item())
        history['train']['update_ratios'].append(update_ratios)
        history['train']['grad_ratios'].append(grads_ratios)


def checkpoint(nets, history, cfg, epoch, encoder_opt=None, decoder_opt=None):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    dict_enc_opt = encoder_opt.state_dict()
    dict_dec_opt = decoder_opt.state_dict()

    torch.save({
            'encoder': dict_encoder,
            'decoder': dict_decoder,
            'encoder_opt': dict_enc_opt,
            'decoder_opt': dict_dec_opt,
            'history': history,
            'epoch': epoch,
            }, '{}/{}'.format(cfg.DIR, cfg.MODEL.checkpoint))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    if cfg.TRAIN.optim == 'SGD':
        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        optimizer_decoder = torch.optim.SGD(
            group_weight(net_decoder),
            lr=cfg.TRAIN.lr_decoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
    if cfg.TRAIN.optim == 'ADAMW':
        optimizer_encoder = torch.optim.AdamW(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            betas=[cfg.TRAIN.beta1, cfg.TRAIN.beta2],
            weight_decay=cfg.TRAIN.weight_decay)
        optimizer_decoder = torch.optim.AdamW(
            group_weight(net_decoder),
            lr=cfg.TRAIN.lr_decoder,
            betas=[cfg.TRAIN.beta1, cfg.TRAIN.beta2],
            weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder

def momentum_decay(optimizers, cur_iter, cfg):
    p = 1. - float(cur_iter) / cfg.TRAIN.max_iters
    cfg.TRAIN.running_beta1 = cfg.TRAIN.beta1 * (p / (1 - cfg.TRAIN.beta1 + cfg.TRAIN.beta1 * p))

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['momentum'] = cfg.TRAIN.running_beta1
    for param_group in optimizer_decoder.param_groups:
        param_group['momentum'] = cfg.TRAIN.running_beta1

def main(cfg):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
        use_pos_emb=cfg.MODEL.use_pos_em,
        sliding=cfg.MODEL.sliding)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        in_channels=cfg.MODEL.in_channels,
        embedding_dim=cfg.MODEL.embedding_dim,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
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
    
    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)
    
    cfg.TRAIN.max_iters = len(loader_train) * cfg.TRAIN.num_epoch
    print('1 Epoch = {} iters'.format(len(loader_train)))

    # create loader iterator
    # iterator_train = iter(loader_train)

    # load nets into gpu
    # if len(gpus) > 1:
    #     segmentation_module = UserScatteredDataParallel(
    #         segmentation_module,
    #         device_ids=gpus)
    #     # For sync bn
    #     patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'miou': [], 'update_ratios': [], 'grad_ratios': []}}

    if cfg.MODEL.checkpoint:
        ckpt_file = os.path.join(cfg.DIR, cfg.MODEL.checkpoint)
        if os.path.isfile(ckpt_file):
            state_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
            segmentation_module.encoder.load_state_dict(state_dict['encoder'])
            segmentation_module.decoder.load_state_dict(state_dict['decoder'])
            optimizers[0].load_state_dict(state_dict['encoder_opt'])
            optimizers[1].load_state_dict(state_dict['decoder_opt'])
            history = state_dict['history']
            cfg.TRAIN.start_epoch = state_dict['epoch']

    ave_total_loss = AverageMeter()
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, loader_train, optimizers, ave_total_loss, history, epoch+1, cfg)

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1, optimizers[0], optimizers[1])

        # plot training loss, acc, and miou
        plt.plot(history['train']['epoch'], history['train']['loss'])
        plt.savefig(os.path.join(cfg.DIR ,'train_loss.png'), bbox_inches='tight')
        plt.clf()
        plt.plot(history['train']['epoch'], history['train']['acc'])
        plt.savefig(os.path.join(cfg.DIR ,'train_acc.png'), bbox_inches='tight')
        plt.clf()
        plt.plot(history['train']['epoch'], history['train']['miou'])
        plt.savefig(os.path.join(cfg.DIR ,'train_miou.png'), bbox_inches='tight')
        plt.clf()
        colors = sns.color_palette('hls', len(history['train']['update_ratios'][0]))
        plt.gca().set_prop_cycle('color', colors)
        # Also plot update_ratios and grad_ratios
        legends = []
        for name in history['train']['update_ratios'][0].keys():
            plt.plot([history['train']['update_ratios'][j][name] for j in range(len(history['train']['update_ratios']))])
            legends.append(name)
        plt.plot([0, len(history['train']['update_ratios'])], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
        plt.legend(legends)
        plt.savefig(os.path.join(cfg.DIR ,'update_ratios.png'), bbox_inches='tight')
        plt.clf()

        colors = sns.color_palette('hls', len(history['train']['update_ratios'][0]))
        plt.gca().set_prop_cycle('color', colors)
        legends = []
        for name in history['train']['grad_ratios'][0].keys():
            temp_list = [history['train']['grad_ratios'][j][name] for j in range(len(history['train']['grad_ratios']))]
            plt.plot([sum(temp_list[i:i+10])/10 for i in range(10, len(temp_list))])
            legends.append(name)
        plt.legend(legends)
        plt.savefig(os.path.join(cfg.DIR ,'grad_ratios.png'), bbox_inches='tight')
        plt.clf()
        plt.close()

        if (epoch+1) % 10 == 0:
            # validation
            segmentation_module.eval()
            segmentation_module.decoder.use_softmax = True
            evaluate(segmentation_module, loader_val, cfg, epoch)
            segmentation_module.train()
            segmentation_module.decoder.use_softmax = False
            torch.cuda.empty_cache()

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
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
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    # if cfg.TRAIN.start_epoch > 0:
    #     cfg.MODEL.weights_encoder = os.path.join(
    #         cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
    #     cfg.MODEL.weights_decoder = os.path.join(
    #         cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
    #     assert os.path.exists(cfg.MODEL.weights_encoder) and \
    #         os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    # gpus = parse_devices(args.gpus)
    # gpus = [x.replace('gpu', '') for x in gpus]
    # gpus = [int(x) for x in gpus]
    # num_gpus = len(gpus)
    # cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    # cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    # cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    # cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg)
