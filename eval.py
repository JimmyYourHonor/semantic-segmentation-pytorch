# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = [AverageMeter() for _ in cfg.DATASET.imgSizes]
    acc_meter.append(AverageMeter())
    intersection_meter = [AverageMeter() for _ in cfg.DATASET.imgSizes]
    intersection_meter.append(AverageMeter())
    union_meter = [AverageMeter() for _ in cfg.DATASET.imgSizes]
    union_meter.append(AverageMeter())
    time_meter = AverageMeter()
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
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

            for i, img in enumerate(img_resized_list):
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

                _, pred = torch.max(scores_tmp, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
                
                # calculate accuracy
                acc, pix = accuracy(pred, seg_label)
                intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
                acc_meter[i].update(acc, pix)
                intersection_meter[i].update(intersection)
                union_meter[i].update(union)
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            
            # calculate accuracy
            acc, pix = accuracy(pred, seg_label)
            intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
            acc_meter[-1].update(acc, pix)
            intersection_meter[-1].update(intersection)
            union_meter[-1].update(union)

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)

    # summary
    iou = intersection_meter[-1].sum / (union_meter[-1].sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter[-1].average()*100, time_meter.average()))
    
    if len(cfg.DATASET.imgSizes) > 1:
        acc_arr = np.array([acc_meter[i].average()*100 for i in range(len(cfg.DATASET.imgSizes))])
        miou_arr = np.array([(intersection_meter[i].sum / (union_meter[i].sum + 1e-10)).mean() 
                             for i in range(len(cfg.DATASET.imgSizes))])
        np.save(os.path.join(cfg.DIR, "result", "accuracy"), acc_arr)
        np.save(os.path.join(cfg.DIR, "result", "miou"), miou_arr)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
        use_pos_emb=cfg.MODEL.use_pos_emb,
        sliding=cfg.MODEL.sliding,
        kernels=cfg.MODEL.kernels)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        in_channels=cfg.MODEL.in_channels,
        embedding_dim=cfg.MODEL.embedding_dim,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # if cfg.MODEL.pretrained_segformer is not None:
    #     # Load segformer
    #     pretrained_weight = torch.load(
    #         cfg.MODEL.pretrained_segformer, 
    #         map_location=lambda storage, loc: storage)['state_dict']
    #     new_pretrained_weight = {}
    #     for k in pretrained_weight.keys():
    #         if 'backbone' in k:
    #             new_k = k.replace('backbone', 'encoder')
    #         if 'decode_head' in k:
    #             if 'conv_seg' in k:
    #                 continue
    #             elif k == 'decode_head.linear_fuse.conv.weight':
    #                 new_k = 'decoder.linear_fuse.0.weight'
    #             elif k == 'decode_head.linear_fuse.bn.weight':
    #                 new_k = 'decoder.linear_fuse.1.weight'
    #             elif k == 'decode_head.linear_fuse.bn.bias':
    #                 new_k = 'decoder.linear_fuse.1.bias'
    #             elif k == 'decode_head.linear_fuse.bn.running_mean':
    #                 new_k = 'decoder.linear_fuse.1.running_mean'
    #             elif k == 'decode_head.linear_fuse.bn.running_var':
    #                 new_k = 'decoder.linear_fuse.1.running_var'
    #             elif k == 'decode_head.linear_fuse.bn.num_batches_tracked':
    #                 continue
    #             else:
    #                 new_k = k.replace('decode_head', 'decoder')
    #         new_pretrained_weight[new_k] = pretrained_weight[k]
    #     del pretrained_weight
    #     segmentation_module.load_state_dict(new_pretrained_weight, strict=False)
    if cfg.MODEL.checkpoint:
        ckpt_file = os.path.join(cfg.DIR, cfg.MODEL.checkpoint)
        if os.path.isfile(ckpt_file):
            state_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
            segmentation_module.encoder.load_state_dict(state_dict['encoder'])
            segmentation_module.decoder.load_state_dict(state_dict['decoder'])

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

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
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

    # absolute paths of model weights
    # cfg.MODEL.weights_encoder = os.path.join(
    #     cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    # cfg.MODEL.weights_decoder = os.path.join(
    #     cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    # assert os.path.exists(cfg.MODEL.weights_encoder) and \
    #     os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
