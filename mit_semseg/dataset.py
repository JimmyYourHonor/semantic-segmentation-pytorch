import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def random_sample_ratio(img_scale, ratio_range):
    assert isinstance(img_scale, tuple) and len(img_scale) == 2
    min_ratio, max_ratio = ratio_range
    assert min_ratio <= max_ratio
    ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
    scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
    return scale

def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]]) -> tuple:
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)
    return new_size

def imrescale(
    img: Image,
    scale: Union[float, int, Tuple[int, int]],
    interpolation: str = 'bilinear',
) -> Image:
    w, h = img.size
    new_size = rescale_size((w, h), scale)
    rescaled_img = imresize(
        img, new_size, interp=interpolation)
    return rescaled_img

def get_crop_bbox(img, crop_size):
    """Randomly get a crop bounding box."""
    margin_h = max(img.shape[0] - crop_size[0], 0)
    margin_w = max(img.shape[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    return img

def random_crop(img, segm, crop_size, cat_max_ratio):
    crop_bbox = get_crop_bbox(img, crop_size)
    if cat_max_ratio < 1.:
        # Repeat 10 times
        for _ in range(10):
            seg_temp = crop(segm, crop_bbox)
            labels, cnt = np.unique(seg_temp, return_counts=True)
            cnt = cnt[labels != 0]
            if len(cnt) > 1 and np.max(cnt) / np.sum(
                    cnt) < cat_max_ratio:
                break
            crop_bbox = get_crop_bbox(img, crop_size)
    img = crop(img, crop_bbox)
    segm = crop(segm, crop_bbox)
    return img, segm

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(img) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(segm).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        # self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.imgScales = opt.imgScales
        self.imgRatio = opt.imgRatio
        self.imgCropSize = opt.imgCropSize
        self.cat_max_ratio = opt.cat_max_ratio
        # self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        #self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        # self.cur_idx = 0
        # self.if_shuffled = False

    # def _get_sub_batch(self):
    #     while True:
    #         # get a sample record
    #         this_sample = self.list_sample[self.cur_idx]
    #         if this_sample['height'] > this_sample['width']:
    #             self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
    #         else:
    #             self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

    #         # update current sample pointer
    #         self.cur_idx += 1
    #         if self.cur_idx >= self.num_sample:
    #             self.cur_idx = 0
    #             np.random.shuffle(self.list_sample)

    #         if len(self.batch_record_list[0]) == self.batch_per_gpu:
    #             batch_records = self.batch_record_list[0]
    #             self.batch_record_list[0] = []
    #             break
    #         elif len(self.batch_record_list[1]) == self.batch_per_gpu:
    #             batch_records = self.batch_record_list[1]
    #             self.batch_record_list[1] = []
    #             break
    #     return batch_records

    def __getitem__(self, index):
        this_sample = self.list_sample[index]
        image_path = os.path.join(self.root_dataset, this_sample['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_sample['fpath_segm'])

        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        # random_flip
        if np.random.choice([0, 1]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
        
        # random rescale
        scale = random_sample_ratio(self.imgScales, self.imgRatio)
        img = imrescale(img, scale, 'bilinear')
        segm = imrescale(segm, scale, 'nearest')
        
        # Convert to numpy
        img = np.array(img)
        segm = np.array(segm)

        # random crop
        img, segm = random_crop(img, segm, self.imgCropSize, self.cat_max_ratio)

        # Pad
        img_pad = np.zeros((self.imgCropSize[0], self.imgCropSize[1], 3), np.uint8)
        segm_pad = np.zeros((self.imgCropSize[0], self.imgCropSize[1]), np.uint8)
        img_pad[:img.shape[0], :img.shape[1], :] = img
        segm_pad[:segm.shape[0], :segm.shape[1]] = segm
        img = img_pad
        segm = segm_pad
        
        # image transform, to torch float tensor 3xHxW
        img = self.img_transform(img)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)

            # # put into batch arrays
            # batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            # batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = img
        output['seg_label'] = segm
        return output

        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        # if not self.if_shuffled:
        #     np.random.seed(index)
        #     np.random.shuffle(self.list_sample)
        #     self.if_shuffled = True

        # # get sub-batch candidates
        # batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        # if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
        #     this_short_size = np.random.choice(self.imgSizes)
        # else:
        #     this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        # batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        # batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        # for i in range(self.batch_per_gpu):
        #     img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
        #     this_scale = min(
        #         this_short_size / min(img_height, img_width), \
        #         self.imgMaxSize / max(img_height, img_width))
        #     batch_widths[i] = img_width * this_scale
        #     batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        # batch_width = np.max(batch_widths)
        # batch_height = np.max(batch_heights)
        # batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        # batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        # assert self.padding_constant >= self.segm_downsampling_rate, \
        #     'padding constant must be equal or large than segm downsamping rate'
        # batch_images = torch.zeros(
        #     self.batch_per_gpu, 3, batch_height, batch_width)
        # batch_segms = torch.zeros(
        #     self.batch_per_gpu,
        #     batch_height // self.segm_downsampling_rate,
        #     batch_width // self.segm_downsampling_rate).long()

        # for i in range(self.batch_per_gpu):
        #     this_record = batch_records[i]

        #     # load image and label
        #     image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        #     segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

        #     img = Image.open(image_path).convert('RGB')
        #     segm = Image.open(segm_path)
        #     assert(segm.mode == "L")
        #     assert(img.size[0] == segm.size[0])
        #     assert(img.size[1] == segm.size[1])

        #     # random_flip
        #     if np.random.choice([0, 1]):
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #         segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        #     # note that each sample within a mini batch has different scale param
        #     img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
        #     segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

        #     # further downsample seg label, need to avoid seg label misalignment
        #     segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
        #     segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
        #     segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
        #     segm_rounded.paste(segm, (0, 0))
        #     segm = imresize(
        #         segm_rounded,
        #         (segm_rounded.size[0] // self.segm_downsampling_rate, \
        #          segm_rounded.size[1] // self.segm_downsampling_rate), \
        #         interp='nearest')

        #     # image transform, to torch float tensor 3xHxW
        #     img = self.img_transform(img)

        #     # segm transform, to torch long tensor HxW
        #     segm = self.segm_transform(segm)

        #     # put into batch arrays
        #     batch_images[i][:, :img.shape[1], :img.shape[2]] = img
        #     batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        # output = dict()
        # output['img_data'] = batch_images
        # output['seg_label'] = batch_segms
        # return output

    def __len__(self):
        # return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        return self.num_sample


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
