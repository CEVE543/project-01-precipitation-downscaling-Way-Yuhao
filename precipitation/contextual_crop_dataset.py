__author__ = 'yuhao liu'

import os
import os.path as p
import xarray as xr
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
# import pytorch_lightning as pl
# from pytorch_lightning import LightningModule, LightningDataModule, Trainer
# from pytorch_lightning.loggers import WandbLogger
import wandb
from natsort import natsorted
import re
from util.helper import deprecated


def get_rainfall_contextual_crop_dataset(config, uniform_dequantization=False, evaluation=False):
    if uniform_dequantization:
        raise NotImplementedError('Uniform dequantization not yet supported.')
    if evaluation:
        raise NotImplementedError('Evaluation not yet supported.')
    high_res_dim = [config.data.image_size, config.data.image_size]
    train_val_dataset = ContextualCropDataset(config.data.dataset_path, res_ratio=config.data.resolution_ratio,
                                            high_res_dim=high_res_dim)
    dataset_size = len(train_val_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.data.train_val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    generator = torch.Generator().manual_seed(config.seed)
    train_sampler = SubsetRandomSampler(train_indices, generator=generator)
    val_sampler = SubsetRandomSampler(val_indices, generator=generator)

    train_loader = DataLoader(train_val_dataset, batch_size=config.data.batch_size,
                              num_workers=config.hardware.num_workers, sampler=train_sampler)

    val_loader = DataLoader(train_val_dataset, batch_size=config.data.batch_size,
                            num_workers=config.hardware.num_workers, sampler=val_sampler)

    return train_loader, val_loader


def stack_conditional_layers(**kwargs):
    """
    Stacks the conditional layers into a single tensor.
    :param kwargs:
    :return:
    """
    conditional_layers = []
    for k, v in kwargs.items():
        if k == 'precip_hr':
            pass
        conditional_layers.append(v)
    conditional_layers = torch.cat(conditional_layers, dim=1)
    return conditional_layers

class ContextualCropDataset(Dataset):

    def __init__(self, path, res_ratio, high_res_dim=(512, 512)):
        super().__init__()
        self.path = path
        self.res_ratio = res_ratio
        files = os.listdir(self.path)
        self.all_files = natsorted([f for f in files if '.npy' in f])
        self.rainfall_files = natsorted([f for f in self.all_files if 'precip' in f])

        self.res_ratio = res_ratio  # resolution ration between high-res and low-res images, in length
        self.high_res_dim = high_res_dim  # (h, w) of high-res image, after random crop
        self.low_res_dim = (high_res_dim[0] / self.res_ratio, high_res_dim[1] / self.res_ratio)
        assert self.low_res_dim[0].is_integer()  # checks divisibility
        assert self.low_res_dim[1].is_integer()
        self.low_res_dim = (int(self.low_res_dim[0]), int(self.low_res_dim[1]))
        self.transform = None  # TODO
        self.xmin, self.xmax = 0, 2326.7  # empirically acquired, raw min/max pixel values from dataset
        self.theta = 0.17  # following https://ieeexplore.ieee.org/document/9246532
        self.eps = 0.00001  # for numerical precision

    def __len__(self):
        return len(self.rainfall_files)

    @staticmethod
    def _parse_file_name(filename):
        """
        Extract the frame and crop number from the filename.
        """
        # Use a regular expression to match the pattern and extract X and Y
        match = re.match(r'precip_f(\d+)_c(\d+).npy', filename)

        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return x, y
        else:
            raise ValueError("Filename does not match the expected pattern.")

    def __getitem__(self, item):
        """
        Returns a batch of data.
        Each batch contains the following:
            surf_temp: ERA5 2-meter air temperature image
            elevation: ERA5 elevation image
            vflux_e: ERA5 eastward vapor flux image
            vflux_n: ERA5 northward vapor flux image
            wind_u: ERA5 u component of wind image
            wind_v: ERA5 v component of wind image
            precip_lr: low resolution MRMS precipitation image
            precip_up: naively upsampled low resolution MRMS precipitation image via CV2.INTER_LINEAR
            precip_gt: ground-truth high resolution MRMS precipitation image
        All images are torch tensors of dimension (1, h, w).
        """
        rainfall_file = self.rainfall_files[item]
        X, Y = self._parse_file_name(rainfall_file)
        # handling rainfall file
        precip = np.load(p.join(self.path, rainfall_file))
        surf_temp = np.load(p.join(self.path, f'2mt_f{X}_c{Y}.npy'))
        elevation = np.load(p.join(self.path, f'elevation_f{X}_c{Y}.npy'))
        vflux_e = np.load(p.join(self.path, f'vflux_e_f{X}_c{Y}.npy'))
        vflux_n = np.load(p.join(self.path, f'vflux_n_f{X}_c{Y}.npy'))
        wind_u = np.load(p.join(self.path, f'u_f{X}_c{Y}.npy'))
        wind_v = np.load(p.join(self.path, f'v_f{X}_c{Y}.npy'))

        batch = {'precip': precip, 'surf_temp': surf_temp, 'elevation': elevation,
                 'vflux_e': vflux_e, 'vflux_n': vflux_n, 'wind_u': wind_u, 'wind_v': wind_v}

        batch = self.correct(**batch)
        high_res_img = self.normalize(batch['precip'])
        low_res_img, upsampled_img = self.down_sample(high_res_img)
        batch['precip_lr'] = low_res_img
        batch['precip_up'] = upsampled_img
        batch['precip_gt'] = high_res_img
        batch = self.cvt_to_tensor(**batch)
        del batch['precip']
        # high_res_t = torch.from_numpy(high_res_img)
        # low_res_t = torch.from_numpy(low_res_img)
        # low_res_t, high_res_t = low_res_t.unsqueeze(dim=0), high_res_t.unsqueeze(dim=0)

        return batch

    @staticmethod
    def correct(**kwargs) -> dict:
        for k, arr in kwargs.items():
            arr[np.isnan(arr)] = -1
            arr[arr < 0.] = 0.
            kwargs[k] = arr
        return kwargs


    def down_sample(self, img):
        ratio = self.res_ratio
        h, w = img.shape[0], img.shape[1]
        assert (h / ratio).is_integer(), f'The height of the high-res image ({h}) is not divisible by ratio ({ratio})'
        assert (h / ratio).is_integer(), f'The width of the high-res image ({w}) is not divisible by ratio ({ratio})'
        low_res_dim = (int(h / ratio), int(w / ratio))
        low_res_img = cv2.resize(img, low_res_dim, interpolation=cv2.INTER_AREA)
        upsampled_img = cv2.resize(low_res_img, (h, w), interpolation=cv2.INTER_LINEAR)
        return low_res_img, upsampled_img


    @deprecated
    def down_sample_old(self, img):
        # generates a low resolution image from high resolution ground truth
        # DEPRECATED: this function reruns a course-grid like low-res input
        # whose actual dimension matches with the ground truth.
        ratio = self.res_ratio
        h, w = img.shape[0], img.shape[1]
        assert (h / ratio).is_integer(), f'The height of the high-res image ({h}) is not divisible by ratio ({ratio})'
        assert (h / ratio).is_integer(), f'The width of the high-res image ({w}) is not divisible by ratio ({ratio})'
        low_res_dim = (int(h / ratio), int(w / ratio))
        low_res_img = cv2.resize(img, low_res_dim, interpolation=cv2.INTER_AREA)
        low_res_img = cv2.resize(low_res_img, (h, w), interpolation=cv2.INTER_NEAREST)
        return low_res_img

    def normalize(self, img):
        """
        Normalizes input data by taking log. Maps valid pixel input [xmin, xmax] to [theta, 1].
        Maps invalid pixels to 0.
        :param img:
        :return:
        """
        # scale by taking log
        valid_region = img != -1
        scaled = np.zeros_like(img)
        scaled[valid_region] = np.log(img[valid_region] + self.eps)
        # normalized = scaled
        normalized = scaled / 5.  # scale to 0 to 1 TODO not scientific
        normalized[normalized < self.theta] = 0  # clip lower end at theta
        return normalized

    @staticmethod
    def cvt_to_tensor(**kwargs) -> dict:
        for k, arr in kwargs.items():
            t = torch.from_numpy(arr)
            kwargs[k] = t.unsqueeze(dim=0)
        return kwargs
