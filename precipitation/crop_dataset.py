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


def get_rainfall_crop_dataset(config, uniform_dequantization=False, evaluation=False):
    if uniform_dequantization:
        raise NotImplementedError('Uniform dequantization not yet supported.')
    if evaluation:
        raise NotImplementedError('Evaluation not yet supported.')
    high_res_dim = [config.data.image_size, config.data.image_size]
    train_val_dataset = RainFallCropDataset(config.data.dataset_path, res_ratio=config.data.resolution_ratio,
                                            high_res_dim=high_res_dim)
    dataset_size = len(train_val_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.data.train_val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    generator = torch.Generator().manual_seed(config.seed)
    train_sampler = SubsetRandomSampler(train_indices, generator=generator)
    val_sampler = SubsetRandomSampler(val_indices, generator=generator)

    train_loader = DataLoader(train_val_dataset, batch_size=config.data.batch_size,
                              num_workers=config.hardware.num_workers, sampler=train_sampler)

    val_loader = DataLoader(train_val_dataset, batch_size=config.data.batch_size,
                            num_workers=config.hardware.num_workers, sampler=val_sampler)

    return train_loader, val_loader


class RainFallCropDataset(Dataset):
    def __init__(self, path, res_ratio, high_res_dim=(512, 512)):
        super().__init__()
        self.path = path
        self.res_ratio = res_ratio
        files = os.listdir(self.path)
        self.files = natsorted([f for f in files if '.npy' in f])

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
        files = os.listdir(self.path)
        return len([f for f in files if '.npy' in f])

    def __getitem__(self, item):
        file = self.files[item]
        arr = np.load(p.join(self.path, file))
        arr[np.isnan(arr)] = -1
        arr[arr < 0.] = 0.
        high_res_img = self.normalize(arr)
        low_res_img = self.down_sample(high_res_img)
        high_res_t = torch.from_numpy(high_res_img)
        low_res_t = torch.from_numpy(low_res_img)
        low_res_t, high_res_t = low_res_t.unsqueeze(dim=0), high_res_t.unsqueeze(dim=0)
        return low_res_t, high_res_t

    def down_sample(self, img):
        # generates a low resolution image from high resolution ground truth
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
        normalized = scaled
        normalized = scaled / 5.  # scale to 0 to 1 TODO not scientific
        normalized[normalized < self.theta] = 0  # clip lower end at theta
        return normalized
