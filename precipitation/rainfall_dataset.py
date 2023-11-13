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

def get_rainfall_dataset(config, uniform_dequantization=False, evaluation=False):
    if uniform_dequantization:
        raise NotImplementedError('Uniform dequantization not yet supported.')
    if evaluation:
        raise NotImplementedError('Evaluation not yet supported.')
    high_res_dim = [config.data.image_size, config.data.image_size]
    train_val_dataset = RainFallDataset(config.data.dataset_path, res_ratio=config.data.resolution_ratio,
                                        high_res_dim=high_res_dim, crop_retry=config.data.crop_retry)
    dataset_size = len(train_val_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.data.train_val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # TODO: consider implementing dynamic resource allocator like tf
    train_loader = DataLoader(train_val_dataset, batch_size=config.data.batch_size,
                              num_workers=config.hardware.num_workers, sampler=train_sampler)

    val_loader = DataLoader(train_val_dataset, batch_size=config.data.batch_size,
                            num_workers=config.hardware.num_workers, sampler=val_sampler)

    return train_loader, val_loader



class RainFallDataset(Dataset):
    def __init__(self, root, res_ratio=16, high_res_dim=(512, 512), crop_retry=0):
        super().__init__()
        self.root = root
        self.ds = self.scan_xarray_dataset()  # xarray.Dataset of dim [time, lon, lat, alt]
        self.da = self.ds.to_array()  # xarray.DataArray of dim [var, time, lon, lat, alt]
        assert self.ds is not None
        assert self.da is not None

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
        # theta is a value at which minimum-detectable reading maps to after all scaling/normalizations
        self.crop_retry = crop_retry
        assert self.crop_retry >= 1, 'crop_retry must be greater than or equal to 1'

    def __len__(self):
        return self.ds.sizes['time']  # number of hourly readings

    def __getitem__(self, item):
        arr = self.da[0, item, 0, :, :]  # 2D tensor of shape [lon, alt]
        arr = arr.to_numpy()
        arr[np.isnan(arr)] = -1
        arr[arr < 0.] = 0.
        high_res_img = self.rand_crop(arr)
        high_res_img = self.normalize(high_res_img)
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
        # normalized[~valid_region] = -1

        # below_threshold_region = scaled < self.theta
        # below_threshold_region = np.bitwise_and(below_threshold_region, valid_region)
        # scaled[below_threshold_region] = self.theta  # maps to [theta, log(xmax)]
        #
        # # normalize to [theta, 1]
        # normalized = (scaled - self.theta) / (np.log(self.xmax) - self.theta)
        # normalized[~valid_region] = 0
        return normalized  # TODO: test this

    def rand_crop(self, img):
        crop_width, crop_height = self.high_res_dim
        max_h = img.shape[0] - crop_height
        max_w = img.shape[1] - crop_width
        min_detected_perc = .1
        tries = 0
        threshold_satisfied = False
        crop_img = None
        while (not threshold_satisfied) and tries < self.crop_retry:
            h = np.random.randint(0, max_h)
            w = np.random.randint(0, max_w)
            crop_img = img[h: h + crop_height, w: w + crop_width]
            detected_perc = np.count_nonzero(crop_img) / (crop_img.shape[0] * crop_img.shape[1])
            if detected_perc >= min_detected_perc:
                threshold_satisfied = True
            tries += 1
        return crop_img

    def smooth_filter(self):
        # apply gaussian filter to suppress artifacts around edges
        raise NotImplementedError

    def scan_xarray_dataset(self):
        files = os.listdir(self.root)
        files = [f for f in files if '.nc' in f]
        print(f'scanning xarray dataset ({len(files)} frames)...', end=' ')
        pre_merge = [p.join(self.root, f) for f in files if 'GaugeCorr' in f]
        post_merge = [p.join(self.root, f) for f in files if 'MultiSensor' in f]
        ds_pre, ds_post, ds = None, None, None
        if len(pre_merge) != 0:
            ds_pre = (
                xr.open_mfdataset(
                    paths=pre_merge,
                    combine="nested",
                    concat_dim="time",
                    chunks={"time": 10},
                    parallel=False
                )
                .sortby("time")
                .rename({"param9.6.209": "prcp_rate"}))
        if len(post_merge) != 0:
            ds_post = (
                xr.open_mfdataset(
                    paths=post_merge,
                    combine="nested",
                    concat_dim="time",
                    chunks={"time": 10},
                    parallel=False
                )
                .sortby("time")
                .rename({"param37.6.209": "prcp_rate"}))
        if ds_pre is not None and ds_post is not None:
            ds = xr.concat([ds_pre, ds_post], dim='time')
        elif ds_pre is None:
            ds = ds_post
        elif ds_post is None:
            ds = ds_pre
        print('Done!')
        return ds


# class RainFallDataModule(LightningDataModule):
#
#     def __init__(self, data_params, train_val_split, num_workers):
#         super().__init__()
#         self.data_dir = data_params['dataset_path']
#         self.batch_size = data_params['batch_size']
#         self.res_ratio = data_params['resolution_ratio']
#         self.high_res_dim = data_params['high_res_dim']
#         self.crop_retry = data_params['crop_retry']
#         self.transform = None  # TODO
#         self.train_val_dataset = None
#         self.test_dataset = None
#         self.train_val_split = train_val_split
#         self.train_sampler = None
#         self.val_sampler = None
#         self.num_workers = num_workers
#
#     def prepare_data(self):
#         return  # nothing to do?
#
#     def setup(self, stage=None):
#         if stage == 'fit':
#             self.train_val_dataset = RainFallDataset(self.data_dir, res_ratio=self.res_ratio,
#                                                      high_res_dim=self.high_res_dim, crop_retry=self.crop_retry)
#             dataset_size = len(self.train_val_dataset)
#             indices = list(range(dataset_size))
#             split = int(np.floor(self.train_val_split * dataset_size))
#             train_indices, val_indices = indices[split:], indices[:split]
#             self.train_sampler = SubsetRandomSampler(train_indices)
#             self.val_sampler = SubsetRandomSampler(val_indices)
#         if stage == 'test':
#             raise NotImplementedError()
#
#     def train_dataloader(self):
#         train_loader = DataLoader(self.train_val_dataset, batch_size=self.batch_size,
#                                   num_workers=self.num_workers, sampler=self.train_sampler)
#         return train_loader
#
#     def val_dataloader(self):
#         val_loader = DataLoader(self.train_val_dataset, batch_size=self.batch_size,
#                                 num_workers=self.num_workers, sampler=self.val_sampler)
#         return val_loader
#
#     def test_dataloader(self):
#         raise NotImplementedError()