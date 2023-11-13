"""
This script takes an existing dataset and separates each frame into crops of size crop_size x crop_size.
The script will save unscaled [0, ?] npy files and some example scaled [0, 1] png files.
"""
__author__ = 'yuhao liu'

import os
import os.path as p
import xarray as xr
import numpy as np
import cv2
import wandb
from matplotlib import pyplot as plt
from rich.progress import track, Progress, TextColumn, \
    BarColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn

dataset_path = '/home/yl241/data/CLIMATE/nexrad_min_0.2'
crop_path = '/home/yl241/data/CLIMATE/nexrad_min_0.2_crops'
# contextual_crop_path = '/home/yl241/data/CLIMATE/nexrad_min_0.2_contextual_crops'
contextual_crop_path = '/home/yl241/data/CLIMATE/nexrad_min_0.2_crops'
# ERA5 data
era5_vars_path = {
    # single level
    '2mt': "/home/yl241/data/CLIMATE/ERA5/single_level/2m_temperature_{}.nc",
    'elevation': "/home/yl241/data/CLIMATE/ERA5/single_level/elevation.nc",
    'vflux_e': "/home/yl241/data/CLIMATE/ERA5/single_level/vertical_integral_of_eastward_water_vapour_flux_{}.nc",
    'vflux_n': "/home/yl241/data/CLIMATE/ERA5/single_level/vertical_integral_of_northward_water_vapour_flux_{}.nc",
    # pressure level
    'u': "/home/yl241/data/CLIMATE/ERA5/pressure_level/u_component_of_wind_500_{}.nc",
    'v': "/home/yl241/data/CLIMATE/ERA5/pressure_level/v_component_of_wind_500_{}.nc",
}

crop_size = 256
lon_limit = slice(-125, -65)  # limited by available ERA5 data
lat_limit = slice(50, 25)  # limited by available ERA5 data


def scan_xarray_dataset():
    files = os.listdir(dataset_path)
    files = [f for f in files if '.nc' in f]
    print(f'scanning xarray dataset ({len(files)} frames)...', end=' ')
    pre_merge = [p.join(dataset_path, f) for f in files if 'GaugeCorr' in f]
    post_merge = [p.join(dataset_path, f) for f in files if 'MultiSensor' in f]
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


def segment_image(crop_dim, **layers):
    """
    Segment an image into crops of size crop_h x crop_w. If the entire canvas is not
    evenly divisible by the crop size, the remaining pixels will be discarded.
    """

    # Get the image dimensions
    H, W = layers['precip'].shape
    crop_h, crop_w = crop_dim

    # Calculate the number of crops in each dimension
    num_crops_h = H // crop_h
    num_crops_w = W // crop_w

    crop_layers = {}
    for k, image in layers.items():
        # Crop the image to be evenly divisible by the crop size
        cropped_image = image[:num_crops_h * crop_h, :num_crops_w * crop_w]

        # Reshape the image so each crop_h x crop_w chunk is a separate block
        reshaped = cropped_image.reshape(num_crops_h, crop_h, -1, crop_w)

        # Swap axes so the block index is first, followed by pixel indices
        swapped_axes = reshaped.swapaxes(1, 2)

        # Create the final array of image crops
        crops = swapped_axes.reshape(-1, crop_h, crop_w)
        crop_layers[k] = crops

    return crop_layers


def sample_crops():
    png_examples_path = p.join(crop_path, 'png_examples')
    if not p.exists(crop_path):
        os.mkdir(crop_path)
    if not p.exists(png_examples_path):
        os.mkdir(png_examples_path)
    ds = scan_xarray_dataset()
    da = ds.to_array()
    ds_size = ds.sizes['time']

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        auto_refresh=True
    )
    crops_written = 0
    with progress:  # this line is necessary
        task = progress.add_task("Generating crops...", total=ds_size, start=True)
        for i in range(ds_size):
            arr = da[0, i, 0, :, :]
            # preprocess
            arr = arr.to_numpy()
            arr[np.isnan(arr)] = -1
            arr[arr < 0.] = 0.
            crops = segment_image(arr, crop_size, crop_size)
            for c in range(crops.shape[0]):
                crop = crops[c, :, :]
                rainfall_perc = np.count_nonzero(crop) / (crop_size * crop_size)
                if rainfall_perc > 0.2:
                    np.save(p.join(crop_path, f'{i}_{c}.npy'), crop)
                    progress.update(task, description=f'Generating crops [{crops_written} written]')
                    crops_written += 1
                    if crops_written % 1000 == 0:
                        plt.imshow(crop)
                        plt.savefig(p.join(png_examples_path, f'{i}_{c}.png'))
            progress.update(task, advance=1)


def reduce_coverage(precip):
    return precip.sel(lat=lat_limit, lon=lon_limit)


def cvt_lon_coords(d):
    d.coords['lon'] = (d.coords['lon'] + 180) % 360 - 180
    d = d.sortby(d.lon)
    return d

def load_context(year, time):
    contexts = {}
    for k, v in era5_vars_path.items():
        if k == 'elevation':  # static in time
            new_layer = xr.open_dataarray(v)
            contexts[k] = new_layer[0, :, :]
        else:  # dynamic in time
            new_layer = xr.open_dataarray(v.format(year))
            contexts[k] = new_layer
            contexts[k] = new_layer.sel(time=time)
    return contexts

def interpolate_context(**layers):
    lat = layers['precip']['lat'].values
    lon = layers['precip']['lon'].values
    for k, v in layers.items():
        if k != 'precip':
            layers[k] = v.interp(latitude=lat, longitude=lon)
    return layers


def preprocess(**kwargs) -> dict:
    for k, da in kwargs.items():
        # preprocess
        arr = da.to_numpy()
        arr[np.isnan(arr)] = -1
        arr[arr < 0.] = 0.
        kwargs[k] = arr
    return kwargs


def remove_contexual_files():
    """
    Remove all contextual files from the crop directory. Only preserve precipitation data
    """
    path_ = '/home/yl241/data/CLIMATE/nexrad_min_0.2_crops'
    print("REMOVING CONTEXTUAL FILES FROM ", path_)
    files = os.listdir(path_)
    files = [f for f in files if '.npy' in f]
    files = [f for f in files if 'precip' not in f]
    for f in files:
        os.remove(p.join(crop_path, f))
def sample_crops_with_context(save_context=True):
    """
    Sample crops from the dataset, and save them to disk.
    @param save_context: if True, saves contexual layers; otherwise, saves rainfall data only
    """
    png_examples_path = p.join(contextual_crop_path, 'png_examples')
    if not p.exists(contextual_crop_path):
        os.mkdir(contextual_crop_path)
    if not p.exists(png_examples_path):
        os.mkdir(png_examples_path)
    ds = scan_xarray_dataset()
    da = ds.to_array()
    ds_size = ds.sizes['time']

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        auto_refresh=True
    )
    crops_written = 0
    with progress:  # this line is necessary
        task = progress.add_task("Generating crops...", total=ds_size, start=True)
        da = cvt_lon_coords(da)
        da = reduce_coverage(da)
        for i in range(ds_size):
            precip = da[0, i, 0, :, :]  # precipitation
            time_ = precip['time']  # datetime
            year_ = time_.dt.year.values.item()  # int
            if save_context:
                layers = load_context(year_, time_)
            else:
                layers = {}
            layers.update({'precip': precip})
            layers = interpolate_context(**layers)
            layers = preprocess(**layers)
            layer_crop = segment_image(crop_dim=(crop_size, crop_size), **layers)
            for c in range(layer_crop['precip'].shape[0]):
                precip_crop = layer_crop['precip'][c, :, :]
                rainfall_perc = np.count_nonzero(precip_crop) / (crop_size * crop_size)
                if rainfall_perc > 0.2:
                    for k, v in layer_crop.items():
                        np.save(p.join(contextual_crop_path, f'{k}_f{i}_c{c}.npy'), v[c, :, :])
                    progress.update(task, description=f'Generating crops [{crops_written} written]')
                    crops_written += 1
                    if crops_written % 1000 == 0:
                        for k, v in layer_crop.items():
                            plt.imshow(v[c, :, :])
                            plt.title(f'{k}_f{i}_c{c}')
                            plt.savefig(p.join(png_examples_path, f'{k}_f{i}_c{c}.png'))
            progress.update(task, advance=1)


if __name__ == '__main__':
    # wandb.init()
    # sample_crops_with_context()
    remove_contexual_files()
