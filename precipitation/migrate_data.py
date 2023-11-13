"""
Create a subset of rainfall dataset that only contains images with rainfall above a certain threshold
"""

import os
import os.path as p
import shutil
from tqdm import tqdm
import xarray as xr
import wandb
from util.helper import time_func


def create_small_dataset(year):
    entire_dataset_path = '/home/yl241/data/CLIMATE/nexrad'
    output_path = f'/home/yl241/data/CLIMATE/nexrad_{year}'
    assert p.exists(entire_dataset_path)
    if p.exists(output_path) and len(os.listdir(output_path)) > 0:
        raise FileExistsError()
    if not p.exists(output_path):
        os.mkdir(output_path)
        print(f'Created new directory {output_path} to store outputs')
    files = os.listdir(entire_dataset_path)
    files = [f for f in files if '_' + year in f]
    print(f'detected {len(files)} files for year {year}')
    for f in tqdm(files):
        shutil.copyfile(p.join(entire_dataset_path, f), p.join(output_path, f))
    print(f'{len(files)} files moved to {output_path}')


@time_func
def select_above_threshold(min_=0.25):
    wandb.init()
    input_path = f'/home/yl241/data/CLIMATE/nexrad_min_0.2'
    output_path = f'/home/yl241/data/CLIMATE/nexrad_min_{min_}'
    if p.exists(output_path) and len(os.listdir(output_path)) > 0:
        raise FileExistsError(f'{output_path} already exists and contain files')
    if not p.exists(output_path):
        os.mkdir(output_path)
        print(f'Created new directory {output_path} to store outputs')
    assert p.exists(input_path)
    files = os.listdir(input_path)
    files_saved = 0
    with tqdm(total=len(files)) as pbar:
        for f in files:
            d = xr.open_dataarray(p.join(input_path, f))
            mean = d.mean()
            if mean >= min_:
                shutil.copyfile(p.join(input_path, f), p.join(output_path, f))
                files_saved += 1
            pbar.update(n=1)
            pbar.set_description(f'{files_saved} saved')
    print(f'{files_saved} / {len(files)} saved to {output_path}')
    wandb.alert(title='Migration complete', text=f'{files_saved} / {len(files)} saved to {output_path}')


def main():
    # create_small_dataset('2020')
    select_above_threshold()


if __name__ == '__main__':
    main()