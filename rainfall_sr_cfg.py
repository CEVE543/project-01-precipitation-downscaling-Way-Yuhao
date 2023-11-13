### TEMPORARILY MOVED TO PARENT FOLDER SO THAT THE SCRIPT CAN BE RAN IN TERMINAL ###

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import numpy as np
import hydra
import controllable_generation
from precipitation import contextual_crop_dataset
from utils import restore_checkpoint
from omegaconf import OmegaConf, DictConfig

sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ncsnpp_cond
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      EulerMaruyamaPredictor,
                      AncestralSamplingPredictor,
                      NoneCorrector,
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
from util.helper import yprint


def image_grid(x, size=None, config=None):
    # config cannot be none...
    if size is None:
        size = config.data.image_size
    channels = config.data.num_channels
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img


def show_samples(x, size=None, caption=None, config=None, save_path=None):
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x, size, config)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img)
    if caption is not None:
        plt.title(caption)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


@hydra.main(version_base=None, config_path="configs", config_name="ncsnpp_rainfall")
def main(config):
    config.training.batch_size = config.data.batch_size
    # global config
    # config = OmegaConf.to_container(config, resolve=True)
    # load score_based models
    checkpoint_path = '/home/yl241/models/NCSNPP/wandb/run-20231103_114852-72wzm2c4/checkpoints/'
    sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
    if sde.lower() == 'vesde':
        ckpt_filename = checkpoint_path + "checkpoint_100.pth"
        # from configs.ml_collections_configs.ve import cifar10_ncsnpp_continuous as configs
        # ckpt_filename = checkpoint_path + "/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
        # config = configs.get_config()
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    elif sde.lower() == 'vpsde':
        raise NotImplementedError()
        # from configs.ml_collections_configs.vp import cifar10_ddpmpp_continuous as configs
        # ckpt_filename = checkpoint_path + "/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
        # config = configs.get_config()
        # sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        # sampling_eps = 1e-3
    elif sde.lower() == 'subvpsde':
        raise NotImplementedError()
        # from configs.ml_collections_configs.subvp import cifar10_ddpmpp_continuous as configs
        # ckpt_filename = checkpoint_path + "/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
        # config = configs.get_config()
        # sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        # sampling_eps = 1e-3

    # config.device = torch.device('cuda:1')  # OVERRIDING

    batch_size = 4  # @param {"type":"integer"}
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    config.data.batch_size = batch_size

    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    # PC super-resolution
    config.data.resolution_ratio = int(config.data.image_size / config.data.condition_size)
    train_ds, eval_ds = contextual_crop_dataset.get_rainfall_contextual_crop_dataset(config,
                                                               uniform_dequantization=config.data.uniform_dequantization)
    eval_iter = iter(eval_ds)
    bpds = []

    predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.16  # @param {"type": "number"}
    n_steps = 1  # @param {"type": "integer"}
    probability_flow = False  # @param {"type": "boolean"}
    # scale_factor = 0.25


    pc_upsampler = controllable_generation.get_pc_cfg_upsampler(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=True)
    which_batch = 100
    for i in range(which_batch):
        # low_res, high_res = next(eval_iter)
        batch = next(eval_iter)
    if config.data.condition_mode == 1:
        low_res, high_res = batch['precip_lr'], batch['precip_gt']
    elif config.data.condition_mode == 2:
        raise NotImplementedError()
    low_res_display = F.interpolate(batch['precip_lr'], size=(config.data.image_size, config.data.image_size),
                                                        mode='nearest')
    msg = config.msg
    assert 'msg' in config, 'Please specify a message for the experiment'
    save_plt_path = os.path.join(f'/home/yl241/workspace/NCSN/plt/', msg)
    if msg != 'debug':
        if os.path.exists(save_plt_path):
            raise FileExistsError()
        else:
            os.makedirs(save_plt_path)

    # save config file
    yaml_str = OmegaConf.to_yaml(config)
    # Write the string to a file
    with open(os.path.join(save_plt_path, 'config.yaml'), 'w') as f:
        f.write(yaml_str)

    low_res, high_res = low_res.to(config.device), high_res.to(config.device)
    show_samples(high_res, config=config, caption='Original image', save_path=os.path.join(save_plt_path, 'gt.png'))
    show_samples(low_res_display, config=config, caption='input', save_path=os.path.join(save_plt_path, 'input.png'))
    x = pc_upsampler(score_model, scaler(low_res), w=config.model.w_guide,
                     out_dim=(batch_size, 1, config.data.image_size, config.data.image_size), save_dir=save_plt_path,
                     null_token=config.model.null_token)
    show_samples(x, config=config, caption='output', save_path=os.path.join(save_plt_path, 'output.png'))
    yprint(f'Job {msg} finished.')


if __name__ == '__main__':
    main()

