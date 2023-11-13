__author__ = 'yuhao liu'

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
# import tensorflow as tf
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, ncsnpp_cond
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from precipitation import contextual_crop_dataset
import evaluation
import likelihood
import sde_lib
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import wandb
from utils import save_checkpoint, restore_checkpoint
from util.helper import print_segment, get_training_progressbar, InfiniteLoader, cm_, show_inputs, wandb_display_grid
# from util.criteria import crps_empirical
from pathlib import Path
from natsort import natsorted
from util.helper import yprint


def train(config, workdir):
    """Runs the training pipeline.

        Args:
          config: Configuration to use.
          workdir: Working directory for checkpoints and TF summaries. If this
            contains checkpoint training will be resumed from the latest checkpoint.
        """
    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    if wandb.run.resumed:
        path_ = Path(checkpoint_dir)
        runs_ = os.listdir(path_.parent.parent)  # all wandb runs
        # previous run with the same run id (the last one is the current run)
        prev_run = natsorted([r for r in runs_ if wandb.run.id in r])[-2]
        prev_run_ckpt_path = os.path.join(path_.parent.parent, prev_run, 'checkpoints', 'checkpoint.pth')
        state = restore_checkpoint(prev_run_ckpt_path, state, config.device)
    elif config.logger.load_ckpt_path is not None:
        yprint(f'Loading checkpoint from {config.logger.load_ckpt_path}')
        state = restore_checkpoint(config.logger.load_ckpt_path, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = contextual_crop_dataset.get_rainfall_contextual_crop_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)
    train_iter = InfiniteLoader(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = InfiniteLoader(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    batch_ = next(train_iter)
    if initial_step == 0:  # do not show inputs if resumed training
        show_inputs(**batch_)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    sampling_shape = (config.sampling.sampling_batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    s = config.sampling.sampling_batch_size
    num_train_steps = config.training.n_iters

    wandb.watch(models=score_model, log_freq=config.logger.train_log_param_freq)
    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info(f"Starting training loop at step {initial_step}/{num_train_steps}")
    progress = get_training_progressbar()
    with progress:
        cur_ep = initial_step // len(train_ds)  # current epoch
        task = progress.add_task(f"Training... [Epoch {cur_ep}]", total=num_train_steps, start=True)
        progress.update(task, advance=initial_step)
        # generate null condition for sampling
        if config.data.condition_mode == 1:
            sampling_null_condition = torch.ones((s, config.data.num_channels, config.data.condition_size, config.data.condition_size)) * config.model.null_token
        elif config.data.condition_mode == 2:
            sampling_null_condition = torch.ones((s, config.data.num_channels + config.data.num_context_chs, config.data.image_size, config.data.image_size)) * config.model.null_token
        sampling_null_condition = sampling_null_condition.to(config.device)

        # TRAINING LOOP
        for step in range(initial_step, num_train_steps + 1):
            batch_dict = next(train_iter)
            batch = scaler(batch_dict['precip_gt'].to(config.device))

            if config.data.condition_mode == 0:
                raise AttributeError()  # deprecated
            elif config.data.condition_mode == 1:
                condition = batch_dict['precip_lr'].to(config.device)
            elif config.data.condition_mode == 2:
                exclude_keys = ['precip_lr', 'precip_gt']
                tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
                stacked_tensor = torch.cat(tensors_to_stack, dim=1)
                condition = stacked_tensor.to(config.device)

            # if config.training.task == 'class_conditional_generation':
            #     condition = torch.from_numpy(label_._numpy()).to(config.device)
            #     # implement dropout
            #     context_mask = torch.bernoulli(torch.zeros(condition.shape[0]) + (1 - config.model.drop_prob))
            #     # context_mask = context_mask[:, None, None, None]  # shape: (batch_size, 1, 1, 1)
            #     context_mask = context_mask.to(config.device)  # shape: (batch_size,)
            #     # condition = (condition * context_mask).int()
            #     condition[context_mask == 0] = config.data.num_classes  # set to null token
            #     condition = condition.int()
            # elif config.training.task == 'super_resolution':
            #     if config.data.condition_size < config.data.image_size:
            #         condition = F.interpolate(batch, size=config.data.condition_size, mode='area')  # down sample to get condition
            #     elif config.data.condition_size == config.data.image_size:
            #         condition = batch.detach().clone()

            # implement dropout
            context_mask = torch.bernoulli(torch.zeros(condition.shape[0]) + (1 - config.model.drop_prob))
            # context_mask[context_mask == 0] = -1
            context_mask = context_mask[:, None, None, None]  # shape: (batch_size, 1, 1, 1)
            context_mask = context_mask.to(config.device)  # shape: (batch_size,)
            condition = condition * context_mask  # shape: (batch_size, c, h, w)
            # condition[context_mask == 0] = -1
            null_token_mask = torch.zeros_like(context_mask)
            null_token_mask[context_mask == 0] = config.model.null_token  # set to null token
            condition = condition + null_token_mask

            # Execute one training step
            loss, loss_dict = train_step_fn(state, batch, condition)
            if step % config.logger.train_log_freq == 0:
                wandb.log({"train/loss": loss}, step=step)
                # wandb.log({"train/CRPS": crps_empirical(loss_dict['score'], loss_dict['target'])}, step=step)

            if step % config.logger.train_log_score_freq == 0:
                # LOG SCORE
                # {'sigmas': sigmas, 'noise': noise, 'score': score, 'target': target, 'error_map': losses}
                if config.data.condition_mode == 1:
                    wandb_display_grid(condition, log_key='train_score/condition', caption='condition', step=step, ncol=s)
                elif config.data.condition_mode == 2:
                    wandb_display_grid(batch_dict['precip_up'], log_key='train_score/condition', caption='low_res upsampled',
                                       step=step, ncol=s)
                wandb_display_grid(context_mask, log_key='train_score/mask', caption='context_mask', step=step, ncol=s)
                wandb_display_grid(loss_dict['score'], log_key='train_score/score', caption='score', step=step, ncol=s)
                wandb_display_grid(loss_dict['target'], log_key='train_score/target', caption='target', step=step, ncol=s)
                # wandb_display_grid(loss_dict['noise'], log_key='train_score/noise', caption='noise', step=step, ncol=s)
                wandb_display_grid(loss_dict['perturbed_data'], log_key='train_score/perturbed_data', caption='perturbed_data', step=step, ncol=s)
                wandb_display_grid(loss_dict['denoised_data'], log_key='train_score/denoised_data',
                                   caption='denoised_data', step=step, ncol=s)
                wandb_display_grid(loss_dict['error_map'], log_key='train_score/error_map', caption='error_map', step=step, ncol=s)
                wandb_display_grid(batch, log_key='train_score/gt', caption='gt',
                                   step=step, ncol=s)

            if step != 0 and step % config.logger.train_log_img_freq == 0 and config.logger.show_unconditional_samples:
                # UNCONDITIONAL SAMPLING
                progress.update(task, description=f'Unconditional Sampling....')
                sample, n = sampling_fn(score_model, null_condition=sampling_null_condition)
                grid = make_grid(sample[0:s, :, :, :])
                grid_mono = grid[0, :, :].unsqueeze(0)
                cm_grid = cm_(grid_mono.detach().cpu())
                images = wandb.Image(cm_grid, caption='unconditional samples')
                wandb.log({"eval/unconditional_samples": images}, step=step)
                progress.update(task, description=f'Training... [Epoch {cur_ep}]')

            # Save a temporary checkpoint to resume training after pre-emption periodically
            if step != 0 and step % config.logger.snapshot_freq_for_preemption == 0:
                save_checkpoint(os.path.join(checkpoint_dir, 'checkpoint.pth'), state)

            # Report the loss on an evaluation dataset periodically
            if step % config.logger.train_log_freq == 0:
                # next_batch = next(eval_iter)
                # batch, label_ = next_batch['image'], next_batch['label']
                # batch = torch.from_numpy(batch._numpy()).to(config.device).float()
                # batch = batch.permute(0, 3, 1, 2)
                # batch = scaler(batch)
                eval_batch_dict = next(eval_iter)
                batch = scaler(eval_batch_dict['precip_gt'].to(config.device))
                if config.data.condition_mode == 1:
                    condition = eval_batch_dict['precip_lr'].to(config.device)
                elif config.data.condition_mode == 2:
                    exclude_keys = ['precip_lr', 'precip_gt']
                    tensors_to_stack = [tensor for key, tensor in eval_batch_dict.items() if key not in exclude_keys]
                    stacked_tensor = torch.cat(tensors_to_stack, dim=1)
                    condition = stacked_tensor.to(config.device)
                # if config.training.task == 'class_conditional_generation':
                #     # condition = torch.from_numpy(label_._numpy()).to(config.device)
                #     raise NotImplementedError()
                # elif config.training.task == 'super_resolution':
                #     # condition = F.interpolate(batch, size=config.data.condition_size, mode='area')
                #     if config.data.condition_size < config.data.image_size:
                #         condition = F.interpolate(batch, size=config.data.condition_size,
                #                                   mode='area')  # down sample to get condition
                #     elif config.data.condition_size == config.data.image_size:
                #         condition = batch.detach().clone()
                #
                # else:
                #     raise NotImplementedError()

                eval_loss, _ = eval_step_fn(state, batch, condition)
                # logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item())) # output to stdout
                wandb.log({"eval/loss": eval_loss}, step=step)

            if step != 0 and step % config.logger.train_log_img_freq == 0:
                # CONDITIONAL SAMPLING
                progress.update(task, description=f'Conditional Sampling....')
                sample, n = sampling_fn(score_model, condition=condition[0:s],
                                        w=config.model.w_guide, null_condition=sampling_null_condition)
                if config.data.condition_mode == 1:
                    # display condition and output in one grid
                    if config.data.condition_size < config.data.image_size:
                        low_res_display = F.interpolate(condition, size=(config.data.image_size, config.data.image_size),
                                                        mode='nearest')
                    elif config.data.condition_size == config.data.image_size:
                        low_res_display = condition.detach().clone()
                elif config.data.condition_mode == 2:
                    if config.data.condition_size < config.data.image_size:
                        low_res_display = F.interpolate(eval_batch_dict['precip_lr'], size=(config.data.image_size, config.data.image_size),
                                                        mode='nearest')
                    elif config.data.condition_size == config.data.image_size:
                        low_res_display = eval_batch_dict['precip_lr'].detach().clone()
                low_res_display = low_res_display.to(config.device)

                concat_samples = torch.cat((low_res_display[0:s, :, :, :], sample[0:s, :, :, :], batch[0:s, :, :, :]), dim=0)
                grid = make_grid(concat_samples, nrow=s)
                grid_mono = grid[0, :, :].unsqueeze(0)
                cm_grid = cm_(grid_mono.detach().cpu())
                images = wandb.Image(cm_grid, caption='conditional generation [rainfall / output / gt]')
                wandb.log({"eval/conditional_samples": images}, step=step)

                progress.update(task, description=f'Training... [Epoch {cur_ep}]')

            # Save a checkpoint periodically
            if step != 0 and step % config.logger.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // config.logger.snapshot_freq
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            progress.update(task, advance=1)
            cur_ep = step // len(train_ds)
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}_FINAL.pth'), state)
        yprint('Training finished.')


def check_config(config):
    """
    Check the config file.
    """
    if config.logger.resume_id is not None and config.logger.load_ckpt_path is not None:
        raise ValueError('Cannot resume from both wandb and local ckpt. Please choose one.')
    return


@hydra.main(version_base=None, config_path="./configs", config_name="ncsnpp_rainfall")
def main(config):
    if config.mode == "train":
        # pre-process config file
        config.data.resolution_ratio = int(config.data.image_size / config.data.condition_size)
        check_config(config)
        # init wandb
        config_ = OmegaConf.to_container(config, resolve=True)
        if config.logger.resume_id is not None:
            id_ = config.logger.resume_id
            resume_ = 'must'
        else:
            id_ = None
            resume_ = 'never'
        run = wandb.init(project=config.logger.project_name, config=config_, dir=config.logger.save_dir,
                         save_code=True, settings=wandb.Settings(code_dir="."), notes=config.msg,
                         mode='online' if config.msg != 'debug' else 'disabled',
                         id=id_, resume=resume_)
        save_dir = run.dir[:-5]
        run.summary['run_dir'] = save_dir
        run.summary['run_id'] = run.id
        # run.summary.update()
        gfile_stream = open(os.path.join(save_dir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        if run.resumed:
            yprint(f'Resuming training for run {run.id}...')
        print_segment()

        train(config, save_dir)  # Run the training pipeline
    elif config.mode == "eval":
        # Run the evaluation pipeline
        raise NotImplementedError('eval is not implemented yet.')
        run_lib.evaluate(config.config, config.workdir, config.eval_folder)
    else:
        raise ValueError(f"Mode {config.mode} not recognized.")


if __name__ == '__main__':
    main()
