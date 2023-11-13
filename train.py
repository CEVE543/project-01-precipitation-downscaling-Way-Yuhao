__author__ = 'yuhao liu'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import numpy as np
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import tensorflow as tf
import tensorflow_gan as tfgan
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, ncsnpp_cond
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
# from precipitation import crop_dataset
from precipitation import contextual_crop_dataset
import evaluation
import likelihood
import sde_lib
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import wandb
from rich.progress import track
from utils import save_checkpoint, restore_checkpoint
from util.helper import print_segment, get_training_progressbar, InfiniteLoader, cm_


def train(config, workdir):
    """Runs the training pipeline.

        Args:
          config: Configuration to use.
          workdir: Working directory for checkpoints and TF summaries. If this
            contains checkpoint training will be resumed from the latest checkpoint.
        """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    # train_ds, eval_ds = contextual_crop_dataset.get_rainfall_contextual_crop_dataset(config,
    #                                             uniform_dequantization=config.data.uniform_dequantization)
    # train_iter = InfiniteLoader(train_ds)  # pytype: disable=wrong-arg-types
    # eval_iter = InfiniteLoader(eval_ds)  # pytype: disable=wrong-arg-types
    # # Create data normalizer and its inverse
    # scaler = datasets.get_data_scaler(config)
    # inverse_scaler = datasets.get_data_inverse_scaler(config)
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

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
    sampling_batch_size = 4  # OVVERRIDE
    sampling_shape = (sampling_batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info(f"Starting training loop at step {initial_step}/{num_train_steps}")
    progress = get_training_progressbar()
    with progress:
        # cur_ep = initial_step // len(train_ds)  # current epoch
        cur_ep = 'UNKNOWN'
        task = progress.add_task(f"Training... [Epoch {cur_ep}]", total=num_train_steps, start=True)
        progress.update(task, advance=initial_step)
        # for step in track(range(initial_step, num_train_steps + 1),
        #                   description="Training...", refresh_per_second=1, transient=True):

        for step in range(initial_step, num_train_steps + 1):
            # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
            # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
            # low_res_samples, high_res_samples = next(train_iter)

            # batch = next(train_iter)
            # low_res_samples, high_res_samples = batch['precip_lr'], batch['precip_hr']
            # high_res_samples = high_res_samples.to(config.device)
            # print('got a batch!')
            # batch = batch.permute(0, 3, 1, 2)
            # high_res_samples = scaler(high_res_samples)

            batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
            batch = batch.permute(0, 3, 1, 2)
            batch = scaler(batch)

            # Execute one training step
            loss = train_step_fn(state, batch)
            if step % config.logger.train_log_freq == 0:
                # logging.info("step: %d, training_loss: %.5e" % (step, loss.item())) # output to stdout
                wandb.log({"train/loss": loss}, step=step)
            if step % config.logger.train_log_img_freq == 0:
                progress.update(task, description=f'Sampling....')
                sample, n = sampling_fn(score_model)
                grid = make_grid(sample[0:8, :, :, :])
                # grid_mono = grid[0, :, :].unsqueeze(0)
                # cm_grid = cm_(grid_mono.detach().cpu())
                # grid = make_grid(cm_(sample[0:8, :, :, :].detach().cpu()))
                # images = wandb.Image(cm_(grid.detach().cpu()), caption='training outputs')
                images = wandb.Image(grid, caption='training outputs')
                wandb.log({"train/outputs": images}, step=step)
                progress.update(task, description=f'Training... [Epoch {cur_ep}]')

            # Save a temporary checkpoint to resume training after pre-emption periodically
            if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
                save_checkpoint(checkpoint_meta_dir, state)

            # Report the loss on an evaluation dataset periodically
            if step % config.logger.train_log_freq == 0:
                # eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
                # eval_batch = next(eval_iter)
                # low_res_samples, high_res_samples = next(eval_iter)
                # low_res_samples, high_res_samples = eval_batch['precip_lr'], eval_batch['precip_hr']
                # high_res_samples = high_res_samples.to(config.device)
                # # eval_batch = eval_batch.permute(0, 3, 1, 2)
                # high_res_samples = scaler(high_res_samples)

                batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
                batch = batch.permute(0, 3, 1, 2)
                batch = scaler(batch)

                eval_loss = eval_step_fn(state, batch)
                # logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item())) # output to stdout
                wandb.log({"eval/loss": eval_loss}, step=step)

            if step != 0 and step % config.logger.train_log_img_freq == 0:
                pass
                # TODO: fix this
                # grid = make_grid(eval_batch[0:8, :, :, :])
                # images = wandb.Image(grid, caption='eval outputs')
                # wandb.log({"eval/outputs": images}, step=step)
                # # writer.add_image('eval/outputs', grid, global_step=step)

            # Save a checkpoint periodically and generate samples if needed
            if step % config.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // config.training.snapshot_freq
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if config.training.snapshot_sampling:
                    progress.update(task, description=f'Sampling....')
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample, n = sampling_fn(score_model)
                    ema.restore(score_model.parameters())
                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    tf.io.gfile.makedirs(this_sample_dir)
                    nrow = int(np.sqrt(sample.shape[0]))
                    image_grid = make_grid(sample, nrow, padding=2)
                    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    with tf.io.gfile.GFile(
                            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                        np.save(fout, sample)

                    with tf.io.gfile.GFile(
                            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                        save_image(image_grid, fout)
                    progress.update(task, description=f'Training....')
            progress.update(task, advance=1)
            # cur_ep = step // len(train_ds)


@hydra.main(version_base=None, config_path="./configs", config_name="ncsnpp_cifar10_config")
def main(FLAGS):
    if FLAGS.mode == "train":
        # init wandb
        config_ = OmegaConf.to_container(FLAGS, resolve=True)
        run = wandb.init(project=FLAGS.logger.project_name, config=config_, dir=FLAGS.logger.save_dir,
                         save_code=True, settings=wandb.Settings(code_dir="."), notes=FLAGS.msg,
                         mode='online' if FLAGS.msg != 'debug' else 'disabled')
        save_dir = run.dir[:-5]
        gfile_stream = open(os.path.join(save_dir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        print_segment()
        train(FLAGS, save_dir)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        raise NotImplementedError('eval is not implemented yet.')
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == '__main__':
    main()