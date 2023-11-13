# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""
import run_lib
from absl import app
from absl import flags
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch # must import before tf
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from util.helper import print_segment


@hydra.main(version_base=None, config_path="./configs", config_name="ncsnpp_cifar10_config")
def main(FLAGS):
    if FLAGS.mode == "train":
        # init wandb
        config_ =OmegaConf.to_container(FLAGS, resolve=True)
        run = wandb.init(project=FLAGS.logger.project_name, config=config_, dir=FLAGS.logger.save_dir)
        save_dir = run.dir[:-5]
        # Create the working directory
        # tf.io.gfile.makedirs(save_dir)

        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(save_dir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        print_segment()
        run_lib.train(FLAGS, save_dir)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    main()
