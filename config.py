# coding=utf-8
# Lint as: python3

    # Copyright (C) 2025  Jean-Daniel PASCAL PRIETO

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Training NCSN++ on Church with VE SDE."""

import ml_collections
import torch
from configs.default_MultiRIR_ncsnpp_configs import get_default_configs


def get_config() -> ml_collections.ConfigDict:
    """
    Configuration for model shape, hyper parameters, training and eval options, etc

    Returns:
        ml_collections.ConfigDict: Configuration object
    """
    config = get_default_configs()
    # training
    training = config.training
    training.batch_size = 4
    training.n_iters = 400000
    training.snapshot_freq = 10000
    training.log_freq = 1000
    training.eval_freq = 1000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 2000
    ## produce samples at each snapshot.
    training.snapshot_sampling = False
    training.sde = "sbvesde"

    # sampling
    sampling = config.sampling 
    sampling.n_steps_each = 1
    sampling.method = "pc"  # pc or ode

    # evaluation
    evaluate = config.eval
    evaluate.begin_ckpt = 23
    evaluate.end_ckpt = 23
    # for now only support batch size of 1
    evaluate.batch_size = 1
    evaluate.enable_sampling = True
    evaluate.num_samples = 200   ### number of iteration of the diffusion model, for now evaluate.num_samples should be the same as model.num_scales
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"
    # set distance_peaks to True to do the sampling on all the samples in the dataset test and calculate error of estimation peaks
    evaluate.distance_peaks = False

    # data
    data = config.data
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.dataset = "Multichannel_RIR"
    data.rir_samples_count = 512  # length of the rir used for training, the difference with total_rir_samples_count is if we want to use chunks of the rir for training
    data.total_rir_samples_count = 512 # should be a multiple of rir_samples_count
    data.begining = 0
    data.image_size = data.rir_samples_count / 2  # size of the input, number of channels
    data.channels = 32
    data.num_channels = 1
    data.npz_path = "./dataset_measurement_arpege/"
    data.measured_rir = True # if True the dataset should contain the measured rir in addition to the perfect and real rirs


    data.num_room = 50000
    data.pos_per_room = 3
    data.sample_rate = 16000
    if data.npz_path == "./dataset_ircam/":
        data.num_room = 1
        data.pos_per_room = 10
    if data.npz_path == "./dataset_measurement_arpege/":
        data.num_room = 1
        data.pos_per_room = 45

    # model
    model = config.model 
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.name = "ncsnpp"
    model.k = 2.6           ## you can adjust k and c to change the shape of the noise schedule, k =2.6 and c = 0.4 shows good performance
    model.c = 0.4
    model.num_scales = 200  # number of diffusion steps, for now if you train with 200 steps you should sample with 200 steps
    model.nf = int(data.rir_samples_count / 4) # for 512 samples
    model.ch_mult = (1,1,1,2,2,2,2) # for 512 samples
    model.num_res_blocks = 1 # for 512 samples
    model.attn_resolutions = (8,) # for 512 samples
    model.kernel_first_convolution = 15 

    # optimization
    optim = config.optim
    optim.lr = 2e-4

    return config
