import os

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torchvision as tv

import pickle
import random
import time

from ttfs import TTFS_encoder
from isi import ISI_encoding
from multiplexing_ttfs import multiplexing_encoding_TTFS_phase
from multiplexing_isi import multiplexing_encoding_ISI_phase
from datasets import *

from torch_device import dtype, device
from sparse_data_generator import sparse_generator
from surrogate_encoder import encode_data

from surrogate_model import run_snn
from surrogate_train import init_model, compute_classification_accuracy, train

print(device)

def gen_encoded(x_train, y_train, x_test, y_test, encoder_type, grp_size, div_data, nb_steps):
    if encoder_type not in ['TTFS', 'ISI', 'Phase+TTFS', 'Phase+ISI']:
        raise Exception('Incorrect encoder type')
    if grp_size:
        nb_unites = int(x_train.shape[1]/grp_size)
    else:
        nb_unites = x_train.shape[1]

    mask = np.ones_like(y_train, dtype=np.bool)
    mask[::div_data] = 0

    trn_d = encode_data(x_train[mask,:].reshape(-1, 784), y_train[mask], 
                        nb_units=nb_unites, 
                      encoder_type=encoder_type, 
                  group_size=grp_size,
                  batch_size=512, 
                  nb_steps=nb_steps, 
                  TMAX=100)

    val_d = encode_data(x_train[~mask,:].reshape(-1, 784), y_train[~mask], 
                        nb_units=nb_unites, 
                      encoder_type=encoder_type, 
                  group_size=grp_size,
                  batch_size=1024, 
                  nb_steps=nb_steps, 
                  TMAX=100)

    test_d = encode_data(x_test, y_test, 
                         nb_units=nb_unites, 
                       encoder_type=encoder_type, 
                  group_size=grp_size,
                  batch_size=1024, 
                  nb_steps=nb_steps, 
                  TMAX=100)
    return trn_d, val_d, test_d

confs = (
    {'gs': 1,  'enc': 'TTFS', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 1,  'enc': 'ISI', 'nb_steps': 100, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234}, 
    # {'gs': 4,  'enc': 'Phase+TTFS', 'nb_steps': 1000, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
    # {'gs': 4,  'enc': 'Phase+ISI', 'nb_steps': 1000, 'lr': .0002, 'dataset': 'MNIST', 'seed': 1234},
)

epochs = 25
time_step = .001
div_data = 5


for c in confs:
    if c['dataset'] == 'MNIST':
        dataset = load_mnist()
    elif c['dataset'] == 'FMNIST':
        dataset = load_fmnist()
    elif c['dataset'] == 'CIFAR10_gray':
        dataset = load_cifar10_gray()
    else:
        raise Exception('Wrong database name')
    print('*'* 20)
    nb_inputs  = int(dataset['x_train'].shape[1]/c['gs'])
    nb_hidden  = int(25*c['gs'])
    nb_outputs = 10

    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    random.seed(c['seed'])

    trn_d, val_d, test_d = gen_encoded(dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test'], 
                                     c['enc'], c['gs'], div_data, c['nb_steps'])

    print(c['dataset'])
    print('Encoder: {}, group_size: {}, lr: {}, nb_steps: {}'.format(c['enc'], c['gs'], c['lr'], c['nb_steps']))

    # print('CUDA MEM:')
    params, alpha, beta = init_model(nb_inputs, nb_hidden, nb_outputs, time_step ) #TODO: tau_mem and tau_syn
    print("W1:", params[0].shape)
    print("W2:", params[1].shape)
    total_weights = params[0].numel() + params[1].numel()
    print("#weights:", total_weights)

    loss_hist, train_acc, val_acc, w_traj = train(trn_d, val_d, c['nb_steps'], params, alpha, beta, 
                                                lr=c['lr'], nb_epochs=epochs, return_weights=True)
    test_acc = compute_classification_accuracy(test_d, c['nb_steps'], params, alpha, beta)
    print("Test accuracy: %.3f" % test_acc)

    fpath = 'SG_{}__{}__gs{}_i{}_h{}_tot{}__epochs{}_lr{}_div{}_nbs{}_ts{}__seed{}.h'.format(
      'MNIST', c['enc'].replace('+', '_'), c['gs'], nb_inputs, nb_hidden, total_weights, epochs, 
      str(c['lr']).replace('.', '_'), div_data, c['nb_steps'], time_step, c['seed']
  )
    with open(fpath, 'wb') as fo:
        pickle.dump({
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'w1_shape': list(params[0].shape),
            'w2_shape': list(params[0].shape),
            'numel': w1.numel() + w2.numel(),
            'params': params,
            'w_traj': w_traj,
        }, fo, protocol=pickle.HIGHEST_PROTOCOL)
    print(fpath)