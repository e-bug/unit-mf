# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       5/24/2019
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

from npmf.error_metrics import *
import numpy as np

import matplotlib.pyplot as plt

import dill
import pickle
import bz2
import os.path


def get_values_tensor(hypers, losses, train_errors, valid_errors):
    hypers2idx = [{'%e' % v: i for i,v in enumerate(hyper)} for hyper in hypers]
    idx2hypers = [{v: i for i, v in d.items()} for d in hypers2idx]
    tensor_shape = []
    for hyper in hypers:
        tensor_shape += [len(hyper)]
    tensor_shape += [3]

    values_tensor = np.nan * np.ones(tensor_shape)

    for idx, d in enumerate(losses):
        k, v = list(d.items())[0]
        hyperparam_vals = [s.split('=')[1] for s in k.split(',')]
        mean_loss = np.mean(v[-1])
        mean_train = np.mean(list(train_errors[idx].values())[0])
        mean_valid = np.mean(list(valid_errors[idx].values())[0])

        indices = [hypers2idx[j][val] for j, val in enumerate(hyperparam_vals)]
        values_tensor[tuple(indices)] = [mean_loss, mean_train, mean_valid]

    return values_tensor, hypers2idx, idx2hypers


def print_minima(values_tensor, idx2hypers):
    min_ = np.nanmin(values_tensor[..., 0])
    min_idx = list(np.argwhere(values_tensor == min_)[0])
    print('min loss: %e (train: %e, test: %e) | for hyperparameters=%s' %
          (min_, values_tensor[tuple(min_idx[:-1] + [1])], 
           values_tensor[tuple(min_idx[:-1] + [2])],
           str([float(idx2hypers[i][idx]) 
           for i, idx in enumerate(min_idx[:-1])])))

    min_ = np.nanmin(values_tensor[..., 1])
    min_idx = list(np.argwhere(values_tensor == min_)[0])
    print('min train: %e (loss: %e, test: %e) | for hyperparameters=%s' %
          (min_, values_tensor[tuple(min_idx[:-1] + [0])], 
           values_tensor[tuple(min_idx[:-1] + [2])],
           str([float(idx2hypers[i][idx]) 
                for i, idx in enumerate(min_idx[:-1])])
          )
         )

    min_ = np.nanmin(values_tensor[..., 2])
    min_idx = list(np.argwhere(values_tensor == min_)[0])
    print('min valid: %e (loss: %e, train: %e) | for hyperparameters=%s' %
          (min_, values_tensor[tuple(min_idx[:-1] + [0])], 
           values_tensor[tuple(min_idx[:-1] + [1])],
           str([float(idx2hypers[i][idx]) 
                for i, idx in enumerate(min_idx[:-1])])
          )
         )


def print_ks_minValid(values_tensor, ks2idx, idx2hypers):
    ks = list(map(lambda x: int(float(x)), ks2idx.keys()))
    for k in ks:
        slice_ = values_tensor[ks2idx['%e' % k]]
        min_ = np.nanmin(slice_[..., 2])
        min_idx = list(np.argwhere(slice_ == min_)[0])
        print('K: {0:3d}\tVALID: {1:e}\tTRAIN: {2:e}\tLOSS: {3:e}\t '\
              '| for hyperparameters={4:s}'.format(k, min_, 
              slice_[tuple(min_idx[:-1] + [1])], 
              slice_[tuple(min_idx[:-1] + [0])],
              str([float(idx2hypers[1+i][idx]) 
                   for i, idx in enumerate(min_idx[:-1])])
             ))


def plot_ks_minValid(values_tensor, ks2idx, err_fn_name='RMSE'):
    k2minValid_TrainLoss = dict()
    ks = list(map(lambda x: int(float(x)), ks2idx.keys()))
    for k in ks:
        slice_ = values_tensor[ks2idx['%e' % k]]
        min_ = np.nanmin(slice_[..., 2])
        min_idx = list(np.argwhere(slice_ == min_)[0])
        k2minValid_TrainLoss[k] = {'valid': min_,
                                   'train': slice_[tuple(min_idx[:-1] + [1])],
                                   'loss': slice_[tuple(min_idx[:-1] + [0])]}

    f, axs = plt.subplots(1, 2, figsize=(24, 8))

    axs[0].plot(k2minValid_TrainLoss.keys(), [v['train'] for v in k2minValid_TrainLoss.values()],
                '-o', label='train')
    axs[0].plot(k2minValid_TrainLoss.keys(), [v['valid'] for v in k2minValid_TrainLoss.values()],
                '-s', color='orange', label='validation')
    axs[0].set_title('Minimum validation error and corresponding train error per latent dimension',
                     fontsize='xx-large')
    axs[0].set_ylabel(err_fn_name, fontsize='x-large')
    axs[0].set_xlabel('Latent dimension (K)', fontsize='x-large')
    axs[0].legend(fontsize='x-large')
    axs[0].grid()

    axs[1].plot(k2minValid_TrainLoss.keys(), [v['loss'] for v in k2minValid_TrainLoss.values()],
                '-o', color='brown')
    axs[1].set_title('Loss corresponding to minimum validation error per latent dimension',
                     fontsize='xx-large')
    axs[1].set_ylabel('Loss', fontsize='x-large')
    axs[1].set_xlabel('Latent dimension (K)', fontsize='x-large')
    axs[1].grid()


def plot_lambdas_heatmaps(values_tensor, lambdas_u2idx, lambdas_i2idx, ks2idx, ks=None):
    import seaborn as sns

    if type(ks) == list:
        ks2idx = ks2idx.copy()
        for k in ks:
            ks2idx.pop('%e' % k)

    ks = list(map(lambda x: int(float(x)), ks2idx.keys()))
    lambdas_u = list(map(lambda x: float(x), lambdas_u2idx.keys()))
    lambdas_i = list(map(lambda x: float(x), lambdas_i2idx.keys()))

    valids_m = []
    trains_m = []
    losses_m = []
    for k in ks:
        valid_m = np.nan * np.ones((len(lambdas_u), len(lambdas_i)))
        train_m = np.nan * np.ones_like(valid_m)
        loss_m = np.nan * np.ones_like(valid_m)
        for i, _ in enumerate(lambdas_u):
            for j, _ in enumerate(lambdas_i):
                slice_ = values_tensor[ks2idx['%e' % k], i, j]
                min_ = np.nanmin(slice_[..., 2])
                if np.isfinite(min_):
                    min_idx = list(np.argwhere(slice_ == min_)[0])
                    valid_m[i, j] = slice_[tuple(min_idx[:-1] + [2])]
                    train_m[i, j] = slice_[tuple(min_idx[:-1] + [1])]
                    loss_m[i, j] = slice_[tuple(min_idx[:-1] + [0])]

        valids_m.append(valid_m.copy())
        trains_m.append(train_m.copy())
        losses_m.append(loss_m.copy())

    train_min, train_max = min([np.nanmin(a) for a in trains_m]), max([np.nanmax(a) for a in trains_m])
    valid_min, valid_max = min([np.nanmin(a) for a in valids_m]), max([np.nanmax(a) for a in valids_m])
    loss_min, loss_max = min([np.nanmin(a) for a in losses_m]), max([np.nanmax(a) for a in losses_m])
    for i, k in enumerate(ks):
        f, axs = plt.subplots(1, 3, figsize=(24, 6))
        mask = np.isnan(valids_m[i])
        sns.heatmap(valids_m[i], cmap="autumn", mask=mask, ax=axs[0],
                    xticklabels=['%.4f' % l for l in lambdas_u],
                    yticklabels=['%.4f' % l for l in lambdas_i],
                    linewidths=1, linecolor='gray', vmin=valid_min, vmax=valid_max)
        sns.heatmap(trains_m[i], cmap="winter", mask=mask, ax=axs[1],
                    xticklabels=['%.4f' % l for l in lambdas_u],
                    yticklabels=['%.4f' % l for l in lambdas_i],
                    linewidths=1, linecolor='gray', vmin=train_min, vmax=train_max)
        sns.heatmap(losses_m[i], cmap="copper", mask=mask, ax=axs[2],
                    xticklabels=['%.4f' % l for l in lambdas_u],
                    yticklabels=['%.4f' % l for l in lambdas_i],
                    linewidths=1, linecolor='gray', vmin=loss_min, vmax=loss_max)
        axs[0].set_ylabel('lambda user', fontsize='xx-large')
        axs[0].set_title('Validation error (K=%d)' % k, fontsize='xx-large')
        axs[1].set_title('Train error (K=%d)' % k, fontsize='xx-large')
        axs[2].set_title('Loss (K=%d)' % k, fontsize='xx-large')
    axs[0].set_xlabel('lambda item', fontsize='xx-large')
    axs[1].set_xlabel('lambda item', fontsize='xx-large')
    axs[2].set_xlabel('lambda item', fontsize='xx-large')


def plot_lr_minValid(values_tensor, lrs2idx, ks2idx, err_fn_name='RMSE', ks=None):
    if type(ks) == list:
        ks2idx = ks2idx.copy()
        for k in ks:
            ks2idx.pop('%e' % k)

    ks = list(map(lambda x: int(float(x)), ks2idx.keys()))
    lrs = list(map(lambda x: float(x), lrs2idx.keys()))

    for k in ks:
        slice_ = values_tensor[ks2idx['%e' % k]]
        min_ = np.nanmin(slice_[..., 2])
        min_idx = list(np.argwhere(slice_ == min_)[0])
        slice_ = slice_[tuple(min_idx[:-2])]

        valid_a = slice_[:, 2]
        train_a = slice_[:, 1]
        loss_a = slice_[:, 0]

        valid_a[np.isnan(valid_a)] = 0
        train_a[np.isnan(train_a)] = 0
        loss_a[np.isnan(loss_a)] = 0

        f, axs = plt.subplots(1, 2, figsize=(16, 4))
        width = 0.35
        x = np.arange(len(lrs))
        axs[0].bar(x - width / 2, train_a, width=width, align='center', label='train')
        axs[0].bar(x + width / 2, valid_a, width=width, color='orange', align='center', label='validation')
        axs[0].set_ylabel(err_fn_name + ' (K=%d)' % k, fontsize='xx-large')
        axs[0].set_xlabel('Learning rate for hyperparams giving minimum validation error', fontsize='x-large')
        axs[0].set_xticklabels(lrs)
        axs[0].legend(fontsize='x-large')
        axs[0].grid()
        axs[1].bar(x, loss_a, width=width, color='brown', align='center')
        axs[1].set_ylabel('Loss (K=%d)' % k, fontsize='xx-large')
        axs[1].set_xlabel('Learning rate for hyperparams giving minimum validation error', fontsize='x-large')
        axs[1].set_xticklabels(lrs)
        axs[1].grid()


def plot_losses_and_errors(loss_list, err_list, n_details=20):
    f, axs = plt.subplots(1, 2, figsize=(24, 8))

    xs = range(max([len(l) for l in loss_list]))
    max_eps = max([len(l) for l in loss_list])
    n_det = n_details

    # Plot losses
    ys = []
    for i in range(max_eps):
        t = []
        for l in loss_list:
            if len(l) > i:
                t.append(l[i])
        ys.append(t)
    ys_mean = np.array([np.mean(t) for t in ys])
    ys_std = np.array([np.std(t) for t in ys])
    axs[0].plot(xs, ys_mean)
    axs[0].fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)
    axs[0].grid()

    xs_det = xs[-n_det:]
    ys_mean_det = ys_mean[-n_det:]
    ys_std_det = ys_std[-n_det:]
    sub_axes = plt.axes([.32, .6, .15, .25])
    sub_axes.plot(xs_det, ys_mean_det)
    sub_axes.fill_between(xs_det, ys_mean_det - ys_std_det, ys_mean_det + ys_std_det, alpha=0.2)

    # Plot errors
    ys = []
    for i in range(max_eps):
        t = []
        for l in err_list:
            if len(l) > i:
                t.append(l[i])
        ys.append(t)
    ys_mean = np.array([np.mean(t) for t in ys])
    ys_std = np.array([np.std(t) for t in ys])
    axs[1].plot(xs, ys_mean)
    axs[1].fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)
    axs[1].grid()

    xs_det = xs[-n_det:]
    ys_mean_det = ys_mean[-n_det:]
    ys_std_det = ys_std[-n_det:]
    sub_axes = plt.axes([.74, .6, .15, .25])
    sub_axes.plot(xs_det, ys_mean_det)
    sub_axes.fill_between(xs_det, ys_mean_det - ys_std_det, ys_mean_det + ys_std_det, alpha=0.2)

