# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       5/24/2019
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

from npmf.error_metrics import *
import numpy as np

import dill
import pickle
import bz2
import os.path


def load_loss_and_errors(algorithm, k, output_dir='../outputs'):
    model_fn = algorithm.__name__ + '_' + str(k) + '.pkl.bz2'
    loss_fn = os.path.join(output_dir, 'loss_' + model_fn)
    err_fn = os.path.join(output_dir, 'err_' + model_fn)
    train_errs_fn = os.path.join(output_dir, 'train_' + model_fn)
    valid_errs_fn = os.path.join(output_dir, 'valid_' + model_fn)

    losses = _load_compressed_pickle(loss_fn)
    errors = _load_compressed_pickle(err_fn)
    train_errors = _load_compressed_pickle(train_errs_fn)
    valid_errors = _load_compressed_pickle(valid_errs_fn)
    return losses, errors, train_errors, valid_errors


def load_cv_model(algorithm, k, fn_prefix='', checkpoints_dir='../checkpoints'):
    model_fn = os.path.join(checkpoints_dir, fn_prefix + algorithm.__name__ + '_' + str(k) + '.dill')
    with open(model_fn, 'rb') as f:
        model_wrapper = dill.load(f)
    return model_wrapper


def test(algorithm, k, trains_matrices, tests_matrices, checkpoints_dir='../checkpoints'):
    # Restore best model
    cv = load_cv_model(algorithm, k, checkpoints_dir=checkpoints_dir)

    # RMSE
    tr_rmse_mean, tr_rmse_se = cv.score(rmse, trains_matrices, 'Train', np.mean, se)
    te_rmse_mean, te_rmse_se = cv.score(rmse, tests_matrices, 'Test', np.mean, se)

    # MAE
    tr_mae_mean, tr_mae_se = cv.score(mae, trains_matrices, 'Train', np.mean, se)
    te_mae_mean, te_mae_se = cv.score(mae, tests_matrices, 'Test', np.mean, se)

    # Precision & Recall
    n_iters = len(trains_matrices)
    precisions = np.zeros((n_iters, 5))
    recalls = np.zeros((n_iters, 5))
    for it in range(n_iters):
        train_mat = trains_matrices[it].copy()
        train_mask = np.isfinite(train_mat)
        train_mat[~train_mask] = -1
        pred_mat = cv.pred_fn(cv.user_features_list[it], cv.item_features_list[it],
                              cv.user_biases_list[it], cv.item_biases_list[it])
        p1 = precision_at_n(1, train_mat, pred_mat, train_mask)
        p2 = precision_at_n(2, train_mat, pred_mat, train_mask)
        p3 = precision_at_n(3, train_mat, pred_mat, train_mask)
        p5 = precision_at_n(5, train_mat, pred_mat, train_mask)
        p10 = precision_at_n(10, train_mat, pred_mat, train_mask)
        precisions[it] = [p1, p2, p3, p5, p10]

        r1 = recall_at_n(1, train_mat, pred_mat, train_mask)
        r2 = recall_at_n(2, train_mat, pred_mat, train_mask)
        r3 = recall_at_n(3, train_mat, pred_mat, train_mask)
        r5 = recall_at_n(5, train_mat, pred_mat, train_mask)
        r10 = recall_at_n(10, train_mat, pred_mat, train_mask)
        recalls[it] = [r1, r2, r3, r5, r10]

    precisions_mean, precisions_se = np.mean(precisions, 0), [se(precisions[:, i]) for i in range(precisions.shape[1])]
    recalls_mean, recalls_se = np.mean(recalls, 0), [se(recalls[:, i]) for i in range(precisions.shape[1])]
    print('Precis', '\t', '1: %e \t 2: %e \t 3: %e \t 5: %e \t 10: %e' % (precisions_mean[0], precisions_mean[1],
                                                                          precisions_mean[2], precisions_mean[3],
                                                                          precisions_mean[4]))
    print('Recall', '\t', '1: %e \t 2: %e \t 3: %e \t 5: %e \t 10: %e' % (recalls_mean[0], recalls_mean[1],
                                                                          recalls_mean[2], recalls_mean[3],
                                                                          recalls_mean[4]))

    return tr_rmse_mean, tr_rmse_se, te_rmse_mean, te_rmse_se, tr_mae_mean, tr_mae_se, te_mae_mean, te_mae_se, \
           precisions_mean, precisions_se, recalls_mean, recalls_se


def _load_compressed_pickle(filename):
    with bz2.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

