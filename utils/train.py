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


def fit_and_score_cv_model(model_wrapper, trains_matrices, valids_matrices, 
                           err_fn=rmse, agg_fn=np.mean, dev_fn=se):
    model_wrapper.fit(trains_matrices)
    model_wrapper.score(err_fn, trains_matrices, err_type='train', agg_fn=agg_fn, dev_fn=dev_fn)
    model_wrapper.score(err_fn, valids_matrices, err_type='validation', agg_fn=agg_fn, dev_fn=dev_fn)

    n_iters = len(trains_matrices)
    loss_list = []
    err_list = []
    train_errs = []
    valid_errs = []
    for i in range(n_iters):
        loss_list.append(model_wrapper.loss_lists_list[i])
        err_list.append(model_wrapper.err_lists_list[i])
        train_errs.append(model_wrapper.train_errors_list[i][err_fn.__name__])
        valid_errs.append(model_wrapper.valid_errors_list[i][err_fn.__name__])
    return loss_list, err_list, train_errs, valid_errs


def save_loss_and_errors(losses, errors, train_errors, valid_errors, algorithm, k, output_dir='../outputs'):
    model_fn = algorithm.__name__ + '_' + str(k) + '.pkl.bz2'
    loss_fn = os.path.join(output_dir, 'loss_' + model_fn)
    err_fn = os.path.join(output_dir, 'err_' + model_fn)
    train_errs_fn = os.path.join(output_dir, 'train_' + model_fn)
    valid_errs_fn = os.path.join(output_dir, 'valid_' + model_fn)

    _save_compressed_pickle(losses, loss_fn, protocol=2)
    _save_compressed_pickle(errors, err_fn, protocol=2)
    _save_compressed_pickle(train_errors, train_errs_fn, protocol=2)
    _save_compressed_pickle(valid_errors, valid_errs_fn, protocol=2)


def save_cv_model(model_wrapper, algorithm, k, fn_prefix='', checkpoints_dir='../checkpoints'):
    model_fn = os.path.join(checkpoints_dir, fn_prefix + algorithm.__name__ + '_' + str(k) + '.dill')
    with open(model_fn, 'wb') as f:
        dill.dump(model_wrapper, f)


def _save_compressed_pickle(obj, filename, protocol=-1):
    with bz2.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

