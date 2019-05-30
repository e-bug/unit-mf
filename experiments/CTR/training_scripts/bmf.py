#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       5/24/2019
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
np.random.seed(1234)

import os
import sys
sys.path.append(os.path.abspath('../../..'))
from npmf.models import *
from npmf.error_metrics import *
from npmf.learning_rate_decay import inverse_time_decay
from npmf.wrapper_classes import CvMF
from utils.train import *


# Load data
n_iters = 3
train_pref = '../data/train_eff_%d.gz'
valid_pref = '../data/valid_eff_%d.gz'
out_dir =  '../outputs'
ckp_dir = '../checkpoints'

trains_matrices = [pd.read_csv(train_pref % it, header=None, compression='gzip').values for it in range(n_iters)]
valids_matrices = [pd.read_csv(valid_pref % it, header=None, compression='gzip').values for it in range(n_iters)]

# Hyperparameters
algorithm = bmf
save_best_model = True

nanvalue = -1
max_epochs = 100
err_fn = rmse

k = 20

hyperparams = [k]

# Train
losses = []
errors = []
train_errors = []
valid_errors = []

min_valid_err = np.finfo(np.float64).max
best_cv_model = None
for k in hyperparams:
    hyperparams_str = 'k=%e' % k
    print(hyperparams_str)
    
    cv = CvMF(algorithm, num_features=k, nanvalue=nanvalue, xmin=0, xmax=1, max_epochs=max_epochs, err_fn=err_fn)

    loss_list, err_list, train_errs, valid_errs = fit_and_score_cv_model(cv, trains_matrices, valids_matrices, 
                                                                         err_fn, agg_fn=np.mean, dev_fn=se)
    losses.append({hyperparams_str: loss_list})
    errors.append({hyperparams_str: err_list})
    train_errors.append({hyperparams_str: train_errs})
    valid_errors.append({hyperparams_str: valid_errs})
    
    if np.mean(valid_errs) < min_valid_err:
        min_valid_err = np.mean(valid_errs)
        best_cv_model = cv

        save_loss_and_errors(losses, errors, train_errors, valid_errors, algorithm, k, output_dir=out_dir)
        if save_best_model:
            save_cv_model(best_cv_model, algorithm, k, checkpoints_dir=ckp_dir)

