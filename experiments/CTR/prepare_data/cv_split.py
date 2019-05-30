#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       5/24/2019
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

import os 
import pickle
import numpy as np
import pandas as pd


# ============================================================================ #
#                                  Load data                                   #
# ============================================================================ #

print('Loading data...')

data_dir = '../data'
efficiency_fn = os.path.join(data_dir, 'efficiency.gz')
nimpressions_fn = os.path.join(data_dir, 'nimpressions.gz')

# Efficiency
eff_df = pd.read_csv(efficiency_fn, header=None, 
                     na_values=-1, compression='gzip')

# Indices of the non-null entries of the matrix
entries_indices = np.argwhere(~np.isnan(eff_df.values))


# ============================================================================ #
#                            k-fold Cross-Validation                           #
# ============================================================================ #

print('Cross-Validation splitting...')

# Number of train/dev/test samples
n_folds = 3
ptest = 10   # percentage of test set
np.random.seed(42)

ntest = entries_indices.shape[0]//100*ptest
nsplit = (entries_indices.shape[0] - ntest)//n_folds
ntrain = nsplit * (n_folds - 1)
nvalid = nsplit
print('Number of training samples:', ntrain)
print('Number of validation samples:', nvalid)
print('Number of test samples:', ntest)

# 3-fold CV indices
splits_dicts = [{k: False for k in range(eff_df.shape[0])} for _ in range(n_folds)]
splits_count = [0 for _ in range(n_folds)]
test_count = 0
splits_indices = [[] for _ in range(n_folds)]
test_indices = []
for entry_idx in np.random.permutation(entries_indices):
    put = False
    for i, split_dict in enumerate(splits_dicts):
        # ensure one ad (row) entry per train-valid split
        if (not put) and (not split_dict[entry_idx[0]]):
            splits_indices[i].append(entry_idx.tolist())
            splits_dicts[i][entry_idx[0]] = True
            splits_count[i] += 1
            put = True
    if not put:
        if test_count < ntest:
            # add to test
            test_indices.append(entry_idx.tolist())
            test_count += 1
            put = True
        else:
            # add to the first non-full split
            for i, count in enumerate(splits_count):
                if (not put) and (count < nsplit):
                    splits_indices[i].append(entry_idx.tolist())
                    splits_count[i] += 1
                    put = True

trains_indices = []
valids_indices = []
trains_indices.append(splits_indices[0] + splits_indices[1])
valids_indices.append(splits_indices[2])
trains_indices.append(splits_indices[1] + splits_indices[2])
valids_indices.append(splits_indices[0])
trains_indices.append(splits_indices[2] + splits_indices[0])
valids_indices.append(splits_indices[1])

# 3-fold CV matrices
train_prefix = 'train_eff_%d.gz'
valid_prefix = 'valid_eff_%d.gz'
test_prefix = 'test_eff.gz'

matrix = eff_df.values
trains_matrices = []
valids_matrices = []
for it in range(n_folds):
    train_matrix = matrix.copy()
    valid_matrix = train_matrix.copy()
    test_matrix = train_matrix.copy()
    for idx in trains_indices[it]:
        valid_matrix[idx[0], idx[1]] = None
        test_matrix[idx[0], idx[1]] = None
    for idx in valids_indices[it]:
        train_matrix[idx[0], idx[1]] = None
        test_matrix[idx[0], idx[1]] = None
    for idx in test_indices:
        train_matrix[idx[0], idx[1]] = None
        valid_matrix[idx[0], idx[1]] = None
        
    trains_matrices.append(train_matrix.copy())
    valids_matrices.append(valid_matrix.copy())
    
    pd.DataFrame(train_matrix).to_csv(os.path.join(data_dir, train_prefix % it), 
                                      sep=',', header=False, index=False, 
                                      na_rep=-1, compression='gzip')
    pd.DataFrame(valid_matrix).to_csv(os.path.join(data_dir, valid_prefix % it), 
                                      sep=',', header=False, index=False, 
                                      na_rep=-1, compression='gzip')
pd.DataFrame(test_matrix).to_csv(os.path.join(data_dir, test_prefix), sep=',', 
                                 header=False, index=False, 
                                 na_rep=-1, compression='gzip')

print('Finished!')

