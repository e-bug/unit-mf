# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       5/24/2019
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

from npmf.error_metrics import *
from npmf.learning_rate_decay import *
from npmf.init_functions import *
from npmf.utils import *

import numpy as np
from scipy.stats import norm

import cvxopt
cvxopt.solvers.options['show_progress'] = False


# ==================================================================================================================== #
#                                                                                                                      #
#                                              Matrix Factorization models                                             #
#                                                                                                                      #
# ==================================================================================================================== #

def smf(train, init_fn=rand_init, num_features=6, nanvalue=0,
        lr0=0.01, decay_fn=lambda lr, step: inverse_time_decay(lr, step, 0.5, 2000, False), batch_size=32,
        lambda_user=0.1, lambda_item=0.1, max_epochs=2000, stop_criterion=1e-6,
        err_fn=rmse, display=1, seed=42, **kwargs):
    """
    SMF (Survival Matrix Factorization).

    Args:
        train: Ratings matrix to be factorized
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        lr0: Initial learning rate
        decay_fn: Learning rate decay function. If None, keeps it constant
        batch_size: Number of samples employed for each training step
        lambda_user: Regularization strength for users' parameters
        lambda_item: Regularization strength for item' parameters
        max_epochs: Maximum number of epochs
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, gammas, sigma,
        final training error, function to compute prediction matrix
    """

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []
    if decay_fn is None:
        decay_fn = lambda lr, step: lr

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, gamma, std: norm.sf(gamma, loc=user_feats.dot(item_feats.T), scale=std)

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)
    gammas = np.random.rand(item_features.shape[0])
    sigma = 1

    # find the non-zero ratings indices
    nz_train = list(map(tuple, np.argwhere(train != nanvalue)))
    O = train != nanvalue
    num_nz = np.sum(O)

    # run
    print("start SMF...")
    e = 0
    while change > stop_criterion and e < max_epochs:

        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        batches = get_batches(nz_train, batch_size)

        # decrease step size
        lr = decay_fn(lr0, e)

        for b in batches:
            mask = np.zeros_like(train)
            mask[b[:, 0], b[:, 1]] = 1

            U = user_features.dot(item_features.T)
            P = norm.sf(gammas, loc=U, scale=sigma)
            errs = mask * (train - P)
            pdf = norm.pdf(gammas, loc=U, scale=sigma)
            H = errs * pdf

            # update parameters
            user_features += lr * (H.dot(item_features)/batch_size + lambda_user*user_features)
            item_features += lr * ((H.T).dot(user_features)/batch_size + lambda_item*item_features)
            sigma += lr * np.sum(H/sigma * (gammas-U))/batch_size
            gammas += lr * np.sum(-H, axis=0)/batch_size

        # train error
        P = pred_fn(user_features, item_features, gammas, sigma)
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(np.power(O * (train - P), 2)) / num_nz
                      + lambda_user * (np.sum(np.power(user_features, 2)))
                      + lambda_item * (np.sum(np.power(item_features, 2))))

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, gammas, sigma, loss_list[1:], err_list, pred_fn


def emf(train, init_fn=rand_init, num_features=6, nanvalue=0,
        lr0=0.01, decay_fn=lambda lr, step: inverse_time_decay(lr, step, 0.5, 2000, False),
        lambda_user=0.1, lambda_item=0.1, max_epochs=2000, int_iter=200, stop_criterion=1e-6,
        err_fn=rmse, display=1, seed=42, **kwargs):
    """
    EMF (Expertise Matrix Factorization) with users' biases.
    
    Args:
        train: Ratings matrix to be factorized
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        lr0: Initial learning rate
        decay_fn: Learning rate decay function. If None, keeps it constant
        lambda_user: Regularization strength for users' parameters
        lambda_item: Regularization strength for item' parameters
        max_epochs: Maximum number of epochs
        int_iter: Maximum number of iterations in inner iterative procedures
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, users' biases, None, 
        final training error, function to compute prediction matrix
    """

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []
    if decay_fn is None:
        decay_fn = lambda lr, step: lr

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, user_bias, item_bias: \
        user_feats.dot(item_feats.T) + user_bias[:, None]

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)
    user_biases = np.zeros(user_features.shape[0])

    # find the non-zero ratings indices
    O = train != nanvalue
    num_nz = np.sum(O)

    # run
    print("start EMF...")
    e = 0
    while change > stop_criterion and e < max_epochs:

        # decrease step size
        lr = decay_fn(lr0, e)

        # update user_features
        for d in range(train.shape[0]):
            nz_entries = np.argwhere(train[d, :] != nanvalue).T[0]
            Z = item_features[nz_entries, :]
            Z_tilde = np.concatenate((np.ones((Z.shape[0], 1)), Z), axis=1)
            K_tilde = Z_tilde.shape[1]
            Q = (Z_tilde.T).dot(Z_tilde)/num_nz + lambda_user * np.identity(K_tilde)
            Q = cvxopt.matrix(Q)
            p = cvxopt.matrix(-train[d, nz_entries].T.dot(Z_tilde)/num_nz)
            G1 = - np.identity(K_tilde)
            h1 = np.zeros(K_tilde)
            G2 = np.identity(K_tilde)
            G2 = np.delete(G2, 0, 0)
            G2[:, 0] = 1
            h2 = np.ones(K_tilde - 1)
            G = cvxopt.matrix(np.vstack((G1, G2)))
            h = cvxopt.matrix(np.hstack((h1, h2)))
            sol = cvxopt.solvers.qp(Q, p, G, h)
            wd_tilde = np.ravel(sol['x'])
            user_biases[d] = wd_tilde[0]
            user_features[d] = wd_tilde[1:]

        # update item_features
        int_it = 0
        int_change = 1
        int_loss_list = [np.finfo(np.float64).max]
        while int_change > stop_criterion and int_it < int_iter:
            train_tilde = train - user_biases[:, np.newaxis]
            item_features += lr * (((O*(train_tilde-user_features.dot(item_features.T))).T).dot(user_features)/num_nz
                                   - lambda_item * item_features)
            U = np.fliplr(np.sort(item_features, axis=1))
            cumsum_U = np.cumsum(U, axis=1)
            Tmp = (U + (1 - cumsum_U) / np.arange(1, U.shape[1] + 1)) > 0
            rho = Tmp.shape[1] - np.argmax(np.fliplr(Tmp), axis=1) - 1
            mu = 1 / (rho + 1) * (1 - np.array([cumsum_U[n, rho[n]] for n in range(U.shape[0])]))
            item_features = np.maximum(item_features + mu[:, None], 0)
            P = pred_fn(user_features, item_features, user_biases, None)
            loss = 0.5 * (np.sum(np.power(O * (train - P), 2))/num_nz
                          + lambda_user * (np.sum(np.power(user_features, 2)) + np.sum(np.power(user_biases, 2)))
                          + lambda_item * (np.sum(np.power(item_features, 2))))
            int_loss_list.append(loss)
            int_change = np.fabs(int_loss_list[-1] - int_loss_list[-2]) / np.fabs(int_loss_list[-1])
            int_it += 1

        # train error
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(np.power(O * (train - P), 2)) / num_nz
                      + lambda_user * (np.sum(np.power(user_features, 2)) + np.sum(np.power(user_biases, 2)))
                      + lambda_item * (np.sum(np.power(item_features, 2))))

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, user_biases, None, loss_list[1:], err_list, pred_fn



def lmf(train, init_fn=rand_init, num_features=6, nanvalue=0,
       lr0=0.01, decay_fn=lambda lr, step: inverse_time_decay(lr, step, 0.5, 2000, False), batch_size=32,
       lambda_user=0.1, lambda_item=0.1, max_epochs=2000, stop_criterion=1e-6,
       err_fn=rmse, display=1, seed=42, **kwargs):
    """
    LMF (Logistic Matrix Factorization) with biases.
    
    Args:
        train: Ratings matrix to be factorized
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        lr0: Initial learning rate
        decay_fn: Learning rate decay function. If None, keeps it constant
        batch_size: Number of samples employed for each training step
        lambda_user: Regularization strength for users' parameters
        lambda_item: Regularization strength for item' parameters
        max_epochs: Maximum number of epochs
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, users' biases, items' biases, 
        final training error, function to compute prediction matrix
    """

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []
    if decay_fn is None:
        decay_fn = lambda lr, step: lr

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, user_bias, item_bias: \
        1 / (1 + np.exp(-user_feats.dot(item_feats.T) - user_bias[:, None])) * \
        1 / (1 + np.exp(-item_bias))[:, None].T

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)
    user_biases, item_biases = np.zeros(user_features.shape[0]), np.zeros(item_features.shape[0])

    # find the non-zero ratings indices
    nz_row, nz_col = np.argwhere(train != nanvalue)[:, 0], np.argwhere(train != nanvalue)[:, 1]
    nz_train = list(zip(nz_row, nz_col))
    O = train != nanvalue
    num_nz = np.sum(O)

    # run
    print("start LMF...")
    e = 0
    while change > stop_criterion and e < max_epochs:

        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        batches = get_batches(nz_train, batch_size)

        # decrease step size
        lr = decay_fn(lr0, e)

        for b in batches:
            mask = np.zeros_like(train)
            mask[b[:, 0], b[:, 1]] = 1

            item_biases_frac = 1 / (1 + np.exp(-item_biases))[:, None].T
            user_biases_exp = np.exp(-user_biases)[:, None]
            dots_frac_exp = np.exp(-user_features.dot(item_features.T) - user_biases[:, None])
            dots_frac = 1 / (1 + dots_frac_exp)

            errs = mask * (train - dots_frac * item_biases_frac)
            H1 = errs * item_biases_frac
            H2 = H1 * dots_frac * (1 - dots_frac)
            H3 = H2 * user_biases_exp

            # update user_features and item_features
            user_features += lr * (H3.dot(item_features)/batch_size - lambda_user*user_features)
            item_features += lr * ((H3.T).dot(user_features)/batch_size - lambda_item*item_features)
            user_biases += lr * (np.sum((H2 * dots_frac_exp), axis=1)/batch_size - lambda_user*user_biases)
            item_biases += lr * (np.sum((H1 * (1 - item_biases_frac)), axis=0)/batch_size - lambda_item*item_biases)

        # train error
        P = pred_fn(user_features, item_features, user_biases, item_biases)
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(np.power(O * (train - P), 2))/num_nz
                      + lambda_user * (np.sum(np.power(user_features, 2)) + np.sum(np.power(user_biases, 2)))
                      + lambda_item * (np.sum(np.power(item_features, 2)) + np.sum(np.power(item_biases, 2))))

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, user_biases, item_biases, loss_list[1:], err_list, pred_fn

