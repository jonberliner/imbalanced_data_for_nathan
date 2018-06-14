from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from numpy import random as rng
from sklearn.metrics import accuracy_score, roc_auc_score
from util import split_inds
from balanced_samplers import upsample_minority_class,\
                              downsample_majority_class


def experiment(X, y, model, p_train=0.7):
    """do random cut of data, train, score binary classification performance.
    return accuracy, auc, trained model, and split indices from experiment
    Args:
        X (2d array): data
        y (1d binary array): binary labels
        model (sklearn model): must be able to call predict_proba
        p_train (int): percentage of data that goes into training
    Returns:
        accuracy (dict): accuracies for train and val(idation) splits
        auc (dict): roc aucs for each split
        # NOTE: not returning these atm
        # model: the trained model
        # splits (dict): the indices of data points in train and val splits"""
    # returns split into train and val splits, trying to keep labels balanced
    # between splits
    splits = split_inds(len(X), p_train, balanced=True, labels=y)
    train_X, train_y = X[splits['train']], y[splits['train']]

    # train model
    model.fit(train_X, train_y)

    # init results dictionaries
    accuracy, auc = {}, {}
    # calc results for splits
    for split in splits:
        # get data for this split
        split_X, split_y = X[splits[split]], y[splits[split]]
        # get
        split_probs = model.predict_proba(split_X)
        accuracy[split] = (np.argmax(split_probs, 1) == split_y).mean()
        auc[split] = roc_auc_score(y_true=split_y, y_score=split_probs[:, 1])

    # TODO: just run for now - not saving models or split indices
    return accuracy, auc, model, splits


def to_results_format(exp_name, acc, auc, sampling_split, val_split):
    out = pd.DataFrame([acc, auc])\
            .T\
            .rename(columns={0: 'acc', 1: 'auc'})\
            .reset_index()\
            .rename(columns={'index': 'split'})
    out['sampling_split'] = sampling_split
    out['val_split'] = val_split
    out['exp_name'] = exp_name
    return out


def sweep(exp_name,
          X,
          y,
          model_fn,
          sampling_fn,
          n_sampling_splits,
          n_val_splits,
          p_train=.7):
    """run experiments with multiple data augs and multiple cross-val cuts.
    return dataframe of organized results from sweep"""

    results = list()
    for i_sampling_split in range(n_sampling_splits):
        if sampling_fn:
            _X, _y = sampling_fn(X, y)
        else:
            _X, _y = X, y

        for i_val_split in range(n_val_splits):
            # initialize our model
            _model = model_fn()

            acc, auc, _model, splits = experiment(_X, _y, _model, p_train)

            _results = to_results_format(exp_name,
                                         acc,
                                         auc,
                                         i_sampling_split,
                                         i_val_split)
            results.append(_results)
    return pd.concat(results)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE

    # GENERATE SOME SYNTHETIC DATA
    P_MAJORITY_CLASS = .8  # how unbalanced do we want?
    N_SAM = 301
    DIM_X = 31
    EPS = 2.  # noise in model

    W = rng.randn(1, DIM_X)
    b = rng.randn(DIM_X)
    y = (rng.rand(N_SAM) < P_MAJORITY_CLASS).astype(int)
    # NOTE: @ is matrix multiplication
    X = (y.astype(np.float32)[:, None] @ W) + b + rng.randn(N_SAM, DIM_X) * EPS

    # vis the data
    vX = TSNE().fit_transform(X)
    plt.ion()
    fig, ax  = plt.subplots()
    ax.scatter(vX[y==0][:, 0], vX[y==0][:, 1], color='red', label='minority class')
    ax.scatter(vX[y==1][:, 0], vX[y==1][:, 1], color='blue', label='majority class')
    plt.title('TSNE of our data')
    plt.legend()
    plt.show()
    plt.ioff()


    # LET'S RUN OUR EXPERIMENTS
    N_SAMPLING_SPLITS = 7  # how many times to randomly up/downsample?
    N_VAL_SPLITS = 5  # how many random train/val splits?

    label_weighting_results =\
        sweep('label_weighting',
              X,
              y,
              model_fn=lambda: LogisticRegression(class_weight='balanced'),
              sampling_fn=None,
              n_sampling_splits=1,  # no data aug, so only need one
              n_val_splits=N_VAL_SPLITS,
              p_train=.7)

    majority_downsampling_results =\
        sweep('majority_downsampling',
              X,
              y,
              model_fn=LogisticRegression,
              sampling_fn=downsample_majority_class,
              n_sampling_splits=N_SAMPLING_SPLITS,
              n_val_splits=N_VAL_SPLITS,
              p_train=.7)

    minority_upsampling_results =\
        sweep('minority_upsampling',
              X,
              y,
              model_fn=LogisticRegression,
              sampling_fn=upsample_minority_class,
              n_sampling_splits=N_SAMPLING_SPLITS,
              n_val_splits=N_VAL_SPLITS,
              p_train=.7)

    results = pd.concat([label_weighting_results,
                         majority_downsampling_results,
                         minority_upsampling_results])
    print(results.head())
