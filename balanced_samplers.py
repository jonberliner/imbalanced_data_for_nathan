import numpy as np
from numpy import random as rng


# class balancing through data up/down sampling
def upsample_minority_class(X, y):
    """will upsample minority class so that have balanced dataset with each
    class of size n_majority_class.  currently expects binary labels"""
    classes, n_per_class = np.unique(y, return_counts=True)
    # ensure binary labels
    assert len(classes) == 2
    assert (0 in classes and 1 in classes)

    # get counts and indices of classes
    iu_majority_class = np.argmax(n_per_class)
    n_majority_class = n_per_class[iu_majority_class]
    majority_class = classes[iu_majority_class]

    # there are more elegant ways to get with the neg of the maj class, but w.e
    iu_minority_class = np.argmin(n_per_class)
    minority_class = classes[iu_minority_class]

    i_majority_class = np.where(y == majority_class)[0]
    i_minority_class = np.where(y == minority_class)[0]

    # choose minority sample indices
    i_upsampled_minority_class = rng.choice(i_minority_class,
                                            size=n_majority_class,
                                            replace=True)
    # build augmented dataset
    majority_X = X[i_majority_class]
    minority_X = X[i_upsampled_minority_class]

    majority_y = np.repeat(majority_class, n_majority_class)
    minority_y = np.repeat(minority_class, n_majority_class)

    X = np.concatenate([majority_X, minority_X], 0)
    y = np.concatenate([majority_y, minority_y], 0)

    # shuffle
    order = rng.permutation(len(X))
    X = X[order]
    y = y[order]

    return X, y


def downsample_majority_class(X, y):
    """will downsample majority class so that have balanced dataset with each
    class of size n_minority_class.  currently expects binary labels"""
    classes, n_per_class = np.unique(y, return_counts=True)
    # ensure binary labels
    assert len(classes) == 2
    assert (0 in classes and 1 in classes)

    # get counts and indices of classes
    iu_majority_class = np.argmax(n_per_class)
    majority_class = classes[iu_majority_class]

    # there are more elegant ways to get with the neg of the maj class, but w.e
    iu_minority_class = np.argmin(n_per_class)
    minority_class = classes[iu_minority_class]
    n_minority_class = n_per_class[iu_minority_class]

    i_majority_class = np.where(y == majority_class)[0]
    i_minority_class = np.where(y == minority_class)[0]

    # choose minority sample indices
    i_downsampled_majority_class = rng.choice(i_majority_class,
                                              size=n_minority_class,
                                              replace=False)
    # build downsampled dataset
    majority_X = X[i_downsampled_majority_class]
    minority_X = X[i_minority_class]

    majority_y = np.repeat(majority_class, n_minority_class)
    minority_y = np.repeat(minority_class, n_minority_class)

    X = np.concatenate([majority_X, minority_X], 0)
    y = np.concatenate([majority_y, minority_y], 0)

    # shuffle
    order = rng.permutation(len(X))
    X = X[order]
    y = y[order]

    return X, y
