import numpy as np
from numbers import Number


def split_inds(inds, p_splits, balanced=False, labels=None, seed=None):
    rng = np.random.RandomState(seed)

    if type(inds) == int:
        inds = np.arange(inds)

    assert len(inds.shape) == 1, 'inds must be a vector of ints'

    if isinstance(p_splits, Number):
        assert p_splits > 0. and p_splits <= 1.
        if p_splits == 1.:
            p_splits = {'train': 1.}
        else:
            p_splits = {'train': p_splits,
                        'val': 1.-p_splits}
    elif p_splits is None:
        p_splits = {'train': 1.}

    ps = np.array(list(p_splits.values()))
    assert (ps > 0.).all() and (ps <= 1.).all(),\
        'all p_splits passed must be in (0., 1.]'
    assert ps.sum() <= 1., 'all p_splits must sum to <= 1.'

    if balanced:
        split_inds = {split: list() for split in p_splits}
        assert labels is not None
        assert len(labels) == len(inds)
        for lab in np.unique(labels):
            lab_inds = np.where(labels == lab)[0]
            num_lab_inds = len(lab_inds)
            shuffled_lab_inds = rng.permutation(lab_inds)
            i_start = 0
            for split, p_split in p_splits.items():
                n_split = max(int(p_split * num_lab_inds),
                              1)  # no empty splits
                i_end = i_start + n_split
                split_inds[split].append(shuffled_lab_inds[i_start:i_end])
                i_start = i_end
        split_inds = {split: np.concatenate(sinds, 0)
                      for split, sinds in split_inds.items()}
    else:
        num_inds = len(inds)
        shuffled_inds = rng.permutation(inds)
        split_inds = {split: None for split in p_splits}
        i_start = 0
        for split, p_split in p_splits.items():
            n_split = max(int(p_split * num_inds), 1)  # no empty splits
            i_end = i_start + n_split
            split_inds[split] = shuffled_inds[i_start:i_end]
            i_start = i_end

        # TODO: assign dangling
        # # assign dangling to random split
        # if i_start < num_inds:
        #     split = rng.choice(list(split_inds.keys()))
        #     split_inds[split] = np.concatenate([split_inds[split],
        #                                          inds[i_start:]], 0)
    return split_inds

