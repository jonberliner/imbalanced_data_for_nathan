import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from time import time

COLUMN_NAMES = ['brain_region_id',
                'mouse_id',
                'neuron_in_brain_region_id',
                'neuron_activity_au',
                'lever_press_id',  # 0 for no press, 1 for left, 2 for right
                'cs_id']  # 0 for no CS, 1 for positive, 2 for negative


def str_int(val):
    return str(int(val))


# add timestep
def add_timestep(neuron_df):
    neuron_df['timestep'] = np.arange(len(neuron_df)).astype(int)
    return neuron_df


def prep_df(data_path, prepped_data_path):
    t0 = time()
    print(f'loading raw data from {data_path}...')
    df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES)

    underscores = np.repeat('_', df.shape[0])

    # add unique neuron ids
    print('adding unique neuron ids...')
    df['neuron_id'] =\
        df['mouse_id'].map(str_int)\
        + underscores\
        + df['brain_region_id'].map(str_int)\
        + underscores\
        + df['neuron_in_brain_region_id'].map(str_int)

    # add timesteps for each neuron
    print('adding timesteps for each neuron...')
    df = df.groupby('neuron_id')\
           .apply(add_timestep)\
           .reset_index()\
           .drop(['index', 'Unnamed: 0'], axis=1)

    print(f'saving to {prepped_data_path}...')
    df.to_csv(prepped_data_path)
    print(f'prepped {len(df)} rows in {time()-t0} seconds')
    return df


def data_prep(data_path, prepped_data_path, force_reload, force_reprep):
    if 'df' not in globals() or force_reload:
        print(f'loading or creating prepped data...')
        if (not os.path.exists(prepped_data_path)) or force_reprep:
            # reprep from original data from nathan
            print('reprepping data from {data_path} to {prepped_data_path}...')
            df = prep_df(data_path, prepped_data_path)
        else:
            # load last saved prep
            print(f'loading prepped data from {prepped_data_path}...')
            df = pd.read_csv(prepped_data_path)
    else:
        print('df already in locals().  change force_reload to True if want',
              'to force reload and/or reprep')
    return df


if __name__ == '__main__':
    FORCE_RELOAD = True
    FORCE_REPREP = True

    DATA_PATH = 'nathan_data.csv'
    PREPPED_DATA_PATH = 'prepped_nathan_data.csv'

    df = data_prep(DATA_PATH, PREPPED_DATA_PATH, FORCE_RELOAD, FORCE_REPREP)
