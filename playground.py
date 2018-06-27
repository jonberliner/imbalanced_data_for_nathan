import numpy as np
import pandas as pd
from data_prep import data_prep
from matplotlib import pyplot as plt


FORCE_RELOAD = False
FORCE_REPREP = False

DATA_PATH = 'nathan_data.csv'
PREPPED_DATA_PATH = 'prepped_nathan_data.csv'

df = data_prep(DATA_PATH, PREPPED_DATA_PATH, FORCE_RELOAD, FORCE_REPREP)

df0 = df.query('neuron_id=="1_1_1" and timestep < 1000')
css = df0.query('cs_id in [1, 2]')
lps = df0.query('lever_press_id in [1, 2]')

cmap_hash = {1: 'red', 2: 'blue'}

fig, ax = plt.subplots()

act_line = ax.semilogy(df0.timestep, df0.neuron_activity_au + 1., 'k')
css_line = ax.scatter(css.timestep, np.zeros_like(css.timestep) + 1,
                   color=[cmap_hash[cs_id] for cs_id in css.cs_id],
                   marker='+')
lps_line = ax.scatter(lps.timestep, np.zeros_like(lps.timestep) + 1,
                   color=[cmap_hash[lp_id] for lp_id in lps.lever_press_id],
                   marker='o')
