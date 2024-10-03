import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as g

SUBJECTS = mf.SUBJECTS
CONDITIONS = mf.CONDITIONS # [urgent, nonurgent, omoiyari]
AGENTS = mf.AGENTS
NUM_AGENTS = mf.NUM_AGENTS
TRIALS = mf.TRIALS

df_all = mf.make_dict_of_all_info()

mf.plot_traj_per_trials(df_all, 1, 'urgent', 5)
mf.plot_traj_compare_conds(df_all, 1, 5)

df_part = mf.make_df_trial(df_all, 3, 'nonurgent', 10, 2)
cols = df_part.columns

for cond in CONDITIONS:
    df_small = df_all['ID1'][cond]['agents20']
    for tri in TRIALS:
        plt.plot(df_small[f'trial{tri}']['closest_dists'])
    plt.show()

for cond in CONDITIONS:
    df_small = df_all['ID1'][cond]['agents20']
    for tri in TRIALS:
        plt.plot(df_small[f'trial{tri}']['dist_real_ideal'])
    plt.show()

