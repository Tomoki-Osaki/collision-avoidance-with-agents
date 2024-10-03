import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as g

df_all = mf.make_dict_of_all_info()

mf.plot_traj_per_trials(df_all, 1, 'urgent', 5)
mf.plot_traj_compare_conds(df_all, 1, 5)

df_part = mf.make_df_trial(df_all, 3, 'nonurgent', 10, 2)


