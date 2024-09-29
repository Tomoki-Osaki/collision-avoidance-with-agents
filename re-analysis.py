# control, urgent, nonurgent, omoiyari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf

df_all = mf.make_dict_containing_all_info()

tmp = df_all["ID5"]["omoiyari"]["agents10_tri1"]

# df_id_conditions_NumOfAgents_Trialnumber
# df1omoiyari51 = df_all_participants["ID1"]["omoiyari"]["agents5_tri1"]

mf.plot_traj_compare_conds(df_all, 7, 20)
mf.plot_traj_compare_conds(df_all, 10, 20)
mf.plot_traj_compare_conds(df_all, 18, 20)


