# control, urgent, nonurgent, omoiyari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as gc

df_all = mf.make_dict_containing_all_info(29)

df_small = df_all["ID3"]["omoiyari"]
df_part = df_small["agents5_tri1"]

dists = []
for i in range(len(df_part.index)):
    other = np.array([df_part.other1NextX[i], df_part.other1NextY[i]])
    own = np.array([df_part.myNextX[i], df_part.myNextY[i]])
    dists.append(np.linalg.norm(other - own))

def calc_distance(myX, myY, otherX, otherY):
    mypos = np.array([myX, myY])
    otherpos = np.array([otherX, otherY])
    distance = np.linalg.norm(mypos - otherpos)
    
    return distance
    
dic = pd.DataFrame(columns=([f"other{i}" for i in range(1, 21)]))
for i in range(1, 21):
     tmp = df_part.apply(lambda df: calc_distance(
        df["myNextX"], df["myNextY"], df[f"other{i}NextX"], df[f"other{i}NextY"]
        ), axis=1)
     dic[f"other{i}"] = tmp

newdf = pd.concat([df_part, dic], axis=1)
newdf.dropna(axis=1, inplace=True)


# mf.plot_traj_per_trials(df_all, 1, 'urgent', 5)
# mf.plot_traj_all_trials(df_all, 1, 'urgent', 5)
# mf.plot_traj_compare_conds(df_all, 1, 5)


