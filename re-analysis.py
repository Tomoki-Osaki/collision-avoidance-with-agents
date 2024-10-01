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
df_part = df_small["agents20_tri1"]

columns = df_part.columns

for x, y in zip(df_part.other2NextX, df_part.other2NextY):
    plt.scatter(x, y)
plt.show()

# df_id_conditions_NumOfAgents_Trialnumber
# df1omoiyari51 = df_all_participants["ID1"]["omoiyari"]["agents5_tri1"]

mf.plot_traj_per_trials(df_all, 2, "urgent", 20)

cols = [my for my in df_small.columns if 'my' in my]
cols.extend(['timerTrial', 'posX', 'posY'])
tmp = df_small[cols]
tmp.reset_index(drop=True, inplace=True)
#tmp.drop(tmp.index[:3], inplace=True)
tmp.reset_index(drop=True, inplace=True)

dists = []
for ix, iy, rx, ry in zip(tmp.idealX[:-1], tmp.idealY[:-1], tmp.posX[1:], tmp.posY[1:]):
    c = np.array([ix, iy])
    d = np.array([rx, ry])
    dists.append(np.linalg.norm(c - d))
    
plt.plot(dists)    
print(np.sum(dists))
plt.show()

diff = []
for i in range(1, len(tmp.posX)):
    dif = tmp.posX[i] - tmp.posX[i-1]
    diff.append(dif)
