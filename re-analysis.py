# control, urgent, nonurgent, omoiyari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as gc

df_all = mf.make_dict_containing_all_info()

df_small = df_all["ID12"]["omoiyari"]["agents20_tri4"]

# df_id_conditions_NumOfAgents_Trialnumber
# df1omoiyari51 = df_all_participants["ID1"]["omoiyari"]["agents5_tri1"]

cols = [my for my in df_small.columns if 'my' in my]
cols.extend(['timerTrial', 'posX', 'posY'])
tmp = df_small[cols]
tmp.reset_index(drop=True, inplace=True)
#tmp.drop(tmp.index[:3], inplace=True)
tmp.reset_index(drop=True, inplace=True)

posIdeal = tmp.apply(lambda df: mf.line_equation(df["posX"], df["posY"]), axis=1)
idealX, idealY = [], []
for i in range(len(posIdeal)):
    idealX.append(posIdeal[i][0])
    idealY.append(posIdeal[i][1])
tmp['idealX'] = idealX
tmp['idealY'] = idealY

dists = []
for ix, iy, rx, ry in zip(tmp.idealX[:-1], tmp.idealY[:-1], tmp.posX[1:], tmp.posY[1:]):
    c = np.array([ix, iy])
    d = np.array([rx, ry])
    dists.append(np.linalg.norm(c - d))
    
plt.plot(dists)    
print(np.sum(dists))
plt.show()

diff = []
for i in range(len(tmp.posX)):
    dif = tmp.posX[i+1] - tmp.posX[i]
    diff.append(dif)
