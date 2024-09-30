# control, urgent, nonurgent, omoiyari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf
from gc import collect as gc

df_all = mf.make_dict_containing_all_info(29)

df = mf.make_all_debug(29)

df_small = df_all["ID12"]["omoiyari"]["agents20_tri4"]

# df_id_conditions_NumOfAgents_Trialnumber
# df1omoiyari51 = df_all_participants["ID1"]["omoiyari"]["agents5_tri1"]

cols = [my for my in df_small.columns if 'my' in my]
cols.extend(['timerTrial', 'posX', 'posY'])
tmp = df_small[cols]
tmp.reset_index(drop=True, inplace=True)
#tmp.drop(tmp.index[:3], inplace=True)
tmp.reset_index(drop=True, inplace=True)

def calc_ideal_positions(df):
    posIdeal = df.apply(lambda df: mf.line_equation_to_goal(df["posX"], df["posY"]), axis=1)
    idealX, idealY = [], []
    for i, _ in enumerate(posIdeal):
        idealX.append(posIdeal[i][0])
        idealY.append(posIdeal[i][1])
    df['idealX'] = idealX
    df['idealY'] = idealY
    
    return df

posIdeal = df_5_tri.apply(lambda df: mf.line_equation_to_goal(df["posX"], df["posY"]), axis=1)
posIdeal.reset_index(drop=True, inplace=True)
idealX, idealY = [], []
for i, _ in enumerate(posIdeal):
    idealX.append(posIdeal[i][0])
    idealY.append(posIdeal[i][1])
df_5_tri['idealX'] = idealX
df_5_tri['idealY'] = idealY

tmp = calc_ideal_positions(df_5_tri)

tmp = calc_ideal_positions(tmp)

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
