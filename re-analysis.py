# control, urgent, nonurgent, omoiyari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import myfuncs as mf

df_all = mf.make_dict_containing_all_info()

df_small = df_all["ID5"]["omoiyari"]["agents10_tri1"]

# df_id_conditions_NumOfAgents_Trialnumber
# df1omoiyari51 = df_all_participants["ID1"]["omoiyari"]["agents5_tri1"]

mf.plot_traj_compare_conds(df_all, 7, 20)
mf.plot_traj_compare_conds(df_all, 10, 20)
mf.plot_traj_compare_conds(df_all, 18, 20)

cols = [my for my in df_small.columns if 'my' in my]
cols.extend(['timerTrial', 'posX', 'posY'])
tmp = df_small[cols]
tmp.reset_index(drop=True, inplace=True)
tmp.drop(tmp.index[:3], inplace=True)
tmp.reset_index(drop=True, inplace=True)

def line_equation(x1, y1, x2=880, y2=880):
    xmin, xmax = x1-20, x1+20
    ymin, ymax = y1-20, y1+20
    
    slope = (y2- y1) / (x2 - x1)
    y = slope * (xmax - x1) + y1
    x = (ymax - y1) / slope + x1
    y, x = np.round(y, 3), np.round(x, 3)
    
    if y <= ymax and y >= ymin:
        return xmax, y
    else: 
        return x, ymax

posIdeal = tmp.apply(lambda df: line_equation(df["posX"], df["posY"]), axis=1)
idealX, idealY = [], []
for i in range(len((posIdeal))):
    idealX.append(posIdeal[i][0])
    idealY.append(posIdeal[i][1])
tmp['idealX'] = idealX
tmp['idealY'] = idealY

a = np.array([idealX, idealY])
b = np.array([tmp.posX, tmp.posY])
distance = np.linalg(a - b)

goalxy = np.array([880, 880])
dists = []
for ix, iy, rx, ry in zip(tmp.idealX, tmp.idealY, tmp.posX, tmp.posY):
    c = np.array([ix, iy])
    d = np.array([rx, ry])
    dists.append(np.linalg.norm(c - d))
    
    
