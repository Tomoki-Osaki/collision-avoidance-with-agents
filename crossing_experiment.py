import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from myfuncs import col
import funcs_calc_index as ci
import myfuncs as mf
from tqdm import tqdm

# %% choose data
path = 'crossing_exp/glob_shaped/*.csv'
flist = glob.glob(path)

# # 事例2　corssing1
# path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro2_index_7_8_9_to_12_1.bag.csv"
# print('事例2')

# 事例1　crossing2
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro8_index_3_5_14_to_1_2.bag.csv"
print('事例1')

# 事例3　crossing4
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro10_index_10_14_1_to_8_9.bag.csv"
print('事例3')

df = pd.read_csv(path)

df = pd.read_csv(flist[500])

df_ori = df.copy()
ci.plot_traj(df_ori)

# %% preprocessing
df = mf.make_column_names_shorter(df)
df = mf.calculate_indexes(df)
col(df)

#mf.animate_traj(df)

# %% confirm the trainsitions of awareness model 
awms = []
angles = []
for i, j in enumerate(range(5, df.shape[0])):
    df_t = df.iloc[j, :-1]
    df_tminus1 = df.iloc[j-1, :-1]
    df_tminus5 = df.iloc[j-5, :-1]
    posx1, posy1 = (df_t['B_posx1'], df_t['C_posy1'])
    velx1, vely1 = (df_t['D_velx1'], df_t['E_vely1'])
    posx_tminus5, posy_tminus5 = (df_tminus5['B_posx1'], df_tminus5['C_posy1'])
    
    # calculate velocity by dividing the distance by the peds proceeded for 0.5 sec (m/s)
    # compare to df.D_velx1
    Vselfx = [None] * 3
    for k in range(df.shape[0]):
        try:
            a = df.B_posx1[k]
            b = df.B_posx1[k-5]
            Vselfx.append((a - b) * 10/5)
        except KeyError:
            pass
    Vselfx = pd.Series(Vselfx)
    
    # compare to df.E_vely1
    Vselfy = [None] * 3
    for l in range(df.shape[0]):
        try:
            a = df.C_posy1[l]
            b = df.C_posy1[l-5]
            Vselfy.append((a - b) * 10/5)
        except KeyError:
            pass
    Vselfy = pd.Series(Vselfy)
    
    posx2, posy2 = (df_t['F_posx2'], df_t['G_posy2'])
    velx2, vely2 = (df_t['H_velx2'], df_t['I_vely2'])
    Px = posx2 - posx1
    Py = posy2 - posy1
    dist1 = mf.calc_distance(df_t['B_posx1'], df_t['C_posy1'], 
                             df_tminus5['B_posx1'], df_tminus5['C_posy1'])
    Vself = dist1 * 10/5
    dist2 = mf.calc_distance(df_t['F_posx2'], df_t['G_posy2'], 
                             df_tminus5['F_posx2'], df_tminus5['G_posy2'])
    Vother = dist2 * 10/5
    
    # posx_tminus5, posy_tminus5 = df_tminus5['F_posx2'], df_tminus5['G_posy2']
    # mydir = (posy1 - posy_tminus5) / (posx1 - posx_tminus5)
    
    posx_tminus1, posy_tminus1 = df_tminus1['F_posx2'], df_tminus1['G_posy2']    
    mydir = (posy1 - posy_tminus1) / (posx1 - posx_tminus1)

    dir_to_other = (posy2 - posy1) / (posx2 - posx1)
    theta = np.arctan(np.abs(mydir - dir_to_other) / (1 + mydir * dir_to_other))
    deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    awm = mf.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic=0) 
    awms.append(awm)
    print('deltaTTCP, Px, Py, Vother, theta')
    print(deltaTTCP, Px, Py, Vother, theta)
    
plt.ylim(0, 1)
plt.plot(awms)

# %% try to implement the awareness model and confirm the scale
df_t = df.iloc[30, :-1]
df_tminus5 = df.iloc[25, :-1]
posx1, posy1 = (df_t['B_posx1'], df_t['C_posy1'])
velx1, vely1 = (df_t['D_velx1'], df_t['E_vely1'])
posx_tminus5, posy_tminus5 = (df_tminus5['B_posx1'], df_tminus5['C_posy1'])


plt.plot(df.B_posx1, df.C_posy1, alpha=0.5, color='blue')
plt.plot(df.F_posx2, df.G_posy2, alpha=0.5, color='orange')
plt.scatter(df_t.B_posx1, df_t.C_posy1, color='blue')
plt.scatter(df_t.F_posx2, df_t.G_posy2, color='orange')

# calculate velocity by dividing the distance by the peds proceeded for 0.5 sec (m/s)
# compare to df.D_velx1
Vselfx = [None] * 3
for i in range(df.shape[0]):
    try:
        a = df.B_posx1[i]
        b = df.B_posx1[i-5]
        Vselfx.append((a - b) * 10/5)
    except KeyError:
        pass
Vselfx = pd.Series(Vselfx)
plt.plot(df.D_velx1); plt.plot(Vselfx)

# compare to df.E_vely1
Vselfy = [None] * 3
for i in range(df.shape[0]):
    try:
        a = df.C_posy1[i]
        b = df.C_posy1[i-5]
        Vselfy.append((a - b) * 10/5)
    except KeyError:
        pass
Vselfy = pd.Series(Vselfy)
plt.plot(df.E_vely1); plt.plot(Vselfy)

posx2, posy2 = (df_t['F_posx2'], df_t['G_posy2'])
velx2, vely2 = (df_t['H_velx2'], df_t['I_vely2'])
Px = posx2 - posx1
Py = posy2 - posy1
dist1 = mf.calc_distance(df_t['B_posx1'], df_t['C_posy1'], 
                         df_tminus5['B_posx1'], df_tminus5['C_posy1'])
Vself = dist1 * 10/5
dist2 = mf.calc_distance(df_t['F_posx2'], df_t['G_posy2'], 
                         df_tminus5['F_posx2'], df_tminus5['G_posy2'])
Vother = dist2 * 10/5

posx_tminus5, posy_tminus5 = df_tminus5['F_posx2'], df_tminus5['G_posy2']

slope1 = (posy1 - posy_tminus5) / (posx1 - posx_tminus5)
slope2 = (posy2 - posy1) / (posx2 - posx1)
theta = np.arctan(np.abs(slope1 - slope2) / (1 + slope1 * slope2))
deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
awm = mf.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic=1) 

