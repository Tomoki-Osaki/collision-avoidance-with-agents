import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from myfuncs import col
import funcs_calc_index as ci
import myfuncs as mf
from tqdm import tqdm

# %% choose data
# 事例2　corssing1
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro2_index_7_8_9_to_12_1.bag.csv"
print('事例2')
df = pd.read_csv(path)

# 事例1　crossing2
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro8_index_3_5_14_to_1_2.bag.csv"
print('事例1')
df = pd.read_csv(path)

# 事例3　crossing4
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro10_index_10_14_1_to_8_9.bag.csv"
print('事例3')
df = pd.read_csv(path)

path = 'crossing_exp/glob_shaped/*.csv'
flist = glob.glob(path)
df = pd.read_csv(flist[2200])

df_ori = df.copy()
ci.plot_traj(df_ori)

# %% preprocessing
df = mf.make_column_names_shorter(df)
df = mf.calculate_indexes(df)
col(df)

#mf.animate_traj(df)

# %% confirm the trainsitions of awareness model 
# 速度の計算のために、開始から0.5s進んた時点からのデータを使う
awms = [0]*5
angles = [0]*5
deltas = [0]*5

for i, j in enumerate(range(5, df.shape[0])):
    df_t = df.iloc[j, :-1]
    df_t1 = df.iloc[j-1, :-1]
    df_t5 = df.iloc[j-5, :-1]
    posx1, posy1 = (df_t['B_posx1'], df_t['C_posy1'])
    velx1, vely1 = (df_t['D_velx1'], df_t['E_vely1'])
    posx_t5, posy_t5 = (df_t5['B_posx1'], df_t5['C_posy1'])
    
    posx2, posy2 = (df_t['F_posx2'], df_t['G_posy2'])
    velx2, vely2 = (df_t['H_velx2'], df_t['I_vely2'])
    Px = posx2 - posx1
    Py = posy2 - posy1

    # VselfとVother(m/s)は、0.5sの間に進んだ距離(m)を2倍して求める
    dist1 = mf.calc_distance(df_t['B_posx1'], df_t['C_posy1'], 
                             df_t5['B_posx1'], df_t5['C_posy1'])
    Vself = dist1 * 10/5
    dist2 = mf.calc_distance(df_t['F_posx2'], df_t['G_posy2'], 
                             df_t5['F_posx2'], df_t5['G_posy2'])
    Vother = dist2 * 10/5
    
    posx_t1, posy_t1 = df_t1['B_posx1'], df_t1['C_posy1']    
    line1 = [(posx_t1, posy_t1), (posx1, posy1)]
    # posx_t5, posy_t5 = df_t5['B_posx1'], df_t5['C_posy1']    
    # line1 = [(posx_t5, posy_t5), (posx1, posy1)]
    line2 = [(posx1, posy1), (posx2, posy2)]  
    
    # 角度を計算
    theta = mf.calculate_angle(line1, line2)
    
    angles.append(theta)
    deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    deltas.append(deltaTTCP)
    awm = mf.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic=0) 
    awms.append(awm)
    
#plt.ylim(0, 1)
#plt.plot(awms)

awms = pd.Series(awms)
awms.plot(ylim=(0, 1))
angles = pd.DataFrame({'radition': angles, 'degree': np.rad2deg(angles)})
deltas = pd.Series(deltas)
