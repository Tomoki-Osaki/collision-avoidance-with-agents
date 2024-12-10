import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
from myfuncs import col
import funcs_calc_index as ci

path = 'crossing_exp/glob_shaped/*.csv'
flist = glob.glob(path)

# 事例2　corssing1
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro2_index_7_8_9_to_12_1.bag.csv"

# 事例1　crossing2
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro8_index_3_5_14_to_1_2.bag.csv"

# 事例3　crossing4
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro10_index_10_14_1_to_8_9.bag.csv"

#df = pd.read_csv(flist[3])
df = pd.read_csv(flist[100])
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro10_index_10_14_1_to_8_9.bag.csv"
df = pd.read_csv(path)

df = df[['/vrpn_client_node/body_0/pose/field.pose.position.x',
         '/vrpn_client_node/body_0/pose/field.pose.position.z',
         'body_0_vel_x', 
         'body_0_vel_z',
         '/vrpn_client_node/body_1/pose/field.pose.position.x',
         '/vrpn_client_node/body_1/pose/field.pose.position.z',
         'body_1_vel_x', 
         'body_1_vel_z']]

old_new_cols = {'/vrpn_client_node/body_0/pose/field.pose.position.x': 'B_posx1',
                '/vrpn_client_node/body_0/pose/field.pose.position.z': 'C_posy1',
                'body_0_vel_x': 'D_velx1',
                'body_0_vel_z': 'E_vely1',
                '/vrpn_client_node/body_1/pose/field.pose.position.x': 'F_posx2',
                '/vrpn_client_node/body_1/pose/field.pose.position.z': 'G_posy2',
                'body_1_vel_x': 'H_velx2',
                'body_1_vel_z': 'I_vely2'}

df = df.rename(columns=old_new_cols)

df['J_CPx'] = ci.J_CPx(df)
df['K_CPy'] = ci.K_CPy(df)
df['L_TTCP0'] = df.apply(ci.L_TTCP0, axis=1)
df['M_TTCP1'] = df.apply(ci.M_TTCP1, axis=1)
df['N_deltaTTCP'] = ci.N_deltaTTCP(df)
df['O_Judge'] = ci.O_Judge(df)
df['P_JudgeEntropy'] = ci.P_JudgeEntropy(df) 
df['Q_equA'] = ci.Q_equA(df)
df['R_equB'] = ci.R_equB(df)
df['S_equC'] = ci.S_equC(df)
df['T_TCPA'] = ci.T_TCPA(df) 
df['U_DCPA'] = ci.U_DCPA(df)
df['V_BrakingRate'] = df.apply(ci.V_BrakingRate, axis=1)
df['W_distance'] = df.apply(ci.W_distance, axis=1)

col(df)

# ci.plot_traj(df)

# %%
import matplotlib
matplotlib.rc('font', family='BIZ UDGothic')

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

rep = 0
headwidth = 8
headlength = 8
linewidth = 3
BrakingRate = []
distance = []
JudgeEntropy = []
dist_ymax = max(df['W_distance'])
xmax = len(df)

df.fillna(-0.1, inplace=True)

def update(frame):
    global rep
    if np.isnan(frame[1]['P_JudgeEntropy']) == False:
        JudgeEntropy.append(frame[1]['P_JudgeEntropy'])
    else:
        JudgeEntropy.append(0)
    if np.isnan(frame[1]['V_BrakingRate']) == False:
        BrakingRate.append(frame[1]['V_BrakingRate'])
    else:
        BrakingRate.append(0)
    distance.append(frame[1]['W_distance'])
    
    rep += 1
    ax[0,0].cla()
    ax[0,0].set_title('歩行者の動き')
    ax[0,0].set_xlim(-5, 5)
    ax[0,0].set_ylim(-5, 5)
    ax[0,0].grid()
    ax[0,0].scatter(frame[1]['B_posx1'], frame[1]['C_posy1'], s=80, c='blue', alpha=0.8)
    ax[0,0].scatter(frame[1]['F_posx2'], frame[1]['G_posy2'], s=80, c='red', alpha=0.8)
    ax[0,0].scatter(df['B_posx1'], df['C_posy1'], c='blue', alpha=0.03)
    ax[0,0].scatter(df['F_posx2'], df['G_posy2'], c='red', alpha=0.03)
    if not rep == len(df):
        ax[0,0].annotate('', xy=(df['B_posx1'][frame[0]+1], df['C_posy1'][frame[0]+1]), 
                         xytext=(frame[1]['B_posx1'], frame[1]['C_posy1']),
                         arrowprops=dict(shrink=0, width=1, headwidth=headwidth, 
                                         headlength=headlength,
                                         facecolor='skyblue', edgecolor='blue'))
        
        ax[0,0].annotate('', xy=(df['F_posx2'][frame[0]+1], df['G_posy2'][frame[0]+1]), 
                         xytext=(frame[1]['F_posx2'], frame[1]['G_posy2']),
                         arrowprops=dict(shrink=0, width=1, headwidth=headwidth, 
                                         headlength=headlength,
                                         facecolor='pink', edgecolor='red'))
    ax[0,1].cla()
    ax[0,1].set_title('２者間の距離')
    ax[0,1].set_xlim(0, xmax+1)
    ax[0,1].set_ylim(0, dist_ymax+0.5)
    ax[0,1].grid()
    ax[0,1].plot(distance, lw=linewidth, color='orange')
    ax[0,1].plot(df['W_distance'], lw=linewidth, color='orange', alpha=0.12)
    
    ax[1,0].cla()
    ax[1,0].set_title('ブレーキ率')
    ax[1,0].set_xlim(0, xmax+1)
    ax[1,0].set_ylim(-0.01, 1)
    ax[1,0].grid()
    ax[1,0].plot(BrakingRate, lw=linewidth, color='green')
    ax[1,0].plot(df['V_BrakingRate'], lw=linewidth, color='green', alpha=0.08)
    
    ax[1,1].cla()
    ax[1,1].set_title('判断エントロピー')
    ax[1,1].set_xlim(0, xmax+1)
    ax[1,1].set_ylim(-0.01, 1)
    ax[1,1].grid()
    ax[1,1].plot(JudgeEntropy, lw=linewidth, color='purple')
    ax[1,1].plot(df['P_JudgeEntropy'], lw=linewidth, color='purple', alpha=0.1)
    
    ax[1,1].text(x=-0.22, y=-0.2, s=f'時間：{frame[0]} (100ms)', 
                 size=13, transform=ax[1,1].transAxes)

anim = FuncAnimation(fig, update, frames=df.iterrows(), repeat=False, 
                     interval=200, cache_frame_data=False)
#plt.show()
anim.save("crossing.mp4", writer='ffmpeg')
plt.close()

