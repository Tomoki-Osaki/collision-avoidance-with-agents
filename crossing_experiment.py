import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
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
df = pd.read_csv(path)
df = pd.read_csv(flist[0])
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

# for col in df.columns: print(col)

# ci.plot_traj(df)

# %%
alpha = 0.5
fig, ax = plt.subplots(1, 3, figsize=(16, 6))

rep = 0
headwidth = 8
headlength = 8
linewidth = 3
BrakingRate = []
distance = []
dist_ymax = max(df['W_distance'])
xmax = len(df)
def update(frame):
    global rep
    BrakingRate.append(frame[1]['V_BrakingRate'])
    distance.append(frame[1]['W_distance'])
    rep += 1
    ax[0].cla()
    ax[0].set_title('Moving of pedestrians')
    ax[0].set_xlim(-5, 5)
    ax[0].set_ylim(-5, 5)
    ax[0].grid()
    ax[0].scatter(frame[1]['B_posx1'], frame[1]['C_posy1'], s=100, c='blue', alpha=alpha)
    ax[0].scatter(frame[1]['F_posx2'], frame[1]['G_posy2'], s=100, c='red', alpha=alpha)
    if not rep == len(df):
        ax[0].annotate('', xy=(df['B_posx1'][frame[0]+1], df['C_posy1'][frame[0]+1]), 
                       xytext=(frame[1]['B_posx1'], frame[1]['C_posy1']),
                       arrowprops=dict(shrink=0, width=1, headwidth=headwidth, 
                                       headlength=headlength,
                                       facecolor='gray', edgecolor='blue'))
        
        ax[0].annotate('', xy=(df['F_posx2'][frame[0]+1], df['G_posy2'][frame[0]+1]), 
                       xytext=(frame[1]['F_posx2'], frame[1]['G_posy2']),
                       arrowprops=dict(shrink=0, width=1, headwidth=headwidth, 
                                       headlength=headlength,
                                       facecolor='gray', edgecolor='red'))
    ax[1].cla()
    ax[1].set_title('Braking rate')
    ax[1].set_xlim(0, xmax+1)
    ax[1].set_ylim(-0.01, 1)
    ax[1].grid()
    ax[1].plot(BrakingRate, lw=linewidth, color='green')
    
    ax[2].cla()
    ax[2].set_title('Distance between pedestrians')
    ax[2].set_xlim(0, xmax+1)
    ax[2].set_ylim(0, dist_ymax+0.5)
    ax[2].grid()
    ax[2].plot(distance, lw=linewidth, color='orange')

anim = FuncAnimation(fig, update, frames=df.iterrows(), repeat=False, 
                     interval=200, cache_frame_data=False)
plt.show()
# anim.save("crossing.mp4", writer='ffmpeg')
# plt.close()

# %%
xmax = max([max(df['ped0_body_posx']), max(df['ped1_body_posx'])])
xmin = min([min(df['ped0_body_posx']), min(df['ped1_body_posx'])])
ymax = max([max(df['ped0_body_posy']), max(df['ped1_body_posy'])])
ymin = min([min(df['ped0_body_posy']), min(df['ped1_body_posy'])])

frames= zip(df.index,
            df['ped0_body_posx'], df['ped0_body_posy'], 
            df['ped1_body_posx'], df['ped1_body_posy'])

fig = plt.figure()
ax = fig.add_subplot(111)
delay = 0
keep_former_step = False

x1, y1, x2, y2 = [], [], [], []
def update(frame):
    ax.cla()
    ax.set_xlim(xmin-1, xmax+1)
    ax.set_ylim(ymin-1, ymax+1)
    ax.grid()
    
    x1.append(frame[1])
    y1.append(frame[2])
    x2.append(frame[3])
    y2.append(frame[4])
    
    # if you want to keep the former steps
    if keep_former_step == True:
        ax.scatter(x1, y1, color='blue', alpha=0.5)
        ax.scatter(x2, y2, color='red', alpha=0.5)
    
    else:
        if frame[0] <= delay:
            ax.scatter(x1, y1, color='blue', alpha=0.5)
            ax.scatter(x2, y2, color='red', alpha=0.5)
        else:
            ax.scatter(x1[frame[0]-delay:], y1[frame[0]-delay:], color='blue', alpha=alpha)
            ax.scatter(x2[frame[0]-delay:], y2[frame[0]-delay:], color='red', alpha=alpha)

    ax.text(xmax, ymax+1.2, f'time: {frame[0]/10}s')
    
anim = FuncAnimation(fig, update, frames=frames, interval=200, 
                     repeat=False, cache_frame_data=False)

#plt.show()

anim.save("crossing.mp4", writer='ffmpeg')
plt.close()

