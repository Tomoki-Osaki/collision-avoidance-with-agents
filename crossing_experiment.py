import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
from myfuncs import col
import funcs_calc_index as ci
from tqdm import tqdm

# %% choose data
path = 'crossing_exp/glob_shaped/*.csv'
flist = glob.glob(path)

# # 事例2　corssing1
# path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro2_index_7_8_9_to_12_1.bag.csv"
# df = pd.read_csv(path)
# print('事例2')
# ci.plot_traj(df)

# # 事例1　crossing2
# path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro8_index_3_5_14_to_1_2.bag.csv"
# df = pd.read_csv(path)
# print('事例1')
# ci.plot_traj(df)

# 事例3　crossing4
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro10_index_10_14_1_to_8_9.bag.csv"
df = pd.read_csv(path)
print('事例3')
ci.plot_traj(df)

# #df = pd.read_csv(flist[3])
# df = pd.read_csv(flist[100])
# ci.plot_traj(df)

# %% preprocessing
def make_column_names_shorter(df):
    df = df[['/vrpn_client_node/body_0/pose/field.pose.position.x',
             '/vrpn_client_node/body_0/pose/field.pose.position.z',
             'body_0_vel_x', 
             'body_0_vel_z',
             '/vrpn_client_node/body_1/pose/field.pose.position.x',
             '/vrpn_client_node/body_1/pose/field.pose.position.z',
             'body_1_vel_x', 
             'body_1_vel_z']]
    
    old_new_cols = {
        '/vrpn_client_node/body_0/pose/field.pose.position.x': 'B_posx1',
        '/vrpn_client_node/body_0/pose/field.pose.position.z': 'C_posy1',
        'body_0_vel_x': 'D_velx1',
        'body_0_vel_z': 'E_vely1',
        '/vrpn_client_node/body_1/pose/field.pose.position.x': 'F_posx2',
        '/vrpn_client_node/body_1/pose/field.pose.position.z': 'G_posy2',
        'body_1_vel_x': 'H_velx2',
        'body_1_vel_z': 'I_vely2'
    }
    
    df = df.rename(columns=old_new_cols)
    
    return df

df = make_column_names_shorter(df)

# %% add columns of indexes
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

# %% animte the moving behaviors and the transactions of indexes
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'BIZ UDGothic'

fig, ax = plt.subplots(2, 3, figsize=(14, 10))

rep = 0
headwidth = 8
headlength = 8
linewidth = 3
BrakingRate = []
distance = []
JudgeEntropy = []
TCPA = []
DCPA = []
deltaTTCP = []
dist_ymax = max(df['W_distance'])
TCPA_ymax = max(df['T_TCPA'])
DCPA_ymax = max(df['U_DCPA'])
deltaTTCP_ymax = max(df['N_deltaTTCP'])
xmax = len(df)
tminus1 = None

df.fillna(-0.1, inplace=True)

def update(frame):
    global tminus1
    if np.isnan(frame[1]['P_JudgeEntropy']) == False:
        JudgeEntropy.append(frame[1]['P_JudgeEntropy'])
    else:
        JudgeEntropy.append(0)
    if np.isnan(frame[1]['V_BrakingRate']) == False:
        BrakingRate.append(frame[1]['V_BrakingRate'])
    else:
        BrakingRate.append(0)
    distance.append(frame[1]['W_distance'])
    TCPA.append(frame[1]['T_TCPA'])
    DCPA.append(frame[1]['U_DCPA'])
    deltaTTCP.append(frame[1]['N_deltaTTCP'])
    
    # ax[0,0] 歩行者の動き
    ax[0,0].cla()
    ax[0,0].set_title('歩行者の動き')
    ax[0,0].set_xlim(-5, 5)
    ax[0,0].set_ylim(-5, 5)
    ax[0,0].grid()
    ax[0,0].scatter(frame[1]['B_posx1'], frame[1]['C_posy1'], s=80, c='blue', alpha=0.8)
    ax[0,0].scatter(frame[1]['F_posx2'], frame[1]['G_posy2'], s=80, c='red', alpha=0.8)
    ax[0,0].scatter(df['B_posx1'], df['C_posy1'], c='blue', alpha=0.03)
    ax[0,0].scatter(df['F_posx2'], df['G_posy2'], c='red', alpha=0.03)
    if not frame[0] == 0:
        ax[0,0].scatter(tminus1[1]['B_posx1'], tminus1[1]['C_posy1'], s=80, c='blue', alpha=0.5)
        ax[0,0].scatter(tminus1[1]['F_posx2'], tminus1[1]['G_posy2'], s=80, c='red', alpha=0.5)
    
    if not frame[0]+1 == len(df):
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
    tminus1 = frame
    
    # ax[0,1] deltaTTCP (2者の交点までの時間差)
    ax[0,1].cla()
    ax[0,1].set_title('ΔTTCP (2者の交点までの時間差)')
    ax[0,1].set_xlim(0, xmax+1)
    ax[0,1].set_ylim(0, deltaTTCP_ymax+0.5)
    ax[0,1].grid()
    ax[0,1].plot(deltaTTCP, lw=linewidth, color='green', alpha=0.7)
    ax[0,1].plot(df['N_deltaTTCP'], lw=linewidth, color='green', alpha=0.1)
    
    # ax[0,2] 判断エントロピー 
    ax[0,2].cla()
    ax[0,2].set_title('判断エントロピー：0(確定的)～1(曖昧)', size=13)
    ax[0,2].set_xlim(0, xmax+1)
    ax[0,2].set_ylim(-0.01, 1)
    ax[0,2].grid()
    ax[0,2].plot(JudgeEntropy, lw=linewidth, color='crimson', alpha=0.7)
    ax[0,2].plot(df['P_JudgeEntropy'], lw=linewidth, color='crimson', alpha=0.08)
    
    # ax[1,0] TCPA (最接近点までの時間)
    ax[1,0].cla()
    ax[1,0].set_title('TCPA (最接近点までの時間)')
    ax[1,0].set_xlim(0, xmax+1)
    ax[1,0].set_ylim(0, TCPA_ymax+0.5)
    ax[1,0].grid()
    #ax[1,0].plot(TCPA, lw=linewidth, color='gray')
    ax[1,0].plot(df['T_TCPA'], lw=linewidth, color='gray', alpha=0.12)
    ax[1,0].text(x=0.3, y=-0.2, s=f'時間：{frame[0]} (100ms)', 
                 size=13, transform=ax[1,1].transAxes)    
    
    # ax[1,1] DCPA (最接近距離)
    ax[1,1].cla()
    ax[1,1].set_title('DCPA (最接近距離)')
    ax[1,1].set_xlim(0, xmax+1)
    ax[1,1].set_ylim(0, DCPA_ymax+0.5)
    ax[1,1].grid()
    #ax[1,1].plot(DCPA, lw=linewidth, color='chocolate', alpha=0.7)
    ax[1,1].plot(df['U_DCPA'], lw=linewidth, color='chocolate', alpha=0.12)
    
    # ax[1,2] ブレーキ率
    ax[1,2].cla()
    ax[1,2].set_title('ブレーキ率：0(踏まない)～1(踏む)')
    ax[1,2].set_xlim(0, xmax+1)
    ax[1,2].set_ylim(-0.01, 1)
    ax[1,2].grid()
    #ax[1,2].plot(BrakingRate, lw=linewidth, color='purple', alpha=0.7)
    ax[1,2].plot(df['V_BrakingRate'], lw=linewidth, color='purple', alpha=0.1)
    
anim = FuncAnimation(fig, update, frames=df.iterrows(), repeat=False, 
                     interval=250, cache_frame_data=False)
# plt.show()
anim.save("crossing.mp4", writer='ffmpeg')
plt.close()

# %% try to implement the awareness model and confirm the scale
import myfuncs as mf

def calc_nic(df, agent):
    my_pos = (df['B_posx1'], df['C_posy1'])
    other_pos = (df['F_posx2'], df['G_posy2'])
    cp = ( (my_pos[0] + other_pos[0]) / 2, (my_pos[1] + other_pos[1]) / 2 )
    dist_cp_me = mf.calc_distance(cp[0], cp[1], my_pos[0], my_pos[1])
    
    Nic_agents = []
    for i in range(1, 21):
        other_pos = (df[f'other{i}NextX'], df[f'other{i}NextY'])
        dist_cp_other = mf.calc_distance(cp[0], cp[1], other_pos[0], other_pos[1])
        if dist_cp_other <= dist_cp_me and not i == agent:
            Nic_agents.append(i)
    
    return Nic_agents
    
posx1, posy1 = (df['myNextX'], df['myNextY'])
velx1, vely1 = (df['myNextX'], df['myNextY'])
posx_tminus1, posy_tminus1 = (df2['myNextX'], df2['myNextY'])
posx2, posy2 = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
velx2, vely2 = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
Px = posx2 - posx1
Py = posy2 - posy1
dist1 = mf.calc_distance(df2['myMoveX'], df2['myMoveY'], 
                         df['myMoveX'], df['myMoveY'])
Vself = dist1
dist2 = mf.calc_distance(df2[f'other{agent}MoveX'], df2[f'other{agent}MoveY'], 
                         df[f'other{agent}MoveX'], df[f'other{agent}MoveY'])
Vother = dist2

slope1 = (posy1 - posy_tminus1) / (posx1 - posx_tminus1)
slope2 = (posy2 - posy1) / (posx2 - posx1)
theta = np.arctan(np.abs(slope1 - slope2) / (1 + slope1 * slope2))
deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
awm = mf.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic) 

for agent in range(1, 21):
    Nic = len(calc_nic(df, agent))
    posx1, posy1 = (df['myNextX'], df['myNextY'])
    velx1, vely1 = (df['myNextX'], df['myNextY'])
    posx_tminus1, posy_tminus1 = (df2['myNextX'], df2['myNextY'])
    posx2, posy2 = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
    velx2, vely2 = (df[f'other{agent}NextX'], df[f'other{agent}NextY'])
    Px = posx2 - posx1
    Py = posy2 - posy1
    dist1 = mf.calc_distance(df2['myMoveX'], df2['myMoveY'], 
                             df['myMoveX'], df['myMoveY'])
    Vself = dist1
    dist2 = mf.calc_distance(df2[f'other{agent}MoveX'], df2[f'other{agent}MoveY'], 
                             df[f'other{agent}MoveX'], df[f'other{agent}MoveY'])
    Vother = dist2
    
    slope1 = (posy1 - posy_tminus1) / (posx1 - posx_tminus1)
    slope2 = (posy2 - posy1) / (posx2 - posx1)
    theta = np.arctan(np.abs(slope1 - slope2) / (1 + slope1 * slope2))
    
    deltaTTCP = mf.deltaTTCP_N(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    awm = mf.awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)  
    print('\ndeltaTTCP, Px, Py, Vself, Vother, theta, Nic')
    print(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)
    print(agent, awm)

# %%
velx, vely = [], []
for i in tqdm(range(200)):
    df = pd.read_csv(flist[i])
    df = make_column_names_shorter(df)
    for x1, x2, y1, y2 in zip(
            df['D_velx1'], df['H_velx2'], df['E_vely1'], df['I_vely2']
        ):
        velx.append(x1)
        velx.append(x2)
        vely.append(y1)
        vely.append(y2)
