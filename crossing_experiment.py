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

#mf.animate_traj(df, save_as='crossing_experiment.mp4')

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




# %% for presentation figures
from matplotlib.animation import FuncAnimation
def animate_traj(df, save_as='crossing.mp4'):
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'BIZ UDGothic'
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    headwidth = 8
    headlength = 8
    linewidth = 3
    BrakingRate = []
    distance = []
    JudgeEntropy = []
    TCPA = []
    DCPA = []
    deltaTTCP = []
    TCPA_ymax = np.max(df['T_TCPA'])
    DCPA_ymax = np.max(df['U_DCPA'])
    deltaTTCP_ymax = np.max(df['N_deltaTTCP'])
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
        #ax[0,0].scatter(df['B_posx1'][-1], df['C_posy1'][-1], c='blue', alpha=0.03)
        #ax[0,0].scatter(df['F_posx2'][-1], df['G_posy2'][-1], c='red', alpha=0.03)
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
        
        # ax[1,0] TCPA (最接近点までの時間)
        ax[1,0].cla()
        ax[1,0].set_title('TCPA (最接近点までの時間)')
        ax[1,0].set_xlim(0, xmax+1)
        ax[1,0].set_ylim(0, TCPA_ymax+0.5)
        ax[1,0].grid()
        ax[1,0].plot(TCPA, lw=linewidth, color='gray')
        ax[1,0].plot(df['T_TCPA'], lw=linewidth, color='gray', alpha=0.12)
        ax[1,0].text(x=-0.3, y=-0.2, s=f'時間：{frame[0]} (100ms)', 
                     size=13, transform=ax[1,1].transAxes)    
        
        # ax[1,1] DCPA (最接近距離)
        ax[1,1].cla()
        ax[1,1].set_title('DCPA (最接近距離)')
        ax[1,1].set_xlim(0, xmax+1)
        ax[1,1].set_ylim(0, DCPA_ymax+0.5)
        ax[1,1].grid()
        ax[1,1].plot(DCPA, lw=linewidth, color='chocolate', alpha=0.7)
        ax[1,1].plot(df['U_DCPA'], lw=linewidth, color='chocolate', alpha=0.12)
        
        # ax[1,2] ブレーキ率
        ax[0,1].cla()
        ax[0,1].set_title('ブレーキ率：0(踏まない)～1(踏む)')
        ax[0,1].set_xlim(0, xmax+1)
        ax[0,1].set_ylim(-0.01, 1)
        ax[0,1].grid()
        ax[0,1].plot(BrakingRate, lw=linewidth, color='purple', alpha=0.7)
        ax[0,1].plot(df['V_BrakingRate'], lw=linewidth, color='purple', alpha=0.1)
        
    anim = FuncAnimation(fig, update, frames=df.iterrows(), repeat=False, 
                         interval=200, cache_frame_data=False)
    # plt.show()
    anim.save(save_as, writer='ffmpeg')
    plt.close()

