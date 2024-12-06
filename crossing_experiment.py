import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

path = 'crossing_exp/glob_shaped/*.csv'
flist = glob.glob(path)

# 事例2　corssing1
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro2_index_7_8_9_to_12_1.bag.csv"

# 事例1　crossing2
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro8_index_3_5_14_to_1_2.bag.csv"

# 記載なし　crossing3
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro9_index_11_5_11_to_14_1.bag.csv"

# 事例3　crossing4
path = "crossing_exp/glob_shaped/20220405_all_group_ha_keiro10_index_10_14_1_to_8_9.bag.csv"

#df = pd.read_csv(flist[3])
df = pd.read_csv(path)
df['ped0_body_posx'] = df['/vrpn_client_node/body_0/pose/field.pose.position.x']
df['ped0_body_posy'] = df['/vrpn_client_node/body_0/pose/field.pose.position.z']
df['ped1_body_posx'] = df['/vrpn_client_node/body_1/pose/field.pose.position.x']
df['ped1_body_posy'] = df['/vrpn_client_node/body_1/pose/field.pose.position.z']

# for col in df.columns: print(col)

def plot_traj(df):
    for x1, y1, x2, y2 in zip(df['ped0_body_posx'], df['ped0_body_posy'], 
                              df['ped1_body_posx'], df['ped1_body_posy']):
        plt.scatter(x1, y1, color='red')
        plt.scatter(x2, y2, color='blue')
    plt.show()


tcpa = 0 # Time to Closest of Point of Approach
dcpa = 0 # Distance at Closest Point of Approach

def calc_braking_rate(a1=-5.145, b1=3.348, c1=4.286, d1=-13.689):
    braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                    (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
                    
    return braking_index


xmax = max([max(df['ped0_body_posx']), max(df['ped1_body_posx'])])
xmin = min([min(df['ped0_body_posx']), min(df['ped1_body_posx'])])
ymax = max([max(df['ped0_body_posy']), max(df['ped1_body_posy'])])
ymin = min([min(df['ped0_body_posy']), min(df['ped1_body_posy'])])

frames= zip(df.index,
            df['ped0_body_posx'], df['ped0_body_posy'], 
            df['ped1_body_posx'], df['ped1_body_posy'])

fig = plt.figure()
ax = fig.add_subplot(111)
alpha = 0.5
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

# %% calculate the braking rate
df = pd.read_csv(flist[0])
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

def J_CPx():
    nume = df['H_velx2']*df['$E_vely1']*df['$B_posx1'] - \
           df['$D_velx1']*df['I_vely2']*df['F_posx2'] + \
           df['$D_velx1']*df['H_velx2']*(df['G_posy2'] - df['$C_posy1'])
           
    deno = df['H_velx2']*df['$E_vely1'] - df['$D_velx1']*df['I_vely2']
    
    val = nume / deno
    df['J_CPx'] = val
    
    return df

def K_CPy():
    val = (df['E_vely1'] / df['D_velx1']) * (df['J_CPx'] - df['B_posx1']) + df['C_posy1']
    df['K_CPy'] = val
    return df

def L_TTCP0():
    val = ...
    df['L_TTCP0'] = val
    return df 

def M_TTCP1():
    val = ...
    df['M_TTCP1'] = val
    return df 

def N_deltaTTCP():
    val = abs(df['L_TTCP0'] - df['M_TTCP1'])
    df['N_deltaTTCP'] = val
    
    return df 

def O_Judge():
    val = ...
    df['O_Judge'] = val
    return df  

def P_JudgeEntropy():
    val = ...
    df['P_JudgeEntropy'] = val
    return df   

def Q_equA():
    val = (df['$D_velx1'] - df['H_velx2'])**2 + (df['$E_vely1'] - df['I_vely2'])**2
    df['Q_equA'] = val
    return df   
    
def R_equB():
    val = (2 * (df['D_velx1'] - df['H_velx2']) * (df['$B_posx1'] - df['F_posx2'])) + \
          (2 * (df['$E_vely1'] - df['I_vely2']) * (df['$C_posy1'] - df['G_posy2']))
    df['R_equB'] = val
    return df   

def S_equC():
    val = (df['$B_posx1'] - df['F_posx2'])**2 + (df['$C_posy1'] - df['G_posy2'])**2
    df['S_equC'] = val
    return df    

def T_TCPA():
    val = -(df['R_equB'] / 2*df['Q_equA'])
    df['T_TCPA'] = val
    return df 

def U_DCPA():
    val = np.sqrt((-df['R_equB']**2) + (4 * df['Q_equA'] * df['S_equC']) / 4 * df['Q_equA'])
    df['U_DCPA'] = val
    return df 

def V_BreakingRate():
    val = ...
    df['V_BreakingRate'] = val
    return df 



