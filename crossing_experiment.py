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

"""
B_posx1, C_posy1, D_velx1, E_vely1, F_posx2, 
G_posy2, H_velx2, I_vely2, J_CPx, K_CPy
"""

def J_CPx(df):
    """
    =IFERROR( 
        (H2*$E2*$B2 - $D2*I2*F2 + $D2*H2*(G2-$C2)) / (H2*$E2 - $D2*I2)
    , "")
    """
    nume = df['H_velx2']*df['E_vely1']*df['B_posx1'] - \
           df['D_velx1']*df['I_vely2']*df['F_posx2'] + \
           df['D_velx1']*df['H_velx2']*(df['G_posy2'] - df['C_posy1'])
           
    deno = df['H_velx2']*df['E_vely1'] - df['D_velx1']*df['I_vely2']
    val = nume / deno    
    return val

def K_CPy(df):
    """
    =IFERROR( 
        (E2/D2) * (J2 - B2) + C2
    , "")
    """
    val = (df['E_vely1'] / df['D_velx1']) * (df['J_CPx'] - df['B_posx1']) + df['C_posy1']
    return val

def L_TTCP0(df):
    """
    =IF(
        AND (
            OR ( AND (B2 < J2, D2 > 0), 
                 AND (B2 > J2, D2 < 0)), 
            OR ( AND (C2 < K2, E2 > 0), 
                 AND (C2 > K2, E2 < 0))
            ), 
        SQRT(( J2 - $B2 )^2 + (K2 - $C2)^2) / ( SQRT(($D2^2 + $E2^2 )))
    , "")
    """
    nume = np.sqrt((df['J_CPx'] - df['B_posx1'])**2 + (df['K_CPy'] - df['C_posy1'])**2)
    deno = np.sqrt((df['D_velx1']**2 + df['E_vely1']**2))
    val = nume / deno
    return val 

def M_TTCP1(df):
    """
    =IF( 
        AND (
            OR ( AND (F2 < J2, H2 > 0), 
                 AND (F2 > J2, H2 < 0)), 
            OR ( AND (G2 < K2, I2 > 0), 
                 AND (G2 > K2, I2 < 0))
            ), 
        SQRT( (J2 - F2)^2 + (K2 - G2)^2 ) / (SQRT( (H2^2 + I2^2) ))
    , "")

    """
    nume = np.sqrt((df['J_CPx'] - df['F_posx2'])**2 + (df['K_CPy'] - df['G_posy2'])**2)
    deno = np.sqrt((df['H_velx2']**2 + df['I_vely2']**2))
    val = nume / deno
    return val

def N_deltaTTCP(df):
    """
    =IFERROR( ABS(L2 - M2), -1)
    """
    val = abs(df['L_TTCP0'] - df['M_TTCP1'])
    return val

def O_Judge(df):
    """
    =IFERROR(
        1 / (1 + EXP($DB$1 + $DC$1*(M2 - L2)))
    , "")
    """
    val = ...
    return val

def P_JudgeEntropy(df):
    """
    =IFERROR( 
        -O2*LOG(O2) - (1 - O2)*LOG(1 - O2)
    , "")
    """
    val = -df['O_Judge']*np.log(df['O_Judge'] - (1 - df['O_Judge'])*np.log(1 - df['O_Judge']))
    return val

def Q_equA(df):
    """
    = ($D2 - H2)^2 + ($E2 - I2)^2
    """
    val = (df['D_velx1'] - df['H_velx2'])**2 + (df['E_vely1'] - df['I_vely2'])**2
    return val
    
def R_equB(df):
    """
    = (2*($D2 - H2)*($B2 - F2)) + (2*($E2 - I2)*($C2 - G2))
    """
    val = (2 * (df['D_velx1'] - df['H_velx2']) * (df['B_posx1'] - df['F_posx2'])) + \
          (2 * (df['E_vely1'] - df['I_vely2']) * (df['C_posy1'] - df['G_posy2']))
    return val

def S_equC(df):
    """
    = ($B2 - F2)^2 + ($C2 - G2)^2
    """
    val = (df['B_posx1'] - df['F_posx2'])**2 + (df['C_posy1'] - df['G_posy2'])**2
    return val

def T_TCPA(df):
    """
    = -(R2 / (2*Q2))
    """
    val = -(df['R_equB'] / 2*df['Q_equA'])
    return val

def U_DCPA(df):
    """
    = SQRT( (-(R2^2) + (4*Q2*S2)) / (4*Q2) ) 
    """
    val = np.sqrt(((-df['R_equB']**2) + (4*df['Q_equA']*df['S_equC'])) / (4*df['Q_equA']))
    return val

def V_BreakingRate(df, a1=-0.034298, b1=3.348394, c1=4.252840, d1=-0.003423):
    """
    a1: -5.145 (-0.034298)
    b1: 3.348 (3.348394)
    c1: 4.286 (4.252840)
    d1: -13.689 (-0.003423)
    
    =IF(T2 < 0, "", 
    IFERROR(
        (1 / (1 + EXP(-($DD$1(c1) + ($DE$1(d1)*T2*1000))))) * 
        (1 / (1 + EXP(-($DF$1(b1) + ($DG$1(a1)*30*U2)))))
        , "")
    )
    """
    term1 = (1 / (1 + np.exp(-(c1 + (d1*df['T_TCPA']*1000)))))
    term2 = (1 / (1 + np.exp(-(b1 + (a1*df['U_DCPA']*30)))))
    val = term1 * term2
    return val

df['J_CPx'] = J_CPx(df)
df['K_CPy'] = K_CPy(df)
df['L_TTCP0'] = L_TTCP0(df)
df['M_TTCP1'] = M_TTCP1(df)
df['N_deltaTTCP'] = N_deltaTTCP(df)
df['O_Judge'] = ...
df['P_JudgeEntropy'] = P_JudgeEntropy(df)
df['Q_equA'] = Q_equA(df)
df['R_equB'] = R_equB(df)
df['S_equC'] = S_equC(df)
df['T_TCPA'] = T_TCPA(df)
df['U_DCPA'] = U_DCPA(df)
df['V_BreakingRate'] = V_BreakingRate(df)
