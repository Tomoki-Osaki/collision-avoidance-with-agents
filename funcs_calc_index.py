import numpy as np
import matplotlib.pyplot as plt

"""
B_posx1, 
C_posy1, 
D_velx1, 
E_vely1, 
F_posx2, 
G_posy2, 
H_velx2,
I_vely2, 
J_CPx, 
K_CPy
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
    if ( ((df['B_posx1'] < df['J_CPx'] and df['D_velx1'] > 0) or
          (df['B_posx1'] > df['J_CPx'] and df['D_velx1'] < 0)) 
        and
         ((df['C_posy1'] < df['K_CPy'] and df['E_vely1'] > 0) or
          (df['C_posy1'] > df['K_CPy'] and df['E_vely1'] < 0))
        ):
        
        nume = np.sqrt((df['J_CPx'] - df['B_posx1'])**2 + (df['K_CPy'] - df['C_posy1'])**2)
        deno = np.sqrt((df['D_velx1']**2 + df['E_vely1']**2))
        val = nume / deno
        return val 
    
    else:
        return None

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
    if ( ((df['F_posx2'] < df['J_CPx'] and df['H_velx2'] > 0) or
          (df['F_posx2'] > df['J_CPx'] and df['H_velx2'] < 0)) 
        and
         ((df['G_posy2'] < df['K_CPy'] and df['I_vely2'] > 0) or
          (df['G_posy2'] > df['K_CPy'] and df['I_vely2'] < 0))
        ):
        
        nume = np.sqrt((df['J_CPx'] - df['F_posx2'])**2 + (df['K_CPy'] - df['G_posy2'])**2)
        deno = np.sqrt((df['H_velx2']**2 + df['I_vely2']**2))
        val = nume / deno
        return val
    
    else:
        return None 

def N_deltaTTCP(df):
    """
    =IFERROR( 
        ABS(L2 - M2)
    , -1)
    """
    val = abs(df['L_TTCP0'] - df['M_TTCP1'])
    return val

def O_Judge(df, eta1=-0.303, eta2=0.61):
    """
    =IFERROR(
        1 / (1 + EXP($DB$1[eta1] + $DC$1[eta2]*(M2 - L2)))
    , "")
    """
    val = 1 / (1 + np.exp(eta1 + eta2*(df['M_TTCP1'] - df['L_TTCP0'])))
    return val

def P_JudgeEntropy(df):
    """
    =IFERROR( 
        -O2 * LOG(O2) - (1 - O2) * LOG(1 - O2)
    , "")
    """
    val = -df['O_Judge'] * np.log10(df['O_Judge']) - (1 - df['O_Judge']) * np.log10(1 - df['O_Judge'])
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
    val = -(df['R_equB'] / (2*df['Q_equA']))
    return val

def U_DCPA(df):
    """
    = SQRT( (-(R2^2) + (4*Q2*S2)) / (4*Q2) ) 
    """
    val = np.sqrt(((-df['R_equB']**2) + (4*df['Q_equA']*df['S_equC'])) / (4*df['Q_equA']))
    return val

def V_BrakingRate(df, a1=-0.034, b1=3.348, c1=4.252, d1=-0.003):
    """
    a1: -5.145 (-0.034298)
    b1: 3.348 (3.348394)
    c1: 4.286 (4.252840)
    d1: -13.689 (-0.003423)
    
    =IF(T2 < 0, "", 
    IFERROR(
        (1 / (1 + EXP(-($DD$1[c1] + ($DE$1[d1]*T2*1000))))) * 
        (1 / (1 + EXP(-($DF$1[b1] + ($DG$1[a1]*30*U2)))))
        , "")
    )
    """
    if df['T_TCPA'] < 0:
        return 0
    else:
        term1 = (1 / (1 + np.exp(-(c1 + (d1*df['T_TCPA']*1000)))))
        term2 = (1 / (1 + np.exp(-(b1 + (a1*df['U_DCPA']*30)))))
        val = term1 * term2
        return val
        
def W_distance(df):
    ped1_pos = np.array([df['B_posx1'], df['C_posy1']])
    ped2_pos = np.array([df['F_posx2'], df['G_posy2']])
    distance = np.linalg.norm(ped1_pos - ped2_pos)
    return distance
    

def plot_traj(df):
    for x1, y1, x2, y2 in zip(df['B_posx1'], df['C_posy1'], 
                              df['F_posx2'], df['G_posy2']):
        plt.scatter(x1, y1, color='red')
        plt.scatter(x2, y2, color='blue')
    plt.show()

def col(df):
    for i in df.columns:
        print(i)
