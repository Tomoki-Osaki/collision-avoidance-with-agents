""" 
2025/01/10
シミュレーションのための関数 

1. define_fig_ax(width, height, FIELD_SIZE)
2. calc_rad(pos2, pos1)
3. rotate_vec(vec, rad)
4. calc_distance(posX1, posY1, posX2, posY2)
5. calc_cross_point(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
6. calc_deltaTTCP(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
7. calc_angle_two_lines(line1, line2)
8. awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)
9. debug_theta(s, num, other)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size 
from mpl_toolkits.axes_grid1.mpl_axes import Axes

# %% functions to calculate the vectors
def calc_rad(pos2: np.array, # [float, float]
             pos1: np.array # [float, float]
             ) -> float: 
    """
    pos1からpos2のベクトルの角度を返す
    ex. calc_rad(pos2=np.array([1.5, 2.5]), pos1=np.array([3.0, 1.0])) -> 2.4
    """
    val = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
    
    return val


def rotate_vec(vec: np.array, # [float, float] 
               rad: float
               ) -> np.array: # [float, float]
    """
    ベクトルをradだけ回転させる (回転行列)
    ex. rotate_vec(vec=np.array([3.0, 5.0]), rad=1.2) -> array([-3.6, 4.6])
    """
    rotation = np.array([[np.cos(rad), -np.sin(rad)], 
                         [np.sin(rad), np.cos(rad)]])
    val = np.dot(rotation, vec.T).T
    
    return val


def calc_distance(posX1, posY1, posX2, posY2):
    """
    2点間の距離を計算する
    ex. calc_distance(3.0, 2.5, 4.5, 5.0) -> 2.9
    """
    mypos = np.array([posX1, posY1])
    anotherpos = np.array([posX2, posY2])
    distance = np.linalg.norm(mypos - anotherpos)
    
    return distance

# %% awareness model
def calc_cross_point(velx1: float, vely1: float, posx1: float, posy1: float, 
                     velx2: float, vely2: float, posx2: float, posy2: float
                     ) -> np.array or None: # [float(x), float(y)]
    """
    ex. calc_cross_point(1, 2, 3, 4, 
                         5, 6, 7, 8) -> array([2.0, 2.0])
    """
    nume_x = velx2*vely1*posx1 - velx1*vely2*posx2 + velx1*velx2*(posy2 - posy1)
    deno_x = velx2*vely1 - velx1*vely2
    if deno_x == 0:
        return None
    else:
        CPx = nume_x / deno_x    

        CPy = (vely1 / velx1) * (CPx - posx1) + posy1
        CP = np.array([CPx, CPy])
        
        return  CP


def calc_deltaTTCP(velx1: float, vely1: float, posx1: float, posy1: float, 
                   velx2: float, vely2: float, posx2: float, posy2: float
                   ) -> float or None:
    """
    ex. calc_deltaTTCP(-1.4, 0.4, 1.3, -2.1, 
                       -0.9, -0.3, 2.0, 0.8) -> -2.5
    """
    CP = calc_cross_point(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
    CPx, CPy = CP[0], CP[1]
    TTCP0 = None
    TTCP1 = None
    
    if (
            ( (posx1 < CPx and velx1 > 0) or (posx1 > CPx and velx1 < 0) ) 
        and
            ( (posy1 < CPy and vely1 > 0) or (posy1 > CPy and vely1 < 0) )
        ):
        
        nume0 = np.sqrt( (CPx - posx1)**2 + (CPy - posy1)**2 )
        deno0 = np.sqrt( (velx1**2 + vely1**2) )
        
        TTCP0 = nume0 / deno0
    else:
        return None

    if ( 
            ( (posx2 < CPx and velx2 > 0) or (posx2 > CPx and velx2 < 0) ) 
        and
            ( (posy2 < CPy and vely2 > 0) or (posy2 > CPy and vely2 < 0) )
        ):
        
        nume1 = np.sqrt( (CPx - posx2)**2 + (CPy - posy2)**2 )
        deno1 = np.sqrt( (velx2**2 + vely2**2) )
        
        TTCP1 = nume1 / deno1
    else:
        return None
    
    # TTCP0とTTCP1がどちらもNoneでない場合のみdeltaTTCPを返す
    deltaTTCP = TTCP0 - TTCP1 #　正は道を譲り、負は自分が先に行く
    
    return deltaTTCP    
    

def calc_angle_two_lines(line1, line2):
    """
    2直線がなす角度(radian)を求める関数
    line1, line2: 各直線を表す2点 [(x1,y1), (x2,y2)] or np.array([[x1,y1], [x2,y2]])
    
    ex. calc_angle_two_lines(line1=np.array([[3,　4], [5,　6]]), 
                             line2=np.array([[-4,　-5], [2, 3]]) -> 0.14
    """
    # 傾きを計算
    def slope(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x2 - x1 == 0:  # 垂直な直線の場合
            return np.inf  # 無限大を返す
        return (y2 - y1) / (x2 - x1)
    
    m1 = slope(*line1)
    m2 = slope(*line2)
    
    # 垂直チェック
    if m1 == np.inf and m2 == np.inf:
        return 0  # 同じ垂直方向
    elif m1 == np.inf or m2 == np.inf:
        return np.deg2rad(90) # 片方が垂直の場合(90deg=1.57rad)
    
    # 角度を計算
    angle_rad = np.arctan(np.abs((m2 - m1) / (1 + m1 * m2)))
    
    return angle_rad


def awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic):
    """
    Inputs
        deltaTTCP: 自分のTTCPから相手のTTCPを引いた値 (second)
        Px: 自分から見た相手の相対位置 (x座標m)
        Py: 自分から見た相手の相対位置 (y座標m)
        Vself: 自分の歩行速度 (m/s)
        Vother: 相手の歩行速度 (m/s)
        theta: 自分の向いている方向と相手の位置の角度差 (rad)
        Nic: 円内他歩行者数 (人)
    Output
        0(注視しない) - 1 (注視する)
    """
    if deltaTTCP == None:
        return 0
    
    deno = 1 + np.exp(
        -(-1.2 +0.018*deltaTTCP -0.1*Px -1.1*Py \
          -0.25*Vself +0.29*Vother -2.5*theta -0.62*Nic)    
    )
    val = 1 / deno
    
    return val

def debug_theta(s, num, other):
    a = s.all_agent[num]['p'] + s.all_agent[num]['v']
    b = s.all_agent[num]['p']
    line1 = np.array([b, a])
    
    c = s.all_agent[other]['p']
    line2 = np.array([b, c])
    theta = calc_angle_two_lines(line1, line2)
    
    print('rad:', theta, 'deg:', np.rad2deg(theta))
    
"""
% トレーニングデータの平均と標準偏差で標準化
        mu = mean(X_train);
        sigma = std(X_train);
        X_train = (X_train - mu) ./ sigma;
"""

# %% define figure parameters
def define_fig_ax(width: int = 500, 
                  height: int = 500,
                  field_size: int = 5) -> plt.subplots:
    ax_w_px = width  # プロット領域の幅をピクセル単位で指定
    ax_h_px = height  # プロット領域の高さをピクセル単位で指定
    
    #  サイズ指定のための処理、20行目までhttps://qiita.com/code0327/items/43118813b6085dc7e3d1　を参照
    fig_dpi = 100
    ax_w_inch = ax_w_px / fig_dpi
    ax_h_inch = ax_h_px / fig_dpi
    ax_margin_inch = (0.5, 0.5, 0.5, 0.5)  #  Left, Top, Right, Bottom [inch]
    
    fig_w_inch = ax_w_inch + ax_margin_inch[0] + ax_margin_inch[2] 
    fig_h_inch = ax_h_inch + ax_margin_inch[1] + ax_margin_inch[3]
    
    fig = plt.figure(dpi=fig_dpi, figsize=(fig_w_inch, fig_h_inch))
    ax_p_w = [Size.Fixed(ax_margin_inch[0]), Size.Fixed(ax_w_inch)]
    ax_p_h = [Size.Fixed(ax_margin_inch[1]), Size.Fixed(ax_h_inch)]
    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), ax_p_w, ax_p_h, aspect=False)
    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    fig.add_axes(ax)
    
    ax.set_xlim(-field_size, field_size)
    ax.set_ylim(-field_size, field_size)
    ax.grid(True)
    
    return fig, ax
