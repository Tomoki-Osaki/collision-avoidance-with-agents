""" 
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
from matplotlib.animation import FuncAnimation
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

# %% awareness model    
def extend_line(start_point: np.array, 
                through_point: np.array, 
                length: int) -> np.array:
    # 方向ベクトル
    direction = np.array(through_point) - np.array(start_point)
    # 単位ベクトルに変換
    unit_vector = direction / np.linalg.norm(direction)
    # 指定した長さだけ伸ばした終点を計算
    extended_end = np.array(through_point) + unit_vector * length
 
    return extended_end


def is_point_on_segment(cp: np.array, pos_t: np.array, extended_end: np.array) -> bool:
    
    # 線分の範囲内にあるか (bounding box のチェック)
    if ( min(pos_t[0], extended_end[0]) <= cp[0] <= max(pos_t[0], extended_end[0]) 
         and 
         min(pos_t[1], extended_end[1]) <= cp[1] <= max(pos_t[1], extended_end[1]) ):
        
        return True

    return False


def calc_crossing_point(pos1_tminus1: np.array, 
                        pos1: np.array, 
                        pos2_tminus1: np.array, 
                        pos2: np.array) -> np.array or None:
    
    posx1_tminus1, posy1_tminus1 = pos1_tminus1
    posx1, posy1 = pos1
    posx2_tminus1, posy2_tminus1 = pos2_tminus1
    posx2, posy2 = pos2
    
    # 傾きと切片を求める
    m1 = (posy1 - posy1_tminus1) / (posx1 - posx1_tminus1)
    b1 = posy1_tminus1 - m1 * posx1_tminus1
    
    m2 = (posy2 - posy2_tminus1) / (posx2 - posx2_tminus1)
    b2 = posy2_tminus1 - m2 * posx2_tminus1
    
    # 直線が平行な場合、交点は存在しない
    if m1 == m2:
        return None
    
    # 交点の x 座標を求める
    x_intersect = (b2 - b1) / (m1 - m2)
    # 交点の y 座標を求める
    y_intersect = m1 * x_intersect + b1
    
    crossing_point = np.array([x_intersect, y_intersect])
    
    # crossing_pointが、2つのエージェントの現在値を始点として延長した線分に含まれている場合、crossing_pointを返す
    extended_end1 = extend_line(pos1_tminus1, pos1, length=100)
    is_cp_on_seg1 = is_point_on_segment(crossing_point, pos1, extended_end1)
    
    extended_end2 = extend_line(pos2_tminus1, pos2, length=100)
    is_cp_on_seg2 = is_point_on_segment(crossing_point, pos2, extended_end2)

    if is_cp_on_seg1 and is_cp_on_seg2:    
        return crossing_point
    else:
        return None
    

def calc_deltaTTCP(pos1: np.array, vel1: np.array, 
                   pos2: np.array, vel2: np.array) -> float or None:
    """
    エージェント1(pos1, vel1)とエージェント2(pos2, vel2)のdeltaTTCPを計算する
    deltaTTCP > 0　の場合、エージェント1が後に行く
    deltaTTCP < 0　の場合、エージェント1が先に行く
    deltaTTCP = 0　の場合、エージェント1とエージェント2は衝突する
    pos = (x, y)
    vel = (x, y)
    """
    pos1_tminus1 = pos1 - vel1
    pos2_tminus1 = pos2 - vel2
    
    cp = calc_crossing_point(pos1_tminus1, pos1, pos2_tminus1, pos2)
    if cp is None:
        return None
    
    dist_to_cp1 = cp - pos1
    TTCP1 = dist_to_cp1 / vel1    
    
    dist_to_cp2 = cp - pos2
    TTCP2 = dist_to_cp2 / vel2 
    
    deltaTTCP = (TTCP1 - TTCP2)[0]
    
    return deltaTTCP


def calc_angle_two_lines(line1: np.array, # shape[2, 2]
                         line2: np.array, # shape[2, 2]
                         ) -> float:
    """
    2直線がなす角度(radian)を求める関数
    line1, line2: 各直線を表す2点 np.array([[x1,y1], [x2,y2]])
    
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


def calc_angle_two_vectors(common_pos: np.array, 
                           vec1: np.array, 
                           vec2: np.array, 
                           as_radian: bool = False) -> float:
    #角度の中心位置
    x0, y0 = common_pos
    #方向指定1
    x1, y1 = vec1
    #方向指定2
    x2, y2 = vec2
    
    #角度計算開始
    vec1 = [x1 - x0, y1 - y0]
    vec2 = [x2 - x0, y2 - y0]
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    if absvec1 == 0. or absvec2 == 0.:
        return None
    
    inner = np.inner(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.rad2deg(theta_rad)
        
    if as_radian:
        return theta_rad
    return theta_deg


def awareness_model(deltaTTCP: float, Px: float, Py: float, 
                    Vself: float, Vother: float, theta: float, Nic: int) -> float:
    """
    Inputsの値は標準化されている必要あり
    
    Inputs
        deltaTTCP: 自分のTTCPから相手のTTCPを引いた値
        Px: 自分から見た相手の相対位置
        Py: 自分から見た相手の相対位置
        Vself: 自分の歩行速度
        Vother: 相手の歩行速度 
        theta: 自分の向いている方向と相手の位置の角度差 
        Nic: 円内他歩行者数 
    Output
        0(注視しない) - 1 (注視する)
    """
    if deltaTTCP is None:
        return 0
    
    deno = 1 + np.exp(
        -(-1.2 +0.018*deltaTTCP -0.1*Px -1.1*Py \
          -0.25*Vself +0.29*Vother -2.5*theta -0.62*Nic)    
    )
    val = 1 / deno
    
    return val


def animte_agent_movements(data: np.array, # shape(steps, 2, num_agent)
                           sim_obj, 
                           save_as: str
                           ) -> None:
    plt.rcParams['font.family'] = "MS Gothic"
    plt.rcParams['font.size'] = 14
    
    fig, ax = plt.subplots(figsize=(8,8))
    def update(frame):
        ax.cla()
        for i in range(sim_obj.agent):
            if i < sim_obj.num_dynamic_agent:
                if i == 0:
                    color = 'red'
                    ax.scatter(x=frame[0][i], y=frame[1][i], s=40,
                                marker="o", c=color, label='動的回避')
                else:
                    ax.scatter(x=frame[0][i], y=frame[1][i], s=40,
                                marker="o", c=color)
            else:
                color = 'blue'
                if i == sim_obj.num_dynamic_agent:
                    ax.scatter(x=frame[0][i], y=frame[1][i], s=40,
                                marker="o", c=color, label='単純回避')
                else:
                    ax.scatter(x=frame[0][i], y=frame[1][i], s=40,
                                marker="o", c=color)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid()
        ax.legend(loc='upper left', framealpha=1)
    
    anim = FuncAnimation(fig, update, frames=data, interval=100)
    anim.save(save_as)
    
    
def standardize(array: np.array, mu: float = None, sigma: float = None) -> np.array:
    """
    標準化したarraｙを返す
    except_nanがTrueの場合、muやsigmaの計算時にarray中のnan値を無視する
    """
    if mu is None and sigma is None:
        mu = np.nanmean(array)
        sigma = np.nanstd(array)
    
    standardized = (array - mu) / sigma
    
    return standardized


def show(array):
    """
    arrayの要素を、インデックス付きで表示する
    """
    for i, j in enumerate(array):
        print(i, j)

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

# %% old version of deltaTTCP
# def calc_cross_point(velx1: float, vely1: float, posx1: float, posy1: float, 
#                      velx2: float, vely2: float, posx2: float, posy2: float
#                      ) -> np.array or None: # [float(x), float(y)]
#     """
#     ex. calc_cross_point(1, 2, 3, 4, 
#                          5, 6, 7, 8) -> array([2.0, 2.0])
#     """
#     nume_x = velx2*vely1*posx1 - velx1*vely2*posx2 + velx1*velx2*(posy2 - posy1)
#     deno_x = velx2*vely1 - velx1*vely2
#     if deno_x == 0:
#         return None
#     else:
#         CPx = nume_x / deno_x    

#         CPy = (vely1 / velx1) * (CPx - posx1) + posy1
#         CP = np.array([CPx, CPy])
        
#         return  CP


# def calc_deltaTTCP(velx1: float, vely1: float, posx1: float, posy1: float, 
#                    velx2: float, vely2: float, posx2: float, posy2: float
#                    ) -> float or None:
#     """
#     ex. calc_deltaTTCP(-1.4, 0.4, 1.3, -2.1, 
#                        -0.9, -0.3, 2.0, 0.8) -> -2.5
#     """
#     CP = calc_cross_point(velx1, vely1, posx1, posy1, velx2, vely2, posx2, posy2)
#     try:
#         CPx, CPy = CP[0], CP[1]
#     except TypeError: # 'NoneType' object is not subscriptable
#         return None
    
#     TTCP0 = None
#     TTCP1 = None
    
#     if (
#             ( (posx1 < CPx and velx1 > 0) or (posx1 > CPx and velx1 < 0) ) 
#         and
#             ( (posy1 < CPy and vely1 > 0) or (posy1 > CPy and vely1 < 0) )
#         ):
        
#         nume0 = np.sqrt( (CPx - posx1)**2 + (CPy - posy1)**2 )
#         deno0 = np.sqrt( (velx1**2 + vely1**2) )
        
#         TTCP0 = nume0 / deno0
#     else:
#         return None

#     if ( 
#             ( (posx2 < CPx and velx2 > 0) or (posx2 > CPx and velx2 < 0) ) 
#         and
#             ( (posy2 < CPy and vely2 > 0) or (posy2 > CPy and vely2 < 0) )
#         ):
        
#         nume1 = np.sqrt( (CPx - posx2)**2 + (CPy - posy2)**2 )
#         deno1 = np.sqrt( (velx2**2 + vely2**2) )
        
#         TTCP1 = nume1 / deno1
#     else:
#         return None
    
#     # TTCP0とTTCP1がどちらもNoneでない場合のみdeltaTTCPを返す
#     deltaTTCP = TTCP0 - TTCP1 #　正は道を譲り、負は自分が先に行く
    
#     return deltaTTCP    
