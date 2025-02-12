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

# %% import libraries
import numpy as np
import matplotlib.pyplot as plt

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

# now adjusting the weights of each explanatory variables
def awareness_model(sim_obj, 
                    num: int, 
                    other_num: int, 
                    current_step: int, 
                    dataclass_aware, 
                    awareness_weight,
                    debug=False) -> float:
    """
    awareness modelを用いて、エージェント番号numの注視相手を選定する(0-1)
    ex. awareness_model(sim, num=10, other_num=15, current_step=20, dataclass_aware) -> 0.85
    
    説明変数
            deltaTTCP: 自分のTTCPから相手のTTCPを引いた値
            Px: 自分から見た相手の相対位置
            Py: 自分から見た相手の相対位置
            Vself: 自分の歩行速度
            Vother: 相手の歩行速度 
            theta: 自分の向いている方向と相手の位置の角度差 
            Nic: 円内他歩行者数 
            
    被説明変数
            0(注視しない) - 1 (注視する)
    """ 
    agent = sim_obj.all_agents[num]
    
    all_deltaTTCP = dataclass_aware.deltaTTCP
    deltaTTCP_mean, deltaTTCP_std = np.nanmean(all_deltaTTCP), np.nanstd(all_deltaTTCP)

    all_Px = dataclass_aware.Px
    Px_mean, Px_std = np.nanmean(all_Px), np.nanstd(all_Px)

    all_Py = dataclass_aware.Py
    Py_mean, Py_std = np.nanmean(all_Py), np.nanstd(all_Py)

    all_Vself = dataclass_aware.Vself
    Vself_mean, Vself_std = np.nanmean(all_Vself), np.nanstd(all_Vself)

    all_Vother = dataclass_aware.Vother
    Vother_mean, Vother_std = np.nanmean(all_Vother), np.nanmean(all_Vother)

    all_theta = dataclass_aware.theta
    theta_mean, theta_std = np.nanmean(all_theta), np.nanstd(all_theta)

    all_nic = dataclass_aware.Nic
    nic_mean, nic_std = np.nanmean(all_nic), np.nanstd(all_nic)

    deltaTTCP = (agent['deltaTTCP'][current_step][other_num] - deltaTTCP_mean) / deltaTTCP_std
    if np.isnan(deltaTTCP):
        return 0
    Px = (agent['relPx'][current_step][other_num] - Px_mean) / Px_std
    Py = (agent['relPy'][current_step][other_num] - Py_mean) / Py_std
    Vself = (agent['all_vel'][current_step] - Vself_mean) / Vself_std
    Vother = (agent['all_other_vel'][current_step][other_num] - Vother_mean) / Vother_std        
    theta = (agent['theta'][current_step][other_num] - theta_mean) / theta_std
    Nic = (agent['Nic'][current_step][other_num] - nic_mean) / nic_std
    
    # # awareness model original
    # deno = 1 + np.exp(
    #     -(-1.2 +0.018*deltaTTCP -0.1*Px -1.1*Py -0.25*Vself +0.29*Vother -2.5*theta -0.62*Nic)    
    # )
    # val = 1 / deno
    
    # awareness model adjusted parameters
    w = awareness_weight
    
    deno = 1 + np.exp(
        -(w.bias + w.deltaTTCP*deltaTTCP + w.Px*Px + w.Py*Py + w.Vself*Vself + \
          w.Vother*Vother + w.theta*theta + w.Nic*Nic)    
    )
    val = 1 / deno
    
    if debug:
        print(f'\ndeltaTTCP {deltaTTCP:.3f}; {deltaTTCP*w.deltaTTCP:.3f}')
        print(f'Px {Px:.3f}; {Px*w.Px:.3f}')
        print(f'Py {Py:.3f}; {Py*w.Py:.3f}')
        print(f'Vself {Vself:.3f}; {Vself*w.Vself:.3f}')
        print(f'Vother {Vother:.3f}; {Vother*w.Vother:.3f}')
        print(f'theta {theta:.3f}; {theta*w.theta:.3f}')
        print(f'Nic {Nic:.3f}; {Nic*w.Nic:.3f}')
        print(f'Awareness {val:.3f}\n')
    else:
        return val
    
        
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
