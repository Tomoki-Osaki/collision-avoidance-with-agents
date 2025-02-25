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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from dataclasses import dataclass

# %% functions to calculate the vectors
def calc_rad(pos2: np.ndarray, # [float, float]
             pos1: np.ndarray # [float, float]
             ) -> float: 
    """
    pos1からpos2のベクトルの角度を返す
    ex. calc_rad(pos2=np.array([1.5, 2.5]), pos1=np.array([3.0, 1.0])) -> 2.4
    """
    val = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
    
    return val


def rotate_vec(vec: np.ndarray, # [float, float] 
               rad: float
               ) -> np.ndarray: # [float, float]
    """
    ベクトルをradだけ回転させる (回転行列)
    ex. rotate_vec(vec=np.array([3.0, 5.0]), rad=1.2) -> array([-3.6, 4.6])
    """
    rotation = np.array([[np.cos(rad), -np.sin(rad)], 
                         [np.sin(rad), np.cos(rad)]])
    val = np.dot(rotation, vec.T).T
    
    return val

# %% awareness model    
def extend_line(start_point: np.ndarray, 
                through_point: np.ndarray, 
                length: int) -> np.ndarray:
    # 方向ベクトル
    direction = np.array(through_point) - np.array(start_point)
    # 単位ベクトルに変換
    unit_vector = direction / np.linalg.norm(direction)
    # 指定した長さだけ伸ばした終点を計算
    extended_end = np.array(through_point) + unit_vector * length
 
    return extended_end


def is_point_on_segment(cp: np.ndarray, pos_t: np.ndarray, extended_end: np.ndarray) -> bool:
    
    # 線分の範囲内にあるか (bounding box のチェック)
    if ( min(pos_t[0], extended_end[0]) <= cp[0] <= max(pos_t[0], extended_end[0]) 
         and 
         min(pos_t[1], extended_end[1]) <= cp[1] <= max(pos_t[1], extended_end[1]) ):
        
        return True

    return False

# def calc_crossing_point(pos1_tminus1: np.ndarray, 
#                         pos1: np.ndarray, 
#                         pos2_tminus1: np.ndarray, 
#                         pos2: np.ndarray) -> np.ndarray or None:
    
#     posx1_tminus1, posy1_tminus1 = pos1_tminus1
#     posx1, posy1 = pos1
#     posx2_tminus1, posy2_tminus1 = pos2_tminus1
#     posx2, posy2 = pos2
    
#     # 傾きと切片を求める
#     m1 = (posy1 - posy1_tminus1) / (posx1 - posx1_tminus1)
#     b1 = posy1_tminus1 - m1 * posx1_tminus1
    
#     m2 = (posy2 - posy2_tminus1) / (posx2 - posx2_tminus1)
#     b2 = posy2_tminus1 - m2 * posx2_tminus1
    
#     # 直線が平行な場合、交点は存在しない
#     if m1 == m2:
#         return None
    
#     # 交点の x 座標を求める
#     x_intersect = (b2 - b1) / (m1 - m2)
#     # 交点の y 座標を求める
#     y_intersect = m1 * x_intersect + b1
    
#     crossing_point = np.array([x_intersect, y_intersect])
    
#     # crossing_pointが、2つのエージェントの現在値を始点として延長した線分に含まれている場合、crossing_pointを返す
#     extended_end1 = extend_line(pos1_tminus1, pos1, length=5)
#     is_cp_on_seg1 = is_point_on_segment(crossing_point, pos1, extended_end1)
    
#     extended_end2 = extend_line(pos2_tminus1, pos2, length=5)
#     is_cp_on_seg2 = is_point_on_segment(crossing_point, pos2, extended_end2)

#     if is_cp_on_seg1 and is_cp_on_seg2:    
#         return crossing_point
#     else:
#         return None

def calc_crossing_point(pos1_tminus1: np.ndarray, 
                        pos1: np.ndarray, 
                        pos2_tminus1: np.ndarray, 
                        pos2: np.ndarray) -> np.ndarray or None:
    
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
    
    agent_size = 0.1
    line1_b = b1 + agent_size
    line2_b = b1 - agent_size
    line3_b = b2 + agent_size
    line4_b = b2 - agent_size
    line1_m = line2_m = m1
    line3_m = line4_m = m2
    
    x13 = -( (line1_b - line3_b) / (line1_m - line3_m) )
    y13 = line1_m * x13 + line1_b

    x14 = -( (line1_b - line4_b) / (line1_m - line4_m) )
    y14 = line1_m * x14 + line1_b

    x23 = -( (line2_b - line3_b) / (line2_m - line3_m) )
    y23 = line2_m * x23 + line2_b

    x24 = -( (line2_b - line4_b) / (line2_m - line4_m) )
    y24 = line2_m * x24 + line2_b

    cps = [np.array([x13, y13]), np.array([x14, y14]), 
           np.array([x23, y23]), np.array([x24, y24])]
    
    agent1_pos = np.array([posx1, posy1])
    agent2_pos = np.array([posx2, posy2])

    min_dist = None
    crossing_point = None
    for value in cps:
        agent1_to_cp = np.linalg.norm(agent1_pos - value)
        agent2_to_cp = np.linalg.norm(agent2_pos - value)
        sum_dist = agent1_to_cp + agent2_to_cp

        if not min_dist or sum_dist < min_dist:
            min_dist = sum_dist
            crossing_point = value
    
    # crossing_pointが、2つのエージェントの現在値を始点として延長した線分に含まれている場合、crossing_pointを返す
    extended_end1 = extend_line(pos1_tminus1, pos1, length=5)
    is_cp_on_seg1 = is_point_on_segment(crossing_point, pos1, extended_end1)
    
    extended_end2 = extend_line(pos2_tminus1, pos2, length=5)
    is_cp_on_seg2 = is_point_on_segment(crossing_point, pos2, extended_end2)

    if is_cp_on_seg1 and is_cp_on_seg2:    
        return crossing_point
    else:
        return None    


def calc_deltaTTCP(pos1: np.ndarray, vel1: np.ndarray, 
                   pos2: np.ndarray, vel2: np.ndarray) -> float or None:
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
    
    deltaTTCP = (TTCP1 - TTCP2)[0] # x座標とy座標それぞれで同一の値が出るため、インデックスを使って１つの値を返す
    
    return deltaTTCP


def calc_angle_two_lines(line1: np.ndarray, # shape[2, 2]
                         line2: np.ndarray, # shape[2, 2]
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


def calc_angle_two_vectors(common_pos: np.ndarray, 
                           vec1: np.ndarray, 
                           vec2: np.ndarray, 
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

# %%
def show(obj):
    """
    objの要素を、インデックス付きで表示する
    """
    for i, j in enumerate(obj):
        print(i, j)
        
        
def remove_outliers_and_nan(data, sd=None):
    old_data = data
    not_nan, = np.where(~np.isnan(old_data))
    not_nan_data = old_data[not_nan]
    if sd is None:
        return not_nan_data
    
    new_data = not_nan_data[
        abs(not_nan_data - np.nanmean(not_nan_data)) < sd * np.nanstd(not_nan_data)
    ]
    
    return new_data


@dataclass
class PedData:
    """
    歩行者の交差実験から得られた各パラメータの平均値・標準偏差
    """
    deltaTTCP_mean = 0
    deltaTTCP_std = 16.9540
    Px_mean = 287.4392 
    Px_std = 374.4476
    Py_mean = 0.6823
    Py_std = 3.2914 
    Vself_mean = 0.9839
    Vself_std = 0.8156
    Vother_mean = 0.9839
    Vother_std = 0.8157
    theta_mean = 82.4291
    theta_std = 50.5367
    Nic_mean = 0.6625
    Nic_std = 0.8297
    
    def show_params(self):
        print('\nParameters of pedestrian exp data')
        print('-------------------------')
        print('deltaTTCP_mean:', np.round(self.deltaTTCP_mean, 3))
        print('deltaTTCP_std:', np.round(self.deltaTTCP_std, 3))
        print('mean-std ratio:', np.round(self.deltaTTCP_mean/self.deltaTTCP_std, 3))
        
        print('\nPx_mean:', np.round(self.Px_mean, 3)) 
        print('Px_std:', np.round(self.Px_std, 3))
        print('mean-std ratio:', np.round(self.Px_mean/self.Px_std, 3))        
        
        print('\nPy_mean:', np.round(self.Py_mean, 3))
        print('Py_std:', np.round(self.Py_std, 3))
        print('mean-std ratio:', np.round(self.Py_mean/self.Py_std, 3))
        
        print('\nVself_mean:', np.round(self.Vself_mean, 3))
        print('Vself_std:', np.round(self.Vself_std, 3))
        print('mean-std ratio:', np.round(self.Vself_mean/self.Vself_std, 3))
        
        print('\nVother_mean:', np.round(self.Vother_mean, 3))
        print('Vother_std:', np.round(self.Vother_std, 3))
        print('mean-std ratio:', np.round(self.Vother_mean/self.Vother_std, 3))
        
        print('\ntheta_mean:', np.round(self.theta_mean, 3))
        print('theta_std:', np.round(self.theta_std, 3))
        print('mean-std ratio:', np.round(self.theta_mean/self.theta_std, 3))
        
        print('\nNic_mean:', np.round(self.Nic_mean, 3))
        print('Nic_std:', np.round(self.Nic_std, 3))
        print('mean-std ratio:', np.round(self.Nic_mean/self.Nic_std, 3))
        print('-------------------------\n')
        
        
def animate_agent_movements(sim, save_as: str = 'simulation.mp4', viz_angle: bool = False) -> None:
    plt.rcParams['font.family'] = "MS Gothic"
    plt.rcParams['font.size'] = 14
    
    # data_arrの形を、フレームとしてアニメーションを回せるように作り変える        
    data_arr = pd.DataFrame(columns=['Agent', 'Px', 'Py'])
    for step in range(sim.num_steps+1):
        for agent in range(sim.num_agents):
            px, py = sim.all_agents[agent]['all_pos'][step]
            tmp = pd.DataFrame({'Agent': agent, 'Px': px, 'Py': py}, index=[step])
            data_arr = pd.concat([data_arr, tmp])
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    def update(frame):
        global tminus1
        if frame[0] == 0:
            tminus1 = data_arr.copy()
            tminus1.iloc[:] = np.nan
            
        ax.cla()
        for row, row_tminus1 in zip(frame[1].iterrows(), tminus1.iterrows()):
            # プロットする際の位置をピクセルに合わせる(*50)
            x, y = (row[1]['Px']*50)+250, (row[1]['Py']*50)+250

            color = 'red' if row[1]['Agent'] < sim.num_dynamic_agent else 'blue'                

            ax.text(x, y, row[1]['Agent'], size=10) # エージェントを番号付きで表示
            ax.add_artist(patches.Circle((x, y), radius=5, color=color))
            
            # エージェントの視野を可視化する
            if viz_angle:
                assert sim.awareness == False, """Agents with Awareness model are not for 
                visualizing viewing angles in this way."""
                if frame[0] != 0:
                    angle = sim.viewing_angle / 2
                    x_tminus1 = (row_tminus1[1]['Px']*50)+250
                    y_tminus1 = (row_tminus1[1]['Py']*50)+250
                    
                    # 進行方向の角度（ラジアン → 度）
                    theta = np.arctan2(y - y_tminus1, x - x_tminus1) * 180 / np.pi  
                    
                    # 半円の中心（t時点の位置）
                    radius = 50  # 視野の半径
    
                    # 半円（Wedge）の追加
                    ax.add_artist(patches.Wedge((x, y), radius, theta-angle, theta+angle, 
                                                color=color, alpha=0.1))
                
        tminus1 = frame[1]
            
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_xticks(range(0, 501, 50))
        ax.set_yticks(range(0, 501, 50))
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')
        ax.set_title(f'Step: {row[0]}')
        ax.grid()
                    
    anim = FuncAnimation(fig, update, frames=data_arr.groupby(data_arr.index), interval=sim.interval)
    print('\ndrawing the animation...')
    anim.save(save_as)
    plt.close()


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
