""" 2025/01/06 """
# %% structure of functions, class, and methods
"""
define_fig_ax()
calc_rad(pos2, pos1)
rotate_vec(vec, rad)
calc_distance
calc_cross_point
calc_deltaTTCP
calc_angle_two_lines
awareness_model

class: simulation
  __init__(self)
  1. find_goal(self, agent)
  2. distance(self)
  3. approach_detect(self, dist)
  4. simple_avoidance(self, num)
  5. dynamic_avoidance(self, num, goal)
  6. calc_Nic(self, num)
  7. find_agents_to_focus(self, num)
  8. simulate(self, step)
  9. calc_completion_time(self, num, now_step)
  10. calc_last_completion_time(self, num)
  11. show_image(self)
  12. plot_positions(self)
"""

# %% import libraries
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size 
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from typing import Literal

# %% global variables
SIZE = 5 # グラフの目盛りの最大値・最小値

# 目盛りは最大値5、最小値-5で10目盛り
# グラフ領域の幅と高さは500pxなので、1pxあたり0.02目盛りとなる

INTERVAL = 100 # 100msごとにグラフを更新してアニメーションを作成

# 妨害指標の4係数は標準化したやつを使う
abcd = {'a1': -5.145, # -0.034298
        'b1': 3.348, # 3.348394
        'c1': 4.286, # 4.252840
        'd1': -13.689} # -0.003423

# %% DO NOT EDIT constants
# a1: -5.145 (-0.034298)
# b1: 3.348 (3.348394)
# c1: 4.286 (4.252840)
# d1: -13.689 (-0.003423)

# %% define figure parameters
def define_fig_ax(width: int = 500, 
                  height: int = 500) -> plt.subplots:
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
    
    ax.set_xlim(-SIZE, SIZE)
    ax.set_ylim(-SIZE, SIZE)
    ax.grid(True)
    
    return fig, ax

# %% functions to calculate the vectors
def calc_rad(pos2: np.array, # [float, float]
             pos1: np.array # [float, float]
             ) -> float: 
    """
    ex. calc_rad(pos2=np.array([1.5, 2.5]), pos1=np.array([3.0, 1.0])) -> 2.4
    """
    # pos1からpos2のベクトルの角度を返す
    val = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
    
    return val


def rotate_vec(vec: np.array, # [float, float] 
               rad: float
               ) -> np.array: # [float, float]
    """
    ex. rotate_vec(vec=np.array([3.0, 5.0]), rad=1.2) -> array([-3.6, 4.6])
    """
    # ベクトルをradだけ回転させる (回転行列)
    rotation = np.array([[np.cos(rad), -np.sin(rad)], 
                         [np.sin(rad), np.cos(rad)]])
    val = np.dot(rotation, vec.T).T
    
    return val


def calc_distance(posX1, posY1, posX2, posY2):
    """
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
    TTCP0 = TTCP1 = None
    
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
    
    deltaTTCP = TTCP0 - TTCP1 #　正は道を譲り、負は自分が先に行く
    
    return deltaTTCP    
    

def calc_angle_two_lines(line1, line2):
    """
    2直線の角度(radian)を求める関数
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

# %% シミュレーションに関わるクラス
class simulation():
    def __init__(self, 
                 agent_size: float = 0.1, 
                 agent: int = 25, 
                 view: int = 1, 
                 viewing_angle: int = 360, 
                 goal_vec: float = 0.06, 
                 simple_avoid_vec: float = 0.06, 
                 dynamic_avoid_vec: float = 0.06, 
                 step: int = 500, 
                 avoidance: Literal['simple', 'dynamic'] = 'simple'):
        
        self.agent_size = agent_size # エージェントの半径(目盛り) = 5px
        self.agent = agent # エージェント数
        self.view = view # 視野の半径(目盛り) = 50px:エージェント5体分
        self.viewing_angle = viewing_angle # 視野の角度
        self.goal_vec = goal_vec # ゴールベクトルの大きさ(目盛り)
        self.simple_avoid_vec = simple_avoid_vec # 単純回避での回避ベクトルの大きさ(目盛り)
        self.dynamic_avoid_vec = dynamic_avoid_vec # 動的回避での回避ベクトルの最大値(目盛り)
        self.step = step # 1回の試行(TRIAL)で動かすステップの回数
        self.avoidance = avoidance # 'simple' or 'dynamic'

        self.all_agent = [] # 全エージェントの座標を記録
        self.all_agent2 = [] # ゴールの計算用
        self.first_agent = [] # 初期位置記録用
        self.agent_goal = []
        self.first_pos =[]
        
        for n in range(self.agent):
            # グラフ領域の中からランダムに座標を決定
            pos = np.random.uniform(-SIZE, SIZE, 2)
            vel = np.random.uniform(-SIZE, SIZE, 2)
            
            # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
            # self.all_agentの1つの要素に1体のエージェントの位置と速度が記録
            self.all_agent.append(
                {'#': n, # to caclculate the awareness but may not be useful
                 'p': pos, 
                 'v': rotate_vec(np.array([self.goal_vec, 0]), 
                                 calc_rad(vel, np.array([0, 0])))
                 }
            )
            
        # 初期位置と初期速度をコピー
        self.all_agent2 = deepcopy(self.all_agent)
        self.first_agent = deepcopy(self.all_agent)
        
        # エージェントの初期位置を保存
        for i in range(self.agent):
            self.first_pos.append(self.first_agent[i]['p'])
        
        # エージェントにゴールを8ずつ設定
        for i in range(self.agent):
            goals = []
            for j in range(8):
                goals.append(self.find_goal(self.all_agent2[i]))
                
            self.agent_goal.append(goals)
             
            
        # エージェント間の距離を記録するリスト
        self.dist = np.zeros([self.agent, self.agent])
        
        # ゴールした回数を記録するリスト
        self.goal_count = []
        for i in range(self.agent):
            self.goal_count.append(0)

        # はみ出た時用のゴール
        self.goal_temp = np.zeros([self.agent, 2])
        
        # 完了時間を測るための変数
        self.start_step = np.zeros([self.agent])
        self.goal_step = np.zeros([self.agent])
        self.start_pos = self.first_pos
        self.goal_pos = np.zeros([self.agent, 2])
         
        # 完了時間を記録するリスト
        self.completion_time = []


    # 1. ゴールの計算
    # エージェントが初期速度のまま進んだ場合に通過する、グラフ領域の境界線の座標
    def find_goal(self, agent: dict[str, np.array] 
                  ) -> np.array: # [float, float]
        """ 
        初期位置の値を使うためself.all_agentではなくself.all_agent2を使う
        ex. self.find_goal(agent=self.all_agent2[10]) -> array([-3.0, -5.0])
        """
        while True:
            
            # x座標がグラフ領域を超える
            if ((agent['p'] + agent['v'])[0] < -SIZE):
                # 超えた時の座標をゴールとする
                goal = agent['p'] + agent['v']
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > SIZE - 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] =  goal[1] - 0.1
                    print("調整入りました")
                    
                # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -SIZE + 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] = goal[1] + 0.1
                    print("調整入りました")
                    
                goal[0] = -SIZE
                # 端に到達したエージェントを、反対側の端に移動させる
                agent['p'][0] = SIZE + ((agent['p'] + agent['v'])[0] + SIZE)
                break
                        
            elif ((agent['p'] + agent['v'])[0] > SIZE):
                goal = agent['p'] + agent['v']
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > SIZE - 0.1):
                    goal[1] = goal[1] - 0.1
                    print("調整入りました")

               # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -SIZE + 0.1):
                    goal[1] = goal[1] + 0.1
                    print("調整入りました")
                    
                goal[0] = SIZE
                agent['p'][0] = -SIZE + ((agent['p'] + agent['v'])[0] - SIZE)
                break
                
                
            # y座標がグラフ領域を超える
            elif ((agent['p'] + agent['v'])[1] < -SIZE):
                # 超えた時の座標をゴールとする
                goal = agent['p'] + agent['v']
                goal[1] = -SIZE
                
                agent['p'][1] = SIZE + ((agent['p'] + agent['v'])[1] + SIZE)
                break
                                        
            elif ((agent['p'] + agent['v'])[1] > SIZE):
                goal = agent['p'] + agent['v']
                goal[1] = SIZE
                
                agent['p'][1] = -SIZE + ((agent['p'] + agent['v'])[1] - SIZE)
                break
                
            # エージェントを初期速度のまま動かす
            agent['p'] = agent['p'] + agent['v']

        return goal

        
    # 2. 距離の計算
    def distance_all_agents(self):
        for i in range(self.agent):
            for j in range(self.agent):
                d = self.all_agent[i]['p'] - self.all_agent[j]['p']
                # エージェント間の距離を算出、エージェントのサイズも考慮
                self.dist[i][j] = np.linalg.norm(d) - 2 * self.agent_size
                
                
    # 3. 指定した距離より接近したエージェントの数を返す
    def approach_detect(self, dist: float) -> np.array: 
        """ 
        ex. self.approach_detect(dist=0.5) -> array([[0, 3],[1, 2],...[24, 1]])
        """
        self.distance_all_agents()
        approach_agent = []
        
        # それぞれのエージェントについて、distより接近したエージェントの数を記録
        for t in range(self.agent):
            visible_agents = [i for i, x in enumerate(self.dist[t]) 
                              if x != -(0.2) and x < dist]
            approach_agent.append([t, len(visible_agents)])
        approach_agent = np.array(approach_agent) 
            
        return approach_agent
    
    
    # 4. 単純な回避ベクトルの生成
    def simple_avoidance(self, num: int # エージェントの番号
                         ) -> np.array: # [float, float]
        """
        ex. self.simple_avoidance(num=15) -> array([-0.05, 0.02])
        """
        self.distance_all_agents()
        # near_agentsは360度の視野に入ったエージェント
        # visible_agentsは視野を狭めた場合に視野に入ったエージェント
        near_agents = [i for i, x in enumerate(self.dist[num]) 
                       if x != -(0.2) and x < self.view]
        visible_agents = []
        
        # 回避ベクトル
        avoid_vec = np.zeros(2)
        
        if not near_agents:
            return avoid_vec
        
        # near_agentsについて、awareness modelの値を基にしたエージェントを計算対象にする
        # focus_agents = self.calc_Nic(num)[near_agents]
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            calc_rad(self.agent_goal[num][self.goal_count[num]], 
                     self.all_agent[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                calc_rad(self.all_agent[i]['p'], 
                         self.all_agent[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)
            
            if angle_difference > 180:
                angle_difference = 360 - angle_difference
                
            # 視野に入っているエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
                
        if not visible_agents: 
            return avoid_vec
            
        for i in visible_agents:
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agent[num]['p'] - self.all_agent[i]['p']
            d = d / (self.dist[num][i] + 2 * self.agent_size) # 大きさ1のベクトルにする
            d = d * self.simple_avoid_vec # 大きさを固定値にする
            
            avoid_vec += d # 回避ベクトルを合成する
            
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    # 5. 動的回避ベクトルの生成
    def dynamic_avoidance(self, num: int, goal: np.array) -> np.array:
        """
        ex. self.dynamic_avoidance(num=15, goal=(-5.0, 3.5)) -> array([-0.05, 0.2])
        """
        self.distance_all_agents()
        near_agents = [i for i, x in enumerate(self.dist[num]) 
                       if x != -(0.2) and x < self.view]
        visible_agents = []
        avoid_vec = np.zeros(2)
        
        if not near_agents:
            return avoid_vec
        
        # near_agentsについて、awareness modelの値を基にしたエージェントを計算対象にする
        # focus_agents = self.calc_Nic(num)[near_agents]
        
        # ゴールベクトルの角度を算出する
        goal_angle = np.degrees(
            calc_rad(self.agent_goal[num][self.goal_count[num]], 
                     self.all_agent[num]['p'])
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            agent_angle = np.degrees(
                calc_rad(self.all_agent[i]['p'], 
                         self.all_agent[num]['p'])
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)

            if angle_difference > 180:
                angle_difference = 360 - angle_difference
            
            # 視界に入ったエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents.append(i)
                
        if not visible_agents:
            return avoid_vec
            
        
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.agent_pos = self.all_agent[num]['p']
            self.agent_vel = self.all_agent[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agent[i]['p']
            self.visible_agent_vel = self.all_agent[i]['v']

            
            dist_former = self.dist[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.agent_pos = self.agent_pos + self.agent_vel
            self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
            d = self.agent_pos - self.visible_agent_pos
            dist_latter = np.linalg.norm(d) - 2 * self.agent_size
            
            
            # 視界に入った時が最も近い場合
            if dist_former < dist_latter:
                tcpa = 0
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                    
            # 2者間距離の最小値が出るまでエージェントを動かす
            else:
                while dist_former > dist_latter:
                    dist_former = dist_latter
                    t += INTERVAL
                    self.agent_pos = self.agent_pos + self.agent_vel
                    self.visible_agent_pos = self.visible_agent_pos + self.visible_agent_vel
                    d = self.agent_pos - self.visible_agent_pos
                    dist_latter = np.linalg.norm(d) - 2 * self.agent_size
                    
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                tcpa = t

            a1, b1, c1, d1 = abcd['a1'], abcd['b1'], abcd['c1'], abcd['d1']
            # ブレーキ指標の算出
            braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                            (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
            
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agent[num]['p'] - self.all_agent[i]['p']
            d = d / (self.dist[num][i] + 2 * self.agent_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            avoid_vec += d # ベクトルの合成
    
        # ベクトルの平均を出す
        return avoid_vec / len(visible_agents)
    
    
    # 6. calculate the Nic (Number in the circle)
    def calc_Nic(self, num: int) -> np.array:
        """ 
        エージェントAとエージェントBの中点cpを計算し、cpから他の全てのエージェントXとの距離cpxを計算する
        その中で、cpとエージェントAの距離dist_cp_meより小さいcpxの数を計算する
        ex. self.calc_Nic(num=10) -> array([[0,1], [1,2],...,[24,12]]) (相手,Nic)
        """ 
        all_Nics = []
        posA = self.all_agent[num]['p']
        for i in range(self.agent):
            posB = self.all_agent[i]['p']
            cp = ( (posA[0]+posB[0])/2, (posA[1]+posB[1])/2 )
            radius = calc_distance(*cp, *posA)
            
            Nic_agents = []
            for j in range(self.agent):
                posX = self.all_agent[j]['p']
                dist_cp_posX = calc_distance(*posX, *cp)
                if dist_cp_posX <= radius:
                    Nic_agents.append(i)
            all_Nics.append([i, len(Nic_agents)-2]) # posAとposBの分を引く
        all_Nics = np.array(all_Nics)
        
        return all_Nics
    
    
    # 7. find the agents to focus on using the awareness model
    def find_agents_to_focus(self, num: int) -> np.array:
        """
        ex. self.find_agents_to_focus(
            num=10) -> array([[0, 0.2], [1, 0.3],...,[24, 0.8]]) (相手,awareness)
        """ 
        agents_to_focus = []
        my_posx, my_posy = self.all_agent[num]['p']
        vselfxy = self.all_agent[num]['v']
        Vself = np.linalg.norm(vselfxy)
        
        a = self.all_agent[num]['p'] + self.all_agent[num]['v']
        b = self.all_agent[num]['p']
        line1 = np.array([b, a])
        
        for i in range(self.agent):
            other_posx, other_posy = self.all_agent[i]['p']
            Px = other_posx - my_posx
            Py = other_posy - my_posy
            votherxy = self.all_agent[i]['v']
            Vother = np.linalg.norm(votherxy)
            
            c = self.all_agent[i]['p']
            line2 = np.array([b, c])
            theta = calc_angle_two_lines(line1, line2)
            
            # deltaTTCPが計算できるときはawmを返し、そうでなければ0を返す
            try:
                deltaTTCP = calc_deltaTTCP(*vselfxy, my_posx, my_posy, 
                                           *votherxy, other_posx, other_posy)
                Nic = self.calc_Nic(num)[i][1]
                awm = awareness_model(deltaTTCP, Px, Py, Vself, Vother, theta, Nic)
                agents_to_focus.append([i, awm])
            except TypeError:
                agents_to_focus.append([i, 0])
        agents_to_focus = np.array(agents_to_focus)
        
        return agents_to_focus
    
    
    # 8. シミュレーション
    def simulate(self, step: int) -> None:
        
        # 単純回避
        if self.avoidance == 'simple':
            for i in range(self.agent):
                # はみ出た時用のゴールが設定されていない
                # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
                if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                    self.all_agent[i]['v'] = rotate_vec(
                        np.array([self.goal_vec, 0]), 
                        calc_rad(self.agent_goal[i][self.goal_count[i]],
                                 self.all_agent[i]['p'])
                    ) + self.simple_avoidance(i)        

                # はみ出た時用のゴールが設定されている
                # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
                else:
                    self.all_agent[i]['v'] = rotate_vec(
                        np.array([self.goal_vec, 0]), 
                        calc_rad(self.goal_temp[i], self.all_agent[i]['p'])
                    ) + self.simple_avoidance(i)            
        
        # 動的回避
        if self.avoidance == 'dynamic':
            for i in range(self.agent):
                if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                    self.all_agent[i]['v'] = rotate_vec(
                        np.array([self.goal_vec, 0]), 
                        calc_rad(self.agent_goal[i][self.goal_count[i]],
                                 self.all_agent[i]['p'])
                        ) + self.dynamic_avoidance(
                            i, self.agent_goal[i][self.goal_count[i]]
                            )
                else:
                    self.all_agent[i]['v'] = rotate_vec(
                        np.array([self.goal_vec, 0]), 
                        calc_rad(self.goal_temp[i], 
                                 self.all_agent[i]['p'])
                        ) + self.dynamic_avoidance(
                            i, self.goal_temp[i]
                            )            
        
        
        for i in range(self.agent):
            # x座標が左端をこえる
            if ((self.all_agent[i]['p'] + self.all_agent[i]['v'])[0] < -SIZE):
                # ゴールに到着
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # 通常のゴールに到着
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, step)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        # ゴールした回数を更新
                        self.goal_count[i] = self.goal_count[i] + 1
                        
                    # はみ出た時用のゴールに到着
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                # ゴールに到着せず、境界を超える  
                else:
                    # はみ出た時用のゴールが設定されていない
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # はみ出た時用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0] + 2*SIZE
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1]
                        
                    # はみ出た時用のゴールが設定されている
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                
                # エージェントを反対の端へ移動
                self.all_agent[i]['p'][0] = SIZE + (
                    (self.all_agent[i]['p']+self.all_agent[i]['v'])[0] + SIZE
                )
        
            
            # x座標が右端をこえる
            elif ((self.all_agent[i]['p']+self.all_agent[i]['v'])[0] > SIZE):
                
                # ゴール判定
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, step)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        self.goal_count[i] = self.goal_count[i] + 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0] + (-2 * SIZE)
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1]
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agent[i]['p'][0] = -SIZE + \
                    ((self.all_agent[i]['p']+self.all_agent[i]['v'])[0] - SIZE)

                
            # y座標が下をこえる
            elif ((self.all_agent[i]['p']+self.all_agent[i]['v'])[1] < -SIZE):
                
                # ゴール判定
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, step)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        self.goal_count[i] = self.goal_count[i] + 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1] + 2*SIZE
                    else:        
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agent[i]['p'][1] = SIZE + (
                    (self.all_agent[i]['p']+self.all_agent[i]['v'])[1] + SIZE
                )
                
            # y座標が上をこえる     
            elif ((self.all_agent[i]['p']+self.all_agent[i]['v'])[1] > SIZE):
                
                # ゴール判定
                if (self.all_agent[i]['p'][0] > 
                   self.agent_goal[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agent[i]['p'][0] < 
                   self.agent_goal[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agent[i]['p'][1] > 
                   self.agent_goal[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agent[i]['p'][1] < 
                   self.agent_goal[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i, step)
                        if not completion_time == None:
                            self.completion_time.append(completion_time)
                        self.goal_count[i] = self.goal_count[i] + 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goal[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goal[i][self.goal_count[i]][1] + (-2*SIZE)
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        

                self.all_agent[i]['p'][1] = -SIZE + (
                    (self.all_agent[i]['p']+self.all_agent[i]['v'])[1] - SIZE
                )

                
        for i in range(self.agent):
            # 移動後の座標を確定      
            self.all_agent[i]['p'] = self.all_agent[i]['p'] + self.all_agent[i]['v']
            
         
    # 9. 完了時間を記録
    def calc_completion_time(self, num: int, now_step: int) -> float:
        """ 
        ex. self.calc_completion_time(num=10, now_step=10) -> 35.6
        """
        # 一回のゴールにおける初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = now_step
        
        # 一回目のゴール
        if (self.start_step[num] == 1):
            # 一回目のゴールにおける、ゴール位置を記録
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
            
        # 一回目以降のゴール
        else:
            # 前回のゴールが左端にあるとき
            if (self.goal_pos[num][0] == -SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0] + 2*SIZE
                self.start_pos[num][1] = self.goal_pos[num][1]
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
                
            # 前回のゴールが右端にあるとき
            elif (self.goal_pos[num][0] == SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0] + (-2*SIZE)
                self.start_pos[num][1] = self.goal_pos[num][1]
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
            
            # 前回のゴールが下端にあるとき
            elif (self.goal_pos[num][1] == -SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0]
                self.start_pos[num][1] = self.goal_pos[num][1] + 2*SIZE
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
                
            # 前回のゴールが上端にあるとき
            elif (self.goal_pos[num][1] == SIZE):
                # スタート位置、ゴール位置を記録
                self.start_pos[num][0] = self.goal_pos[num][0]
                self.start_pos[num][1] = self.goal_pos[num][1] + (-2*SIZE)
                self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
                
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / \
                          (np.linalg.norm(self.start_pos[num] - self.goal_pos[num]))
        
        # 外れ値を除外
        if (completion_time > 200):
            print("消します")
            print(completion_time)
            return None
        
        return completion_time
    
    
    # 10. 最後の座標から完了時間を算出(やらなくても良いかもしれない)
    def calc_last_completion_time(self, num: int) -> float:
        
        # 初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = self.step
        
        # ゴールする前に境界をはみ出ている場合
        if not (self.goal_temp[num][0] == 0 and self.goal_temp[num][1] == 0):
            # 左右の境界をはみ出た
            if (abs(self.all_agent[num]['p'][0]) > abs(self.all_agent[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agent[num]['p'][0] = -self.all_agent[num]['p'][0]
            # 上下の境界をはみ出た
            elif (abs(self.all_agent[num]['p'][0]) < abs(self.all_agent[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agent[num]['p'][1] = -self.all_agent[num]['p'][1]
            
        # スタート位置、ゴール位置を算出
        # 前回のゴールが左端にあるとき
        if (self.goal_pos[num][0] == -SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0] + 2 * SIZE
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]

        # 前回のゴールが右端にあるとき
        elif (self.goal_pos[num][0] == SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0] + (-2 * SIZE)
            self.start_pos[num][1] = self.goal_pos[num][1]
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]

        # 前回のゴールが下端にあるとき
        elif (self.goal_pos[num][1] == -SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + 2 * SIZE
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]

        # 前回のゴールが上端にあるとき
        elif (self.goal_pos[num][1] == SIZE):
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + (-2 * SIZE)
            self.goal_pos[num] = self.agent_goal[num][self.goal_count[num]]
            
        
        # スタートからゴールまでの直線の式を計算
        # 傾き
        a = (self.start_pos[num][1] - self.goal_pos[num][1]) / \
            (self.start_pos[num][0] - self.goal_pos[num][0])
        # 切片
        b = -(a * self.start_pos[num][0]) + self.start_pos[num][1]
        
        # エージェントの位置を通り、スタートからゴールまでの直線に垂直な直線の式を計算
        # 傾き
        c = (-1) / a
        # 切片
        d = -(c * self.all_agent[num]['p'][0]) + self.all_agent[num]['p'][1]

        # 2つの直線の交点を算出
        cross_x = (b - d) / (-(a - c))
        cross_y = a * cross_x + b
        cross = np.array([cross_x, cross_y])
        
        # スタートから交点までの距離を計算
        distance = np.linalg.norm(self.start_pos[num] - cross)
        
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / distance

        if (completion_time > 200 or completion_time < 10):
            print("消しました")
            print(completion_time)
            return None
        
        return completion_time


    # 11. 座標を送る
    def show_image(self) -> np.array: # shape(2(x,y), エージェントの数)
        """
        ex. self.show_image() -> array([[2.0, 1.2,...,3.5],
                                       [2.5, 1.5,...,1.5]]) (2, 25)
        """
        pos_array = np.zeros([2, self.agent])
        for i in range(self.agent):
            pos_array[0][i] = self.all_agent[i]['p'][0]
            pos_array[1][i] = self.all_agent[i]['p'][1]
            
        return pos_array


    # 12. 各エージェントの座標を、エージェントの番号付きでプロットする
    # 薄い青色がそのstepでの位置で、濃い青色は次のstepでの位置
    def plot_positions(self) -> None:
        pos_array = self.show_image()
        plt.figure(figsize=(8, 8))
        for i in range(self.agent):
            next_pos = self.all_agent[i]['p'] + self.all_agent[i]['v']
            plt.scatter(pos_array[0][i], pos_array[1][i], color='blue', alpha=0.3)
            plt.scatter(*next_pos, color='blue')
            plt.annotate(i, xy=(pos_array[0][i], pos_array[1][i]))
        plt.show()
        

def debug_theta(s, num, other):
    a = s.all_agent[num]['p'] + s.all_agent[num]['v']
    b = s.all_agent[num]['p']
    line1 = np.array([b, a])
    
    c = s.all_agent[other]['p']
    line2 = np.array([b, c])
    theta = calc_angle_two_lines(line1, line2)
    
    print('rad:', theta, 'deg:', np.rad2deg(theta))
    
