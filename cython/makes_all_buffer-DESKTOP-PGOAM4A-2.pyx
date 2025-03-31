# cython: language_level = 3
from copy import deepcopy
from tqdm import tqdm
import time
import pandas as pd
import cython
from cython import boundscheck, wraparound
import numpy as np
cimport numpy as cnp
cnp.import_array()
#from libcpp.vector cimport vector

cdef int FIELD_SIZE = 5

# numpy配列はmemoryviewer形式にしてインデックスアクセスをする

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef double calc_rad (double[:] pos2, 
                      double[:] pos1): 
    """
    pos1からpos2のベクトルの角度を返す
    ex. calc_rad(pos2=np.array([1.5, 2.5]), pos1=np.array([3.0, 1.0])) -> 2.4
    """    
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]).astype(np.float64)


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef double[:] rotate_vec (double[:] vec,
                           double rad):
    """
    ベクトルをradだけ回転させる (回転行列)
    ex. rotate_vec(vec=np.array([3.0, 5.0]), rad=1.2) -> array([-3.6, 4.6])
    """
    cdef:
        double sin, m_sin, cos
        double[:,:] rotation

    sin = np.sin(rad)
    m_sin = -np.sin(rad)
    cos = np.cos(rad)
    
    rotation = np.array([[cos, m_sin], [sin, cos]])

    return np.dot(rotation, vec.T).T.astype(np.float64)


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef class Simulation:     
    cdef public:
        int num_steps 
        int interval
        int num_agents
        int view
        int viewing_angle
        int random_seed
        int avoidance
        int n, i, j
        int current_step
        int num_dynamic_agent 
        bint returned_results
        
        list data
        
        int[:] goal_count
        double agent_size
        double goal_vec
        double dynamic_percent
        double simple_avoid_vec
        double dynamic_avoid_vec
        double exe_time
        double[:] completion_time
        double[:] start_step
        double[:] goal_step
        double[:] agent_pos
        double[:] agent_vel
        double[:] visible_agent_pos
        double[:] visible_agent_vel
        double[:,:] dist
        double[:,:] vel
        double[:,:] goal_temp
        double[:,:] goal_pos
        double[:,:] first_pos
        double[:,:] start_pos
        double[:,:,:] first_agent
        double[:,:,:] agent_goals
        double[:,:,:] all_agents
        
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    def __init__(self, 
                  int num_steps=500,
                  int interval=100,
                  int num_agents=25, 
                  int view=1, 
                  int viewing_angle=360, 
                  double agent_size=0.1, 
                  double goal_vec=0.06, 
                  double dynamic_percent=1.0,
                  double simple_avoid_vec=0.06, 
                  double dynamic_avoid_vec=0.06,
                  int random_seed=0):
        
        cdef:
            double[:] zero_arr
            double[:] goal_arr
            double[:] rot_v
            double[:] pos
            double[:] row
            double[:] goal
            
            # コピー用の変数はnumpy配列にしてdeepcopyが使えるようにする。memoryviewerではcopyが使えない。
            cnp.ndarray[double, ndim=3] all_agents2
            cnp.ndarray[double, ndim=3] first_agent
            
        self.num_steps = num_steps
        self.interval = interval # 100msごとにグラフを更新してアニメーションを作成
        self.agent_size = agent_size # エージェントの半径(目盛り) = 5px
        self.num_agents = num_agents # エージェント数
        self.view = view # 視野の半径(目盛り) = 50px:エージェント5体分
        self.viewing_angle = viewing_angle # 視野の角度
        self.goal_vec = goal_vec # ゴールベクトルの大きさ(目盛り)
        self.dynamic_percent = dynamic_percent # 動的回避を基に回避するエージェントの割合
        self.simple_avoid_vec = simple_avoid_vec # 単純回避での回避ベクトルの大きさ(目盛り)
        self.dynamic_avoid_vec = dynamic_avoid_vec # 動的回避での回避ベクトルの最大値(目盛り)
        self.random_seed = random_seed
        
        self.all_agents = np.zeros([num_agents, 3, 2]) # 全エージェントの座標を記録
        self.agent_goals = np.zeros([num_agents, 8, 2])
        self.first_pos = np.zeros([num_agents, 2])
        
        #エージェント間の距離を記録するリスト
        self.dist = np.zeros([num_agents, num_agents])
                
        # 動的回避を行うエージェントの数
        self.num_dynamic_agent = int(np.round(num_agents*dynamic_percent))
        
        self.agent_pos = np.zeros(2)
        self.agent_vel = np.zeros(2)
        self.visible_agent_pos = np.zeros(2)
        self.visible_agent_vel = np.zeros(2)
        
        # エージェントの生成
        goal_arr = np.array([goal_vec, 0.])
        zero_arr = np.array([0., 0.])
        
        np.random.seed(random_seed)
        for n in range(num_agents):
            # グラフ領域の中からランダムに座標を決定
            pos = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            vel = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            rot_v = rotate_vec(goal_arr, calc_rad(vel, zero_arr))

            all_pos = np.zeros([num_steps+1, 2])
            all_vel = np.zeros(num_steps+1)
            
            # 1は動的回避で、0は単純回避
            if n < self.num_dynamic_agent:
                avoidance = 1
            else:
                avoidance = 0
            
            # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
            # self.all_agentsの1つの要素に1体のエージェントの位置と速度が記録
            # P(t) - V(t) = P(t-1)
            self.all_agents[n][0] = avoidance
            self.all_agents[n][1] = pos
            self.all_agents[n][2] = rot_v
            
        # 初期位置と初期速度をコピー
        all_agents2 = deepcopy(np.asarray(self.all_agents))
        first_agent = deepcopy(np.asarray(self.all_agents))
        
        # エージェントの初期位置を保存
        for i in range(self.num_agents):
            self.first_pos[i][0] = first_agent[i][1][0]
            self.first_pos[i][1] = first_agent[i][1][1]
            
        # エージェントにゴールを8ずつ設定
        for i in range(self.num_agents):
            for j in range(8):
                goal = self.set_goals(all_agents2[i][1], all_agents2[i][2])
                self.agent_goals[i][j][0] = goal[0]
                self.agent_goals[i][j][1] = goal[1]
                
        # ゴールした回数を記録するリスト
        self.goal_count = np.zeros(num_agents, np.int32)
        
        # はみ出た時用のゴール
        self.goal_temp = np.zeros([num_agents, 2])
        
        # 完了時間を測るための変数
        self.start_step = np.zeros([num_agents])
        self.goal_step = np.zeros([num_agents])
        self.goal_pos = np.zeros([num_agents, 2])
        self.start_pos = self.first_pos
        
        # 完了時間を記録するリスト
        self.completion_time = np.array([])
        
        self.data = []
        #self.data = np.array([])
        row = self.record_agent_information() # 全エージェントの位置と速度、接近を記録
        #self.data = np.append(self.data, row)
        self.data.append(row) # ある時刻でのエージェントの情報が記録されたrowが集まってdataとなる
        
        self.exe_time = 0
        
        self.current_step = 0
        self.update_parameters()
        
        self.returned_results = False

    
    cpdef void disp_info (self):
        print('\nシミュレーションの環境情報')
        print('-----------------------------')
        print('ランダムシード:', self.random_seed)
        print('ステップ数:', self.num_steps)
        print('エージェント数:', self.num_agents)
        print('エージェントの視野角度:', self.viewing_angle)
        print('動的回避エージェントの割合:', self.dynamic_percent)
        print('単純回避の回避量:', self.simple_avoid_vec*50, 'px')
        print('-----------------------------\n')
    
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef void update_parameters (self):
        """
        all_agentsのパラメータを更新する
        """
        cdef: 
            int i 
            double[:] pos
            double vel
        
        for i in range(self.num_agents):
            pos = self.all_agents[i][1]
            vel = np.linalg.norm(self.all_agents[i][2])
    
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef int[:] find_visible_agents (self, 
        double[:,:] dist_all, 
        int num
    ):
        """
        エージェント番号numの視野に入った他のエージェントの番号をリストにして返す
        ex. self.find_visible_agents(dist_all, 5) -> [14, 20, 30]
        """
        cdef: # return
            #int[:] visible_agents
            cnp.ndarray[int, ndim=1] visible_agents
        cdef:
            #list near_agents
            #int[:] near_agents
            cnp.ndarray[int, ndim=1] near_agents
            int i
            double x
            double angle_difference
            double goal_angle
            double[:] my_pos
            double[:] other_pos
            double[:] goal_pos
            
            
        # near_agentsは360度の視野に入ったエージェント
        # visible_agentsは視野を狭めた場合に視野に入ったエージェント
        near_agents = np.array([], dtype=np.int32)
        for i, x in enumerate(dist_all[num]):
            if x != -0.2 and x < self.view:
                near_agents = np.append(near_agents, i).astype(np.int32)
                
        visible_agents = np.array([], dtype=np.int32)
        
        # ゴールベクトルの角度を算出する
        my_pos = self.agent_goals[num][self.goal_count[num]]
        goal_pos = self.all_agents[num][1]
        
        goal_angle = np.degrees(
            calc_rad(my_pos, goal_pos)
        )

        for i in near_agents:
            # 近づいたエージェントとの角度を算出
            other_pos = self.agent_goals[i][self.goal_count[i]]
            
            agent_angle = np.degrees(
                calc_rad(other_pos, goal_pos)
            )
            
            # 近づいたエージェントとの角度とゴールベクトルの角度の差を計算
            angle_difference = abs(goal_angle - agent_angle)
            
            if angle_difference > 180:
                angle_difference = 360 - angle_difference
                
            # 視野に入っているエージェントをvisible_agentsに追加
            if angle_difference <= self.viewing_angle / 2:
                visible_agents = np.append(visible_agents, i).astype(np.int32)
        
        return visible_agents
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double[:,:] calc_distance_all_agents (self):
        cdef: # return
            double[:,:] dist_all
        cdef:
            int i, j
            double norm
            double[:] d
            
        d = np.zeros(2)
        dist_all = np.zeros([self.num_agents, self.num_agents])
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                d[0] = self.all_agents[i][1][0] - self.all_agents[j][1][0]
                d[1] = self.all_agents[i][1][1] - self.all_agents[j][1][1]
                norm = np.linalg.norm(d)
                dist_all[i][j] = norm - 2*self.agent_size
        
        return dist_all
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double[:] simple_avoidance (self, 
        int num
    ):
        cdef: # return
            double[:] avoid_vec
        cdef:
            int[:] visible_agents
            int i
            double to_vec1
            double[:] d
            double[:,:] dist_all
        
        dist_all = self.calc_distance_all_agents()
        visible_agents = self.find_visible_agents(dist_all, num)
        avoid_vec = np.zeros(2)   # 回避ベクトル
        
        if len(visible_agents) == 0:
            return avoid_vec    
        
        ### the followings are simple vectors ###
        for i in visible_agents:
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = np.zeros(2)
            
            d[0] = self.all_agents[num][1][0] - self.all_agents[i][1][0]
            d[1] = self.all_agents[num][1][1] - self.all_agents[i][1][1]
            
            d[0] = d[0] / (dist_all[num][i] + 2 * self.agent_size)
            d[1] = d[1] / (dist_all[num][i] + 2 * self.agent_size)
            
            d[0] = d[0] * self.simple_avoid_vec
            d[1] = d[1] * self.simple_avoid_vec
    
            if not np.isnan(np.asarray(d)).all():
                avoid_vec[0] += d[0] # ベクトルの合成
                avoid_vec[1] += d[1]
            
        # # ベクトルの平均を出す
        # ave_avoid_vec = avoid_vec / len(visible_agents)
        
        # return ave_avoid_vec
        
        # ベクトルを平均しない
        return avoid_vec
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double[:] dynamic_avoidance (self, 
        int num
    ):
        """
        動的回避ベクトルの生成
        ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
        """
        cdef:
            double[:] avoid_vec
        cdef:
            double[:,:] dist_all
            int[:] visible_agents
            int i
            int t
            double dist_former
            double dist_latter
            double tcpa
            double dcpa
            double braking_index
            double bw_a, bw_b, bw_c, bw_d
            double[:] d
        
        dist_all = self.calc_distance_all_agents()
        avoid_vec = np.zeros(2) # 回避ベクトル
        
        visible_agents = self.find_visible_agents(dist_all, num)   

        if len(visible_agents) == 0:
            return avoid_vec    
                    
        ### the followings are dynamic vectors ###
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.agent_pos = self.all_agents[num][1]
            self.agent_vel = self.all_agents[num][2]
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agents[i][1]
            self.visible_agent_vel = self.all_agents[i][2]
   
            dist_former = dist_all[num][i]
            
            t = 0
            # 2体のエージェントを1ステップ動かして距離を測定
            self.agent_pos += np.asarray(self.agent_vel)
            self.visible_agent_pos += np.asarray(self.visible_agent_vel)
            d = np.asarray(self.agent_pos) - np.asarray(self.visible_agent_pos)
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
                    t += self.interval
                    self.agent_pos += np.asarray(self.agent_vel)
                    self.visible_agent_pos += np.asarray(self.visible_agent_vel)
                    d = np.asarray(self.agent_pos) - np.asarray(self.visible_agent_pos)
                    dist_latter = np.linalg.norm(d) - 2 * self.agent_size
                    
                if dist_former < 0:
                    dcpa = 0 # 最も近い距離で接触している場合はdcpaは0とみなす
                else:
                    dcpa = dist_former * 50 # 単位をピクセルに変換
                    
                tcpa = t

            # ブレーキ指標の重み
            bw_a = -5.145
            bw_b = 3.348
            bw_c = 4.286
            bw_d = -13.689
            
            # ブレーキ指標の算出
            braking_index = (1 / (1 + np.exp(-bw_c - bw_d * (tcpa/4000)))) * \
                            (1 / (1 + np.exp(-bw_b - bw_a * (dcpa/50))))
            
        
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = np.zeros(2)
            d[0] = self.all_agents[num][1][0] - self.all_agents[i][1][0]
            d[1] = self.all_agents[num][1][1] - self.all_agents[i][1][1]
            
            # ベクトルの大きさを1にする
            d[0] = d[0] / (dist_all[num][i] + 2 * self.agent_size)
            d[1] = d[1] / (dist_all[num][i] + 2 * self.agent_size)
            
            # ブレーキ指標の値を反映
            d[0] = d[0] * braking_index
            d[1] = d[1] * braking_index
            
            # ベクトルの最大値を決定
            d[0] = d[0] * self.dynamic_avoid_vec
            d[1] = d[1] * self.dynamic_avoid_vec
            
            if not np.isnan(np.asarray(d)).all():
                # ベクトルの合成
                avoid_vec[0] += d[0]                 
                avoid_vec[1] += d[1]
                
        # # ベクトルの平均を出す
        # ave_avoid_vec = avoid_vec / len(visible_agents)

        # return ave_avoid_vec
        
        # ベクトルを平均しない
        return avoid_vec

    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double[:] set_goals (self, 
        double[:] agent_pos,
        double[:] agent_vel
    ):
        cdef:
            double[:] goal

        goal = np.zeros(2)
        while True:
            goal[0] = agent_pos[0] + agent_vel[0]
            goal[1] = agent_pos[1] + agent_vel[1]
            
            # x座標がグラフ領域を超える
            if (agent_pos[0] + agent_vel[0]) < -FIELD_SIZE:
                # 超えた時の座標をゴールとする                                
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > FIELD_SIZE - 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] =  goal[1] - 0.1
                    #print("調整入りました")
                    
                # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    # ゴールの座標がグラフ領域の角にならないように調整
                    goal[1] = goal[1] + 0.1
                    #print("調整入りました")
                    
                goal[0] = -FIELD_SIZE
                # 端に到達したエージェントを、反対側の端に移動させる
                agent_pos[0] = FIELD_SIZE + ((agent_pos[0] + agent_vel[0]) + FIELD_SIZE)
                break
                        
            elif (agent_pos[0] + agent_vel[0]) > FIELD_SIZE:
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > FIELD_SIZE - 0.1):
                    goal[1] = goal[1] - 0.1
                    #print("調整入りました")
    
               # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    goal[1] = goal[1] + 0.1
                    #print("調整入りました")
                    
                goal[0] = FIELD_SIZE
                agent_pos[0] = -FIELD_SIZE + ((agent_pos[0] + agent_vel[0]) - FIELD_SIZE)
                break
                
                
            # y座標がグラフ領域を超える
            elif (agent_pos[1] + agent_vel[1]) < -FIELD_SIZE:
                # 超えた時の座標をゴールとする
                goal[1] = -FIELD_SIZE
                
                agent_pos[1] = FIELD_SIZE + ((agent_pos[1] + agent_vel[1]) + FIELD_SIZE)
                break
                                        
            elif (agent_pos[1] + agent_vel[1]) > FIELD_SIZE:
                goal[1] = FIELD_SIZE
                
                agent_pos[1] = -FIELD_SIZE + ((agent_pos[1] + agent_vel[1]) - FIELD_SIZE)
                break
                
            # エージェントを初期速度のまま動かす
            agent_pos[0] += agent_vel[0]
            agent_pos[1] += agent_vel[1]
    
        return goal


    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef void record_start_and_goal (self, 
        int num
    ):
        """
        完了時間を記録するためのスタート位置とゴール位置を記録
        更新されるパラメータ：
        self.start_pos
        self.goal_pos
        """
        cdef double[:] arr_goal_pos
        cdef double[:] goal
    
        arr_goal_pos = np.asarray(self.goal_pos[num])
        
        # 前回のゴールが左端にあるとき
        if (self.goal_pos[num][0] == -FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0] + 2*FIELD_SIZE
            self.start_pos[num][1] = self.goal_pos[num][1]
            goal = self.agent_goals[num][self.goal_count[num]]
            self.goal_pos[num][0] = goal[0] 
            self.goal_pos[num][1] = goal[1]
            
        # 前回のゴールが右端にあるとき
        elif (self.goal_pos[num][0] == FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0] + (-2*FIELD_SIZE)
            self.start_pos[num][1] = self.goal_pos[num][1]
            goal = self.agent_goals[num][self.goal_count[num]]
            self.goal_pos[num][0] = goal[0] 
            self.goal_pos[num][1] = goal[1]       
            
        # 前回のゴールが下端にあるとき
        elif (self.goal_pos[num][1] == -FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + 2*FIELD_SIZE
            goal = self.agent_goals[num][self.goal_count[num]]
            self.goal_pos[num][0] = goal[0]
            self.goal_pos[num][1] = goal[1]
            
        # 前回のゴールが上端にあるとき
        elif (self.goal_pos[num][1] == FIELD_SIZE):
            # スタート位置、ゴール位置を記録
            self.start_pos[num][0] = self.goal_pos[num][0]
            self.start_pos[num][1] = self.goal_pos[num][1] + (-2*FIELD_SIZE)
            goal = self.agent_goals[num][self.goal_count[num]]
            self.goal_pos[num][0] = goal[0]
            self.goal_pos[num][1] = goal[1]
            
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef void check_if_goaled (self):
        """
        各エージェントがゴールに到達したかどうかをチェックする
        ゴールしていた場合、完了時間の算出、ゴールカウントの更新、はみ出た時用のゴールの初期化を行う
        更新されるパラメータ：
        self.completion_time
        self.goal_count
        self.goal_temp
        self.all_agents
        """
        cdef:
            int i
            double comletion_time
        
        for i in range(self.num_agents):
            # x座標が左端をこえる
            if (self.all_agents[i][1][0] + self.all_agents[i][2][0]) < -FIELD_SIZE:
                # ゴールに到着
                if (self.all_agents[i][1][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i][1][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1
                   and 
                   self.all_agents[i][1][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i][1][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # 通常のゴールに到着
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not (completion_time > 200):
                            self.completion_time = np.append(
                                self.completion_time, completion_time
                            )
                        # ゴールした回数を更新
                        self.goal_count[i] += 1
                        
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
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0] + 2*FIELD_SIZE
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1]
                        
                    # はみ出た時用のゴールが設定されている
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                
                # エージェントを反対の端へ移動
                self.all_agents[i][1][0] = FIELD_SIZE + (
                    (self.all_agents[i][1][0] + self.all_agents[i][2][0]) + FIELD_SIZE
                )
        
            
            # x座標が右端をこえる
            elif (self.all_agents[i][1][0] + self.all_agents[i][2][0]) > FIELD_SIZE:
                
                # ゴール判定
                if (self.all_agents[i][1][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i][1][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i][1][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i][1][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append(
                                self.completion_time, completion_time
                            )
                        self.goal_count[i] += 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0] + (-2*FIELD_SIZE)
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1]
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agents[i][1][0] = -FIELD_SIZE + (
                    (self.all_agents[i][1][0] + self.all_agents[i][2][0]) - FIELD_SIZE
                )

                
            # y座標が下をこえる
            elif (self.all_agents[i][1][1] + self.all_agents[i][2][1]) < -FIELD_SIZE:
                
                # ゴール判定
                if (self.all_agents[i][1][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i][1][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i][1][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i][1][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append( 
                                self.completion_time, completion_time
                            )
                        self.goal_count[i] += 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1] + 2*FIELD_SIZE
                    else:        
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                self.all_agents[i][1][1] = FIELD_SIZE + (
                    (self.all_agents[i][1][1] + self.all_agents[i][2][1]) + FIELD_SIZE
                )
                
            # y座標が上をこえる     
            elif (self.all_agents[i][1][1] + self.all_agents[i][2][1]) > FIELD_SIZE:
                
                # ゴール判定
                if (self.all_agents[i][1][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i][1][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i][1][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i][1][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append(
                                self.completion_time, completion_time
                            )
                        self.goal_count[i] += 1
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        
                else:
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 境界をこえた用のゴールを設定
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0]
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1] + (-2*FIELD_SIZE)
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                        

                self.all_agents[i][1][1] = -FIELD_SIZE + (
                    (self.all_agents[i][1][1] + self.all_agents[i][2][1]) - FIELD_SIZE
                )

    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef void move_agents (self):
        """
        エージェントを動かす
        更新されるパラメータ：
        self.all_agents
        """
        cdef:
            int i
            double[:] goal_arr
            double[:] row
            double[:] simple_vec
            double[:] dynamic_vec
        
        goal_arr = np.array([self.goal_vec, 0])
        for i in range(self.num_agents):
            # はみ出た時用のゴールが設定されていない
            # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                self.all_agents[i][2] = rotate_vec(
                    goal_arr, 
                    calc_rad(self.agent_goals[i][self.goal_count[i]],
                             self.all_agents[i][1])
                )     

            # はみ出た時用のゴールが設定されている
            # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            else:
                self.all_agents[i][2] = rotate_vec(
                    goal_arr, 
                    calc_rad(np.asarray(self.goal_temp[i]), self.all_agents[i][1])
                ) 
                
                
            # 回避ベクトルを足す
            if self.all_agents[i][0][0] == 0: # 単純回避ベクトルを足す
                simple_vec = self.simple_avoidance(i)
                self.all_agents[i][2][0] += simple_vec[0]
                self.all_agents[i][2][1] += simple_vec[1]
                
            elif self.all_agents[i][0][0] == 1: # 動的回避ベクトルを足す
                dynamic_vec = self.dynamic_avoidance(i)
                self.all_agents[i][2][0] += dynamic_vec[0]
                self.all_agents[i][2][1] += dynamic_vec[1]
                
        self.current_step += 1
        self.check_if_goaled()
        
        for i in range(self.num_agents):
            # 移動後の座標を確定      
            self.all_agents[i][1][0] += self.all_agents[i][2][0]
            self.all_agents[i][1][1] += self.all_agents[i][2][1]
            
        self.update_parameters()    
        row = self.record_agent_information()
        #self.data = np.append(self.data, row)
        self.data.append(row)
    
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cpdef void simulate (self):
        cdef:
            int t
            double start, fin
        
        self.disp_info()
        start = time.perf_counter() # 実行時間を結果csvに記録する
        
        for t in tqdm(range(self.num_steps)):
            self.move_agents()
            
        fin = time.perf_counter()
        self.exe_time = fin - start
        
        
    cdef int[:,:] approach_detect (self, 
        double dist
    ): 
        """ 
        指定した距離より接近したエージェントの数を返す
        ex. self.approach_detect(dist=0.5) -> array([[0, 3],[1, 2],...[24, 1]])
        """
        cdef: # return
            int[:,:] approach_agent
        cdef:
            double[:,:] dist_all
            int t
            int len_visible_agents
            double x
        
        dist_all = self.calc_distance_all_agents()
        approach_agent = np.zeros([self.num_agents, 2], dtype=np.int32)
        
        # それぞれのエージェントについて、distより接近したエージェントの数を記録
        for t in range(self.num_agents):
            visible_agents = [i for i, x in enumerate(dist_all[t]) 
                              if x != -(0.2) and x < dist]
            len_visible_agents = len(visible_agents)
            
            approach_agent[t][0] = t
            approach_agent[t][1] = len_visible_agents
                                       
        return approach_agent
    
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double[:] record_agent_information (self):
        cdef: # return
            double[:] row
        cdef:
            int i
            int[:,:] collision_agent
            
        # 初期の位置と速度を記録
        row = np.concatenate([self.all_agents[0][1], self.all_agents[0][2]])
        # rowにはある時刻の全エージェントの位置と速度が入る
        for i in range(1, self.num_agents):
            row = np.concatenate([row, self.all_agents[i][1], self.all_agents[i][2]])
            
        # エージェントの接近を記録
        # 衝突したエージェント
        collision_agent = self.approach_detect(0)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])

        # 視野の半分まで接近したエージェント
        collision_agent = self.approach_detect(0.5)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])
            
        # 視野の4分の1まで接近したエージェント
        collision_agent = self.approach_detect(0.25)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])

        # 視野の8分の1まで接近したエージェント
        collision_agent = self.approach_detect(0.125)
        for i in range(self.num_agents):
            row = np.append(row, collision_agent[i][1])
            
        return row


    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef int[:] record_approaches (self,
        str approach_dist, 
        int step, 
        list sim_data
    ):
        """
        エージェント同士が接近した回数を、接近度合い別で出力する
        ex. self.record_approaches('collision', STEP=500, data) -> [0,0,3,...,12]
        """
        cdef: # return 
            int[:] approach 
        cdef:
            int start
            int stop
            int i
            int total
            
        if approach_dist == 'collision':
            start, stop = 4*self.num_agents, 5*self.num_agents
            
        elif approach_dist == 'half':
            start, stop = 5*self.num_agents, 6*self.num_agents
            
        elif approach_dist == 'quarter':
            start, stop = 6*self.num_agents, 7*self.num_agents
            
        elif approach_dist == 'one_eigth':
            start, stop = 7*self.num_agents, 8*self.num_agents
            
        approach = np.array([], dtype=np.int32)
        for i in range(start, stop, 1):
            total = 0
            for j in range(step):
                # 一試行で何回エージェントに衝突したか
                total += sim_data[j+1][i]
            
            # 全エージェントの衝突した回数を記録
            approach = np.append(approach, total)
        
        return approach


    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double calc_completion_time (self, 
        int num
    ):
        """ 
        ゴールまで到達した際の完了時間を記録
        ex. self.calc_completion_time(num=10) -> 35.6
        """
        cdef:
            double completion_time
        cdef:
            double to_goal
            double[:] goal_pos
        
        # 一回のゴールにおける初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = self.current_step
        
        # 一回目のゴール
        if (self.start_step[num] == 1):
            # 一回目のゴールにおける、ゴール位置を記録
            goal_pos = self.agent_goals[num][self.goal_count[num]]
            self.goal_pos[num][0] = goal_pos[0]
            self.goal_pos[num][1] = goal_pos[1]
            
        # 一回目以降のゴール
        else:
            self.record_start_and_goal(num)
            
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        to_goal = np.linalg.norm(
            np.asarray(self.start_pos[num]) - self.goal_pos[num]
        )
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / to_goal
  
        return completion_time
    
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef double calc_remained_completion_time (self, 
        int num, 
        int step
    ):
        """
        1試行が終わり、ゴールに向かう途中の最後の座標から完了時間を算出(やらなくても良いかもしれない)
        """
        cdef:
            double completion_time 
        cdef: 
            double a, b, c, d
            double cross_x, cross_y
            double[:] cross
            double distance
        
        # 初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = step
        
        # ゴールする前に境界をはみ出ている場合
        if not (self.goal_temp[num][0] == 0 and self.goal_temp[num][1] == 0):
            # 左右の境界をはみ出た
            if (abs(self.all_agents[num][1][0]) > abs(self.all_agents[num][1][1])):
                # はみ出る前に戻してあげる
                self.all_agents[num][1][0] = -self.all_agents[num][1][0]
            # 上下の境界をはみ出た
            elif (abs(self.all_agents[num][1][0]) < abs(self.all_agents[num][1][1])):
                # はみ出る前に戻してあげる
                self.all_agents[num][1][1] = -self.all_agents[num][1][1]
            
        self.record_start_and_goal(num)
        
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
        d = -(c * self.all_agents[num][1][0]) + self.all_agents[num][1][1]

        # 2つの直線の交点を算出
        cross_x = (b - d) / (-(a - c))
        cross_y = a * cross_x + b
        cross = np.array([cross_x, cross_y])
        
        # スタートから交点までの距離を計算
        distance = np.linalg.norm(self.start_pos[num] - np.asarray(cross))
        
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / distance
        
        return completion_time
    
    
    def return_results_as_df(self) -> pd.core.frame.DataFrame:
        """
        1試行の記録をデータフレームにして返す
        """
        # この関数は破壊的操作で1回目と2回目の値が変化するため、2回目以降の呼び出しを禁止する
        assert self.returned_results == False, "Results have been already returend."
        
        # 最後の座標から完了時間を算出
        for i in range(self.num_agents):
            last_completion_time = self.calc_remained_completion_time(i, self.num_steps)
            if not (last_completion_time > 200 or last_completion_time < 10):
                self.completion_time = np.append(
                    self.completion_time, last_completion_time
                )
       
        # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
        collision = self.record_approaches('collision', self.num_steps, self.data)
        half =  self.record_approaches('half', self.num_steps, self.data)
        quarter =  self.record_approaches('quarter', self.num_steps, self.data)
        one_eighth =  self.record_approaches('one_eigth', self.num_steps, self.data)
        
        # collision = self.approach_detect(0)
        
        # approach = np.array([], dtype=np.int32)
        # for i in collision:
        #     total = 0
        #     for j in range(self.num_steps):
        #         # 一試行で何回エージェントに衝突したか
        #         total += self.data[j+1][i]
        #     # 全エージェントの衝突した回数を記録
        #     approach = np.append(approach, total)
        # collision_mean = np.mean(collision)
        
        # half =  self.approach_detect(1/2)
        # quarter =  self.approach_detect(1/4)
        # one_eighth =  self.approach_detect(1/8)
        
        # 各指標の平均を計算
        collision_mean = np.mean(collision)
        half_mean = np.mean(half)
        quarter_mean = np.mean(quarter)
        one_eighth_mean = np.mean(one_eighth)
        completion_time_mean = np.mean(self.completion_time)

        # 結果のデータを保存
        dict_result = {
            'time': completion_time_mean,
            'half': half_mean,
            'quarter': quarter_mean,
            'one_eigth': one_eighth_mean,
            'collision': collision_mean,
            'agent': self.num_agents,
            'viewing_angle': self.viewing_angle,
            'num_steps': self.num_steps,
            'dynamic_percent': self.dynamic_percent,
            'simple_avoid_vec': self.simple_avoid_vec,
            'sum_goal_count': np.sum(self.goal_count),
            'exe_time_second': np.round(self.exe_time, 3),
            'exe_time_min': np.round(self.exe_time/60, 3)
        }
        
        df_result = pd.DataFrame(dict_result, index=[f'seed_{self.random_seed}'])
        self.returned_results = True
        
        return df_result
    
