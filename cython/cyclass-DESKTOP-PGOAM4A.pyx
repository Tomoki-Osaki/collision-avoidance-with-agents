# cython: language_level = 3
from copy import deepcopy
#from tqdm import tqdm
#import time
#import pandas as pd
import cython
from cython import boundscheck, wraparound
import numpy as np
cimport numpy as cnp
#from libcpp.vector cimport vector

ctypedef cnp.float64_t DTYPE_t

cdef int FIELD_SIZE = 5

# 計算が面倒になるので、極力memoryviewerは使わない。クラス変数にのみ例外的に使用する必要がある。
# DTYPE_t[:]
# cnp.ndarray[DTYPE_t, ndim=1]

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef double calc_rad (
    cnp.ndarray[DTYPE_t, ndim=1] pos2,
    cnp.ndarray[DTYPE_t, ndim=1] pos1
): 
    """
    pos1からpos2のベクトルの角度を返す
    ex. calc_rad(pos2=np.array([1.5, 2.5]), pos1=np.array([3.0, 1.0])) -> 2.4
    """    
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])


@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef cnp.ndarray[DTYPE_t, ndim=1] rotate_vec (
    cnp.ndarray[DTYPE_t, ndim=1] vec,
    double rad
):
    """
    ベクトルをradだけ回転させる (回転行列)
    ex. rotate_vec(vec=np.array([3.0, 5.0]), rad=1.2) -> array([-3.6, 4.6])
    """
    cdef:
        double sin, m_sin, cos
        cnp.ndarray[DTYPE_t, ndim=2] rotation

    sin = np.sin(rad)
    m_sin = -np.sin(rad)
    cos = np.cos(rad)
    
    rotation = np.array([[cos, m_sin], [sin, cos]], dtype=np.float64)

    return np.dot(rotation, vec.T).T


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

        list all_agents
        list all_agents2
        list first_agent
        list goal_count
        list data
        
        double agent_size
        double goal_vec
        double dynamic_percent
        double simple_avoid_vec
        double dynamic_avoid_vec
        double exe_time
        
        DTYPE_t[:] completion_time
        DTYPE_t[:] start_step
        DTYPE_t[:] goal_step
        DTYPE_t[:] agent_pos
        DTYPE_t[:] agent_vel
        DTYPE_t[:] visible_agent_pos
        DTYPE_t[:] visible_agent_vel
        DTYPE_t[:,:] dist
        DTYPE_t[:,:] vel
        DTYPE_t[:,:] goal_temp
        DTYPE_t[:,:] goal_pos
        DTYPE_t[:,:] first_pos
        DTYPE_t[:,:] start_pos
        
        DTYPE_t[:,:,:] agent_goals
        
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    def __cinit__(self, 
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
            cnp.ndarray[DTYPE_t, ndim=1] zero_arr
            cnp.ndarray[DTYPE_t, ndim=1] goal_arr
            cnp.ndarray[DTYPE_t, ndim=1] rot_v
            cnp.ndarray[DTYPE_t, ndim=1] all_vel
            cnp.ndarray[DTYPE_t, ndim=1] pos
            cnp.ndarray[DTYPE_t, ndim=1] row
            cnp.ndarray[DTYPE_t, ndim=2] all_pos
            double goal_x, goal_y
            
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
        
        self.all_agents = [] # 全エージェントの座標を記録
        self.all_agents2 = [] # ゴールの計算用
        
        self.agent_goals = np.zeros([num_agents, 8, 2], dtype=np.float64)
        self.first_pos = np.zeros([num_agents, 2], dtype=np.float64)
        
        #エージェント間の距離を記録するリスト
        self.dist = np.zeros([num_agents, num_agents], dtype=np.float64)
                
        # 動的回避を行うエージェントの数
        self.num_dynamic_agent = int(np.round(num_agents*dynamic_percent))
        
        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.agent_vel = np.zeros(2, dtype=np.float64)
        self.visible_agent_pos = np.zeros(2, dtype=np.float64)
        self.visible_agent_vel = np.zeros(2, dtype=np.float64)
        
        # エージェントの生成
        goal_arr = np.array([goal_vec, 0], dtype=np.float64)
        zero_arr = np.array([0, 0], dtype=np.float64)
        
        np.random.seed(random_seed)
        for n in range(num_agents):
            # グラフ領域の中からランダムに座標を決定
            pos = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            vel = np.random.uniform(-FIELD_SIZE, FIELD_SIZE, 2)
            rot_v = rotate_vec(goal_arr, calc_rad(vel, zero_arr))

            all_pos = np.zeros([num_steps+1, 2])
            all_vel = np.zeros(num_steps+1)
            
            # 1は動的回避で、0は単純回避
            avoidance = 1 if n < self.num_dynamic_agent else 0
            
            # 座標(0, 0)から座標velへのベクトルがエージェントの初期速度になる
            # self.all_agentsの1つの要素に1体のエージェントの位置と速度が記録
            # P(t) - V(t) = P(t-1)
            self.all_agents.append(
                {
                    'avoidance': avoidance, 
                    'p': pos, 
                    'v': rot_v,
                    'all_pos': all_pos,
                    'all_vel': all_vel
                }
            )
            
        # # 初期位置と初期速度をコピー
        self.all_agents2 = deepcopy(self.all_agents)
        self.first_agent = deepcopy(self.all_agents)
        
        # エージェントの初期位置を保存
        for i in range(self.num_agents):
            self.first_pos[i][0] = self.first_agent[i]['p'][0]
            self.first_pos[i][1] = self.first_agent[i]['p'][1]
            #self.first_pos.append(self.first_agent[i]['p'])
            
        # エージェントにゴールを8ずつ設定
        for i in range(self.num_agents):
            for j in range(8):
                #print('self.set_goals[0]', self.set_goals(self.all_agents2[i])[0])
                goal_x = self.set_goals(self.all_agents2[i]['p'], self.all_agents2[i]['v'])[0]
                goal_y = self.set_goals(self.all_agents2[i]['p'], self.all_agents2[i]['v'])[1]
                self.agent_goals[i][j][0] = goal_x
                self.agent_goals[i][j][1] = goal_y
        
                #self.agent_goals.append(goals)
        
        # ゴールした回数を記録するリスト
        self.goal_count = [0] * num_agents
        
        # はみ出た時用のゴール
        self.goal_temp = np.zeros([num_agents, 2])
        
        # 完了時間を測るための変数
        self.start_step = np.zeros([num_agents])
        self.goal_step = np.zeros([num_agents])
        self.start_pos = self.first_pos
        self.goal_pos = np.zeros([num_agents, 2])
         
        # 完了時間を記録するリスト
        self.completion_time = np.array([])
        
        self.data = []
        row = self.record_agent_information() # 全エージェントの位置と速度、接近を記録
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
            cnp.ndarray[DTYPE_t, ndim=1] pos
            double vel
        
        for i in range(self.num_agents):
            pos = self.all_agents[i]['p']
            vel = np.linalg.norm(self.all_agents[i]['v'])

            self.all_agents[i]['all_pos'][self.current_step][0] = pos[0] # x
            self.all_agents[i]['all_pos'][self.current_step][1] = pos[1] # y           
            self.all_agents[i]['all_vel'][self.current_step] = vel
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef list find_visible_agents (self, 
        cnp.ndarray[DTYPE_t, ndim=2] dist_all, 
        int num
    ):
        """
        エージェント番号numの視野に入った他のエージェントの番号をリストにして返す
        ex. self.find_visible_agents(dist_all, 5) -> [14, 20, 30]
        """
        cdef: # return
            list visible_agents
        cdef:
            # vector[int] near_agents
            # vector[int] visible_agents
            list near_agents
            int i
            double x
            double angle_difference
            double goal_angle
            cnp.ndarray[DTYPE_t, ndim=1] my_pos
            cnp.ndarray[DTYPE_t, ndim=1] other_pos
            cnp.ndarray[DTYPE_t, ndim=1] goal_pos
            
            
        # near_agentsは360度の視野に入ったエージェント
        # visible_agentsは視野を狭めた場合に視野に入ったエージェント
        near_agents = [i for i, x in enumerate(dist_all[num]) 
                        if x != -(0.2) and x < self.view]
        visible_agents = []
        
        # ゴールベクトルの角度を算出する
        my_pos = self.agent_goals[num][self.goal_count[num]]
        goal_pos = self.all_agents[num]['p']
        
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
                visible_agents.append(i)
        
        return visible_agents
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef cnp.ndarray[DTYPE_t, ndim=2] calc_distance_all_agents (self):
        cdef: # return
            cnp.ndarray[DTYPE_t, ndim=2] dist_all
        cdef:
            int i, j
            double norm
            cnp.ndarray[DTYPE_t, ndim=1] d
            
        dist_all = np.zeros([self.num_agents, self.num_agents], dtype=np.float64)
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                d = self.all_agents[i]['p'] - self.all_agents[j]['p']
                norm = np.linalg.norm(d)
                dist_all[i][j] = norm - 2*self.agent_size
        
        return dist_all
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef cnp.ndarray[DTYPE_t, ndim=1] simple_avoidance (self, 
        int num
    ):
        cdef: # return
            cnp.ndarray[DTYPE_t, ndim=1] avoid_vec
        cdef:
            list visible_agents
            int i
            double to_vec1
            cnp.ndarray[DTYPE_t, ndim=1] d1, d2, d3
            cnp.ndarray[DTYPE_t, ndim=2] dist_all
        
        dist_all = self.calc_distance_all_agents()
        visible_agents = self.find_visible_agents(dist_all, num)
        avoid_vec = np.zeros(2)   # 回避ベクトル
        
        if not visible_agents:
            return avoid_vec    
        
        ### the followings are simple vectors ###
        for i in visible_agents:
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d1 = self.all_agents[num]['p'] - self.all_agents[i]['p']
            to_vec1 = dist_all[num][i] + 2*self.agent_size
            d2 = d1 / to_vec1 # 大きさ1のベクトルにする
            d3 = d2 * self.simple_avoid_vec # 大きさを固定値にする
            
            if not np.isnan(d3).all():
                avoid_vec = avoid_vec + d3 # ベクトルの合成
            
        # # ベクトルの平均を出す
        # ave_avoid_vec = avoid_vec / len(visible_agents)
        
        # return ave_avoid_vec
        
        # ベクトルを平均しない
        return avoid_vec
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef cnp.ndarray[DTYPE_t, ndim=1] dynamic_avoidance (self, 
        int num
    ):
        """
        動的回避ベクトルの生成
        ex. self.dynamic_avoidance(num=15) -> array([-0.05, 0.2])
        """
        cdef:
            cnp.ndarray[DTYPE_t, ndim=1] avoid_vec
        cdef:
            cnp.ndarray[DTYPE_t, ndim=2] dist_all
            list visible_agents
            int i
            int t
            double dist_former
            cnp.ndarray[DTYPE_t, ndim=1] d
            double dist_latter
            double tcpa
            double dcpa
            double braking_index
            double a1, b1, c1, d1
        
        dist_all = self.calc_distance_all_agents()
        avoid_vec = np.zeros(2) # 回避ベクトル
        
        visible_agents = self.find_visible_agents(dist_all, num)   

        if not visible_agents:
            return avoid_vec    
                    
        ### the followings are dynamic vectors ###
        for i in visible_agents:
            # 視野の中心にいるエージェントの位置と速度
            self.agent_pos = self.all_agents[num]['p']
            self.agent_vel = self.all_agents[num]['v']
            # 視野に入ったエージェントの位置と速度
            self.visible_agent_pos = self.all_agents[i]['p']
            self.visible_agent_vel = self.all_agents[i]['v']
   
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
            a1 = -5.145
            b1 = 3.348
            c1 = 4.286
            d1 = -13.689
            
            # ブレーキ指標の算出
            braking_index = (1 / (1 + np.exp(-c1 - d1 * (tcpa/4000)))) * \
                            (1 / (1 + np.exp(-b1 - a1 * (dcpa/50))))
            
        
            # dは視界に入ったエージェントに対して反対方向のベクトル
            d = self.all_agents[num]['p'] - self.all_agents[i]['p']
            d = d / (dist_all[num][i] + 2 * self.agent_size) # ベクトルの大きさを1にする
            d = d * braking_index # ブレーキ指標の値を反映
            d = d * self.dynamic_avoid_vec # ベクトルの最大値を決定
            
            if not np.isnan(d).all():
                avoid_vec += d # ベクトルの合成
                
                    
        # # ベクトルの平均を出す
        # ave_avoid_vec = avoid_vec / len(visible_agents)

        # return ave_avoid_vec
        
        # ベクトルを平均しない
        return avoid_vec

    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef DTYPE_t[:] set_goals (self, 
        cnp.ndarray[DTYPE_t, ndim=1] agent_pos,
        cnp.ndarray[DTYPE_t, ndim=1] agent_vel
    ):
        cdef:
            cnp.ndarray[DTYPE_t, ndim=1] goal

        while True:
            
            # x座標がグラフ領域を超える
            if (agent_pos + agent_vel)[0] < -FIELD_SIZE:
                # 超えた時の座標をゴールとする
                goal = agent_pos + agent_vel
                
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
                agent_pos[0] = FIELD_SIZE + ((agent_pos + agent_vel)[0] + FIELD_SIZE)
                break
                        
            elif (agent_pos + agent_vel)[0] > FIELD_SIZE:
                goal = agent_pos + agent_vel
                
                # y座標も同時にサイズを超えるかつyが正
                if (goal[1] > FIELD_SIZE - 0.1):
                    goal[1] = goal[1] - 0.1
                    #print("調整入りました")
    
               # y座標も同時にサイズを超えるかつyが負
                elif (goal[1] < -FIELD_SIZE + 0.1):
                    goal[1] = goal[1] + 0.1
                    #print("調整入りました")
                    
                goal[0] = FIELD_SIZE
                agent_pos[0] = -FIELD_SIZE + ((agent_pos + agent_vel)[0] - FIELD_SIZE)
                break
                
                
            # y座標がグラフ領域を超える
            elif (agent_pos + agent_vel)[1] < -FIELD_SIZE:
                # 超えた時の座標をゴールとする
                goal = agent_pos + agent_vel
                goal[1] = -FIELD_SIZE
                
                agent_pos[1] = FIELD_SIZE + ((agent_pos + agent_vel)[1] + FIELD_SIZE)
                break
                                        
            elif (agent_pos + agent_vel)[1] > FIELD_SIZE:
                goal = agent_pos + agent_vel
                goal[1] = FIELD_SIZE
                
                agent_pos[1] = -FIELD_SIZE + ((agent_pos + agent_vel)[1] - FIELD_SIZE)
                break
                
            # エージェントを初期速度のまま動かす
            agent_pos += agent_vel
    
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
        cdef cnp.ndarray[DTYPE_t, ndim=1] arr_goal_pos
        cdef cnp.ndarray[DTYPE_t, ndim=1] goal
    
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
            if ((self.all_agents[i]['p'] + self.all_agents[i]['v'])[0] < -FIELD_SIZE):
                # ゴールに到着
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # 通常のゴールに到着
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append(
                                self.completion_time, completion_time
                            )
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
                        self.goal_temp[i][0] = self.agent_goals[i][self.goal_count[i]][0] + 2*FIELD_SIZE
                        self.goal_temp[i][1] = self.agent_goals[i][self.goal_count[i]][1]
                        
                    # はみ出た時用のゴールが設定されている
                    else:
                        # はみ出た時用のゴールを初期化
                        self.goal_temp[i][0] = 0
                        self.goal_temp[i][1] = 0
                
                # エージェントを反対の端へ移動
                self.all_agents[i]['p'][0] = FIELD_SIZE + (
                    (self.all_agents[i]['p'] + self.all_agents[i]['v'])[0] + FIELD_SIZE
                )
        
            
            # x座標が右端をこえる
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[0] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append(
                                self.completion_time, completion_time
                            )
                        self.goal_count[i] = self.goal_count[i] + 1
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
                        
                self.all_agents[i]['p'][0] = -FIELD_SIZE + (
                    (self.all_agents[i]['p'] + self.all_agents[i]['v'])[0] - FIELD_SIZE
                )

                
            # y座標が下をこえる
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[1] < -FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append( 
                                self.completion_time, completion_time
                            )
                        self.goal_count[i] = self.goal_count[i] + 1
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
                        
                self.all_agents[i]['p'][1] = FIELD_SIZE + (
                    (self.all_agents[i]['p']+self.all_agents[i]['v'])[1] + FIELD_SIZE
                )
                
            # y座標が上をこえる     
            elif ((self.all_agents[i]['p']+self.all_agents[i]['v'])[1] > FIELD_SIZE):
                
                # ゴール判定
                if (self.all_agents[i]['p'][0] > 
                   self.agent_goals[i][self.goal_count[i]][0] - 0.1 
                   and 
                   self.all_agents[i]['p'][0] < 
                   self.agent_goals[i][self.goal_count[i]][0] + 0.1 
                   and 
                   self.all_agents[i]['p'][1] > 
                   self.agent_goals[i][self.goal_count[i]][1] - 0.1 
                   and 
                   self.all_agents[i]['p'][1] < 
                   self.agent_goals[i][self.goal_count[i]][1] + 0.1):
                    
                    # ゴールが調整されているか確認
                    if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                        # 完了時間を算出
                        completion_time = self.calc_completion_time(i)
                        if not completion_time == None:
                            self.completion_time = np.append(
                                self.completion_time, completion_time
                            )
                        self.goal_count[i] = self.goal_count[i] + 1
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
                        

                self.all_agents[i]['p'][1] = -FIELD_SIZE + (
                    (self.all_agents[i]['p']+self.all_agents[i]['v'])[1] - FIELD_SIZE
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
            cnp.ndarray[DTYPE_t, ndim=1] goal_arr
            cnp.ndarray[DTYPE_t, ndim=1] row
        
        goal_arr = np.array([self.goal_vec, 0])
        for i in range(self.num_agents):
            # はみ出た時用のゴールが設定されていない
            # 通常のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            if (self.goal_temp[i][0] == 0 and self.goal_temp[i][1] == 0):
                self.all_agents[i]['v'] = rotate_vec(
                    goal_arr, 
                    calc_rad(self.agent_goals[i][self.goal_count[i]],
                             self.all_agents[i]['p'])
                )     

            # はみ出た時用のゴールが設定されている
            # はみ出た時用のゴールに向かうベクトルと、回避ベクトルを足したものが速度になる
            else:
                self.all_agents[i]['v'] = rotate_vec(
                    goal_arr, 
                    calc_rad(np.asarray(self.goal_temp[i]), self.all_agents[i]['p'])
                ) 
                
                
            # 回避ベクトルを足す
            if self.all_agents[i]['avoidance'] == 0: # 単純回避ベクトルを足す
                self.all_agents[i]['v'] += self.simple_avoidance(i)
                
            elif self.all_agents[i]['avoidance'] == 1: # 動的回避ベクトルを足す
                self.all_agents[i]['v'] += self.dynamic_avoidance(i)
                
        self.current_step += 1
        self.check_if_goaled()
        
        for i in range(self.num_agents):
            # 移動後の座標を確定      
            self.all_agents[i]['p'] += self.all_agents[i]['v']
                    
        self.update_parameters()    
        row = self.record_agent_information()
        self.data.append(row)
    
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cpdef void simulate (self):
        cdef:
            int t
            double start, fin
        
        self.disp_info()
        #start = time.perf_counter() # 実行時間を結果csvに記録する
        
        for t in range(self.num_steps):
        #for t in tqdm(range(self.num_steps)):
            self.move_agents()
            
        #fin = time.perf_counter()
        #self.exe_time = fin - start
        
        
    cdef cnp.ndarray[DTYPE_t, ndim=2] approach_detect (self, 
        double dist
    ): 
        """ 
        指定した距離より接近したエージェントの数を返す
        ex. self.approach_detect(dist=0.5) -> array([[0, 3],[1, 2],...[24, 1]])
        """
        cdef: # return
            approach_agent
        cdef:
            cnp.ndarray[DTYPE_t, ndim=2] dist_all
            list approach_agent_list
            int t
            int len_visible_agents
            double x
        
        dist_all = self.calc_distance_all_agents()
        approach_agent_list = []
        
        # それぞれのエージェントについて、distより接近したエージェントの数を記録
        for t in range(self.num_agents):
            visible_agents = [i for i, x in enumerate(dist_all[t]) 
                              if x != -(0.2) and x < dist]
            len_visible_agents = len(visible_agents)
            approach_agent_list.append([t, len_visible_agents])
        approach_agent = np.array(approach_agent_list, dtype=np.float64)
            
        return approach_agent
        
    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    cdef cnp.ndarray[DTYPE_t, ndim=1] record_agent_information (self):
        cdef: # return
            cnp.ndarray[DTYPE_t, ndim=1] row
        cdef:
            int i
            cnp.ndarray[DTYPE_t, ndim=2] collision_agent
            
        # 初期の位置と速度を記録
        row = np.concatenate([self.all_agents[0]['p'], 
                              self.all_agents[0]['v']])
        # rowにはある時刻の全エージェントの位置と速度が入る
        for i in range(1, self.num_agents):
            row = np.concatenate([row, self.all_agents[i]['p'], 
                                  self.all_agents[i]['v']])
            
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
    cdef list record_approaches (self,
        str approach_dist, 
        int step, 
        list sim_data
    ):
        """
        エージェント同士が接近した回数を、接近度合い別で出力する
        ex. self.record_approaches('collision', STEP=500, data) -> [0,0,3,...,12]
        """
        cdef: # return 
            list approach 
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
            
        approach = []
        for i in range(start, stop, 1):
            total = 0
            for j in range(step):
                # 一試行で何回エージェントに衝突したか
                total += sim_data[j+1][i]
            
            # 全エージェントの衝突した回数を記録
            approach.append(total)
        
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
            DTYPE_t completion_time
        cdef:
            DTYPE_t to_goal
            cnp.ndarray[DTYPE_t, ndim=1] goal_pos
        
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
                          
        # # 外れ値を除外
        # if (completion_time > 200):
        #     none = None
        #     return none
        
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
            cnp.ndarray[DTYPE_t, ndim=1] cross
            double distance
        
        # 初めのステップと終わりのステップを記録
        self.start_step[num] = self.goal_step[num] + 1
        self.goal_step[num] = step
        
        # ゴールする前に境界をはみ出ている場合
        if not (self.goal_temp[num][0] == 0 and self.goal_temp[num][1] == 0):
            # 左右の境界をはみ出た
            if (abs(self.all_agents[num]['p'][0]) > abs(self.all_agents[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agents[num]['p'][0] = -self.all_agents[num]['p'][0]
            # 上下の境界をはみ出た
            elif (abs(self.all_agents[num]['p'][0]) < abs(self.all_agents[num]['p'][1])):
                # はみ出る前に戻してあげる
                self.all_agents[num]['p'][1] = -self.all_agents[num]['p'][1]
            
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
        d = -(c * self.all_agents[num]['p'][0]) + self.all_agents[num]['p'][1]

        # 2つの直線の交点を算出
        cross_x = (b - d) / (-(a - c))
        cross_y = a * cross_x + b
        cross = np.array([cross_x, cross_y])
        
        # スタートから交点までの距離を計算
        distance = np.linalg.norm(self.start_pos[num] - cross)
        
        # 完了時間を計算(ゴールまでのステップ/ゴールまでの距離)
        completion_time = (self.goal_step[num] - self.start_step[num] + 1) / distance

        # 外れ値の除外
        # if (completion_time > 200 or completion_time < 10):
        #     #print("消しました")
        #     none = None
        #     return none
        
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
            if not last_completion_time == None:
                self.completion_time = np.append(
                    self.completion_time, last_completion_time
                )
       
        # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
        collision = self.record_approaches('collision', self.num_steps, self.data)
        half =  self.record_approaches('half', self.num_steps, self.data)
        quarter =  self.record_approaches('quarter', self.num_steps, self.data)
        one_eighth =  self.record_approaches('one_eigth', self.num_steps, self.data)
        
        # 各指標の平均を計算
        collision_mean = np.mean(collision)
        half_mean = np.mean(half)
        quarter_mean = np.mean(quarter)
        one_eighth_mean = np.mean(one_eighth)
        completion_time_mean = np.mean(self.completion_time)

        # 結果のデータを保存
        dict_result = {'time': completion_time_mean,
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
                       'exe_time_min': np.round(self.exe_time/60, 3)}
        
        # df_result = pd.DataFrame(dict_result, index=[f'seed_{self.random_seed}'])
        # self.returned_results = True
        
        # return df_result
    
        return dict_result