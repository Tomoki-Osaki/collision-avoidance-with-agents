""" 
2025/01/07 
シミュレーションを実際に実行する
"""
# %%
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import tqdm, gc, time
from datetime import datetime, timedelta
import classSimulation as cs

# %% シミュレーション
# 各シミュレーションの結果を保存する変数
row_label = []
values = []
ims = [] # 図のデータをまとめるもの、これを流すことでアニメーションになる

# 一度にSTEP数simaulateメソッドを使用するシミュレーションを、TRIALの回数行う
NUM_OF_TRIAL = 20 # 試行回数
STEP = 500 # 1回の試行(TRIAL)で動かすステップの回数
dynper = 0.9
t_now = datetime.now()
print(f'シミュレーション開始時刻は {t_now.strftime("%H:%M")} です。\n')

for num in range(NUM_OF_TRIAL):
    np.random.seed(num)
    O = cs.Simulation(interval=100, 
                      agent_size=0.1, agent=50, 
                      view=1, viewing_angle=360, 
                      goal_vec=0.06,  
                      dynamic_percent=dynper,
                      simple_avoid_vec=0.06, 
                      dynamic_avoid_vec=0.06)

    data = []
    column_label = []
    index_label = ['initial_value']

    # 出力データの列のラベルを作成
    for i in range(O.agent):
        column_label.append('agent ' + str(i) + ' pos_x')
        column_label.append('agent ' + str(i) + ' pos_y')
        column_label.append('agent ' + str(i) + ' vel_x')
        column_label.append('agent ' + str(i) + ' vel_y')
        column_label.append('agent ' + str(i) + ' collision') # 衝突(距離0以下)
        column_label.append('agent ' + str(i) + ' half') # 視野の半分の距離まで接近
        column_label.append('agent ' + str(i) + ' quarter') # 視野の4分の1の距離まで接近
        column_label.append('agent ' + str(i) + ' one_eighth') # 視野の8分の1の距離まで接近

    row = O.record_agent_information() # 全エージェントの位置と速度、接近を記録
    data.append(row) # ある時刻でのエージェントの情報が記録されたrowが集まってdataとなる

    # # 最初に表示する図の作成
    # plot_data = O.show_image()
    # ims.append(plot_data)

    ##### シミュレーション (STEP数だけ繰り返す) #####
    start_time = time.time()
    for t in tqdm.tqdm(range(STEP)):
        O.simulate(now_step=t+1)
        index_label.append(t + 1)

        # シミュレーションごとに値を記録
        row = O.record_agent_information()
        data.append(row)

        ## 図を作成
        #plot_data = O.show_image()
        #ims.append(plot_data)
        
    end_time = time.time()
    passed_time = end_time - start_time
    
    print(f'{NUM_OF_TRIAL}回中{num+1}回目の試行が終了しました。')
    print(f'経過時間は{passed_time:.0f}秒です。\n')
    if num == 0:
        expected_passsing_time = passed_time*NUM_OF_TRIAL
        expected_end_time = t_now + timedelta(seconds=expected_passsing_time)
        print(f'シミュレーションの実行時間は 約{expected_passsing_time/60:.0f}分 です。')
        print(f'終了時刻の目安は {expected_end_time.strftime("%H:%M")} です。\n')
    ##### シミュレーション終了 ######    
        
    
    ##### シミュレーションで得たデータを記録 #####
    # csvとしてステップごとに位置、速度、接近した回数を記録
    """
    df = pd.DataFrame(data, columns=column_label, index=index_label)
    df.to_csv('to_csv_out_' + str(num) + '.csv')
    """
    
    # 最後の座標から完了時間を算出
    for i in range(O.agent):
        last_completion_time = O.calc_last_completion_time(i, STEP)
        if not last_completion_time == None:
            O.completion_time.append(last_completion_time)

    # 完了時間をまとめたファイルを作成
    column_label = ["completion_time"]
    index_label = []
    
    for i in range(len(O.completion_time)):
        index_label.append(i + 1)
    
    """
    df = pd.DataFrame(O.completion_time, columns=column_label, index=index_label)
    df.to_csv('to_csv_out_completion_time_' + str(num) + '.csv')
    """
    
    # 加速度はx, y速度の差からなるベクトルの大きさ
    accel = []
    agents_accels = []
    for i in range(2, 4*O.agent+2, 4):
        agent_accels = []
        for j in range(STEP - 1):
            # x軸方向の速度の差分
            x_accel = abs((data[j+1][i] - data[j+2][i]) * 50)
            # y軸方向の速度の差分
            y_accel = abs((data[j+1][i+1] - data[j+2][i+1]) * 50)
            # x, y速度の差からなるベクトルの大きさ
            temp = np.sqrt(x_accel**2 + y_accel**2)
            agent_accels.append(temp)
            
        agents_accels.append(agent_accels) # 全エージェントの加速度を記録
        accel.append(np.sum(agent_accels)) # 全エージェントの加速度の総和を記録

        
    # 加速度をまとめたファイル
    column_label = []
    for i in range(O.agent):
        column_label.append('agent ' + str(i) + ' accel')

    index_label = []
    for i in range(len(agents_accels[0])):
        index_label.append(i + 1)
    
    df = pd.DataFrame(agents_accels, columns=index_label, index=column_label)
    df = df.T
    # df.to_csv('to_csv_out_accel_' + str(num) + '.csv')
   
    # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
    collision = O.record_approaches('collision', STEP, data)
    half =  O.record_approaches('half', STEP, data)
    quarter =  O.record_approaches('quarter', STEP, data)
    one_eighth =  O.record_approaches('one_eigth', STEP, data)
    
    # 各指標の平均を計算
    accel_mean = np.mean(accel)
    collision_mean = np.mean(collision)
    half_mean = np.mean(half)
    quarter_mean = np.mean(quarter)
    one_eighth_mean = np.mean(one_eighth)
    completion_time_mean = np.mean(O.completion_time)

    # 各試行の結果のデータを保存
    row_label.append('seed_' + str(num))
    values.append([accel_mean, completion_time_mean, half_mean, quarter_mean, 
                   one_eighth_mean, collision_mean, O.agent, O.viewing_angle, 
                   STEP, O.dynamic_percent])
    ##### データの記録終了 #####
print(f'シミュレーション終了時刻は {datetime.now().strftime("%H:%M")} です。\n')
print(f'{dynper}終了')    
    
##### 全TRIALの結果の記録 #####
# 値をまとめたcsvファイルの作成
column_label = ['accel', 'time', 'half', 'quarter', 'one_eighth', 'collision', 
                'agent', 'viewing_angle', 'step', 'dynamic_percent']
                      
df_result = pd.DataFrame(values, columns=column_label, index=row_label)
# 保存する場所は自由に決めてください
# df_result.to_csv(f'dynper{int(100*O.dynamic_percent)}_ang{O.viewing_angle}_agt{O.agent}_stp{STEP}.csv')
backup_result = df_result.copy()
df_result.to_csv(f'{dynper}_agt50.csv')

# result = pd.read_csv('agt25_avoidVec3px.csv').reset_index(drop=True)\
# result = df_result.iloc[:,2:].groupby('dynamic_percent').mean()

# dynamic_percent0-0.2, agent50のシミュレーション結果は、20250109_agt50_dynper00-02.csv

# %% make animations 
# ani = animation.ArtistAnimation(fig, ims, interval=O.interval, repeat=False)
# ani.save(f'ani_dynper{int(100*O.dynamic_percent)}_ang{O.viewing_angle}_agt{O.agent}_stp{STEP}.mp4')

fig, ax = plt.subplots(figsize=(8,8))
def update(frame):
    ax.cla()
    for i in range(O.agent):
        if i < O.num_dynamic_agent: 
            color = 'blue'
        else:
            color = 'red'
        ax.scatter(x=frame[0][i], y=frame[1][i], s=40, marker="o", c=color)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid()

anim = FuncAnimation(fig, update, frames=ims, interval=100)
anim.save('simualtion.mp4')

