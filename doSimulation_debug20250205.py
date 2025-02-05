""" 
シミュレーションを実際に実行する
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm, gc, time
from datetime import datetime, timedelta
import classSimulation_debug20250205 as cs

# %% シミュレーション
# 一度にO.num_steps数simaulateメソッドを使用するシミュレーションを、TRIALの回数行う
NUM_OF_TRIAL = 4 # 試行回数

row_label = [f'seed_{i}' for i in range(NUM_OF_TRIAL)]
column_label = ['time', 'half', 'quarter', 'one_eighth', 
                'collision', 'agent', 'viewing_angle', 
                'O.num_steps', 'dynamic_percent', 'simple_avoid_vec']
df_result = pd.DataFrame(0, index=row_label, columns=column_label)

agent = 25
simple_avoid_vec_px = 3 # px
simple_avoid_vec = simple_avoid_vec_px / 50

print('\nsimple_avoid_vec:', simple_avoid_vec_px, 'px')
print('num of agents:', agent)
dyn_prop = float(input('\nProportions of dynamic agents(0-1): '))
print('dyn_prop:', dyn_prop)

t_now = datetime.now()
print(f'\nシミュレーション開始時刻は {t_now.strftime("%H:%M")} です。\n')

for num in range(NUM_OF_TRIAL):
    print(f'Start ({num+1}/{NUM_OF_TRIAL})')
    O = cs.Simulation(interval=100,
                      num_steps=50,
                      agent_size=0.1, 
                      agent=agent, 
                      view=1, 
                      viewing_angle=360, 
                      goal_vec=0.06,  
                      dynamic_percent=dyn_prop,
                      simple_avoid_vec=simple_avoid_vec, 
                      dynamic_avoid_vec=0.06,
                      random_seed=num)
    print('random seed:', O.random_seed)

    ##### シミュレーション (O.num_steps数だけ繰り返す) #####
    start_time = time.time()
    O.simulate()
    
    end_time = time.time()
    passed_time = end_time - start_time
    
    print(f'Finish ({num+1}/{NUM_OF_TRIAL})')
    print(f'試行{num+1}の経過時間は {passed_time:.0f}秒({passed_time/60:.0f}分) です。\n')
    expected_passsing_time_sec = passed_time*NUM_OF_TRIAL
    expected_passsing_time_min = expected_passsing_time_sec / 60
    expected_passsing_time_hr = expected_passsing_time_min / 60
    finish_hr = int(expected_passsing_time_hr)
    finish_min = int((expected_passsing_time_hr - finish_hr) * 60)
    expected_end_time = t_now + timedelta(seconds=expected_passsing_time_sec)
    print(f'シミュレーション開始時刻は {t_now.strftime("%H:%M")} です。')
    print(f'予測されるシミュレーションの実行時間は 約{expected_passsing_time_min:.0f}分', end='') 
    print(f'({finish_hr}時間{finish_min}分) です。')
    print(f'現在時刻は {datetime.now().strftime("%H:%M")} です。')
    print(f'終了時刻の目安は {expected_end_time.strftime("%H:%M")} です。\n')
    print('Settings of simulation')
    print('simple_avoid_vec:', simple_avoid_vec_px, 'px')
    print('dyn_prop:', dyn_prop)
    print('num of agents:', agent)
    print('--------------------------------------------------------------------\n')
    ##### シミュレーション終了 ######    
        
    
    ##### シミュレーションで得たデータを記録 #####
    
    # 最後の座標から完了時間を算出
    for i in range(O.agent):
        last_completion_time = O.calc_remained_completion_time(i, O.num_steps)
        if not last_completion_time == None:
            O.completion_time.append(last_completion_time)
   
    # 衝突した数、視野の半分、視野の四分の一、視野の八分の一に接近した回数
    # must fix the shape of data array
    collision = O.record_approaches('collision', O.num_steps, O.data)
    half =  O.record_approaches('half', O.num_steps, O.data)
    quarter =  O.record_approaches('quarter', O.num_steps, O.data)
    one_eighth =  O.record_approaches('one_eigth', O.num_steps, O.data)
    
    # 各指標の平均を計算
    collision_mean = np.mean(collision)
    half_mean = np.mean(half)
    quarter_mean = np.mean(quarter)
    one_eighth_mean = np.mean(one_eighth)
    completion_time_mean = np.mean(O.completion_time)

    # 各試行の結果のデータを保存
    values = [completion_time_mean, half_mean, quarter_mean, 
              one_eighth_mean, collision_mean, O.agent, O.viewing_angle, 
              O.num_steps, O.dynamic_percent, O.simple_avoid_vec]
    df_result.loc[f'seed_{num}'] = values
    
    ##### データの記録終了 #####
print(f'シミュレーション終了時刻は {datetime.now().strftime("%H:%M")} です。\n')
print(f'dyn_prop {dyn_prop}終了')
    
    
##### 全TRIALの結果の記録 #####
# 値をまとめたcsvファイルの作成
backup_result = df_result.copy()
# df_result.to_csv(
#     f'simulation_results/agt{O.agent}_avoidvec{int(O.simple_avoid_vec*500)}px_dynper0{int(O.dynamic_percent*10)}.csv',
#     mode='x'
# )

# %% make animations
# pos_data = O.all_pos
# fs.animte_agent_movements(pos_data, O, save_as='simulation.mp4')

