""" 
シミュレーションを実際に実行する
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm, gc, time
from datetime import datetime, timedelta
import classSimulation as cs

# %% シミュレーション
# 一度にsim.num_steps数simaulateメソッドを使用するシミュレーションを、TRIALの回数行う
NUM_OF_TRIAL = 20 # 試行回数
NUM_STEPS = 500
NUM_AGENTS = 25

prepared_data = cs.PreparedData('prepared_data_agent25.npz', remove_outliers=2)

df_result = pd.DataFrame()

simple_avoid_vec_px = 3 # px
simple_avoid_vec = simple_avoid_vec_px / 50

w = cs.AwarenessWeight()
t_now = datetime.now()
print(f'\nシミュレーション開始時刻は {t_now.strftime("%H:%M")} です。\n')

for num in range(NUM_OF_TRIAL):
    print(f'Start ({num+1}/{NUM_OF_TRIAL})')
    sim = cs.Simulation(interval=100,
                        num_steps=NUM_STEPS,
                        num_agents=NUM_AGENTS, 
                        dynamic_percent=1,
                        simple_avoid_vec=simple_avoid_vec, 
                        prepared_data=prepared_data,
                        awareness=0.9,
                        awareness_weight=w,
                        random_seed=num)
    
    print('random seed:', sim.random_seed)
    print('with Awareness:', sim.awareness)

    ##### シミュレーション (sim.num_steps数だけ繰り返す) #####
    sim.simulate()
    passed_time = sim.exe_time
    
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
    print('--------------------------------------------------------------------\n')
    ##### シミュレーション終了 ######    
        
    # 結果の記録(trial毎)
    df_tmp = sim.return_results_as_df()
    df_result = pd.concat([df_result, df_tmp])

print(f'シミュレーション終了時刻は {datetime.now().strftime("%H:%M")} です。\n')
    
##### 全TRIALの結果の記録 #####
# 値をまとめたcsvファイルの作成
#backup_result = df_result.copy()
#file = f'simulation_results/agt{sim.num_agents}_avoidvec{int(sim.simple_avoid_vec*500)}px_dynper0{int(sim.dynamic_percent*10)}.csv'
file = f'simulation_results/agt{sim.num_agents}_dynmic_awareness.csv' 
try:
    df_result.to_csv(file, mode='x')
except FileExistsError:
    overwrite = input('\nFile already exists. Overwrite? (y/n): ')
    if overwrite == 'y':
        df_result.to_csv(file)
    
# %% make animations
#sim.animate_agent_movements(save_as='simulation_awareness25.mp4')
