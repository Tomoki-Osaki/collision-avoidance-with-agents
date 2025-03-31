def wd():
    import os
    try: os.chdir('Downloads/cython')
    except FileNotFoundError: pass
wd()

# %%
import numpy as np
import pandas as pd
import cyclass
import classSimulation as cs
import funcSimulation as fs
import time

import makes_all_buffer as mab
import all_double as db

# %%    
agent = 25
steps = 100

start = time.perf_counter()
t = cs.Simulation(num_agents=agent, num_steps=steps, dynamic_percent=0.5, random_seed=0)
py_exe = time.perf_counter() - start
print('py', py_exe)

start = time.perf_counter()
buft = mab.Simulation(num_agents=agent, num_steps=steps, dynamic_percent=0.5, random_seed=0)
buft_exe = time.perf_counter() - start
print('bu', buft_exe)

# start = time.perf_counter()
# dbt = db.Simulation(num_agents=agent, num_steps=steps, dynamic_percent=0.5, random_seed=0)
# db_exe = time.perf_counter() - start
# print('db', db_exe)

# %%
start = time.perf_counter()
buft.simulate()
buft_exe = time.perf_counter() - start
print('\nbu', buft_exe)

start = time.perf_counter()
t.simulate()
py_exe = time.perf_counter() - start
print('\npy', py_exe)

# start = time.perf_counter()
# dbt.simulate()
# db_exe = time.perf_counter() - start
# print('\ndb', db_exe)

# %%
py_res = t.return_results_as_df()
print('\npy_res', py_res)

buft_res = buft.return_results_as_df()
print('buft_res', buft_res)

# db_res = dbt.return_results_as_df()

# %%
agent_goals = t.agent_goals
arr = np.array(agent_goals)
