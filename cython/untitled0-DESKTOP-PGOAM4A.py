def wd():
    import os
    try: os.chdir('C:/Users/ootmo/Downloads/cython')
    except FileNotFoundError: pass
wd()

import numpy as np
import sim_without_awm as sw
import classSimulation as cs
import funcSimulation as fs
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import makes_all_buffer as mb

mbt = mb.Simulation()
mbt.simulate()
mbt.return_results_as_df()

t = cs.Simulation(num_agents=25, num_steps=500)
t.simulate()
print(t.return_results_as_df())

swt = sw.Simulation(num_agents=25, num_steps=500)
#fs.plot_positions(swt, swt.current_step)

swt.simulate()
res = swt.return_results_as_df()
print(res)
#res.to_csv('for_ref.csv')
    
# %%
fs.plot_positions(swt, swt.current_step)
