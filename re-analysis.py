# control condition (no instructions for the moving behavior)
import os
os.chdir("C:/Users/ootmo/OneDrive/Desktop/修論/共同P_論文/松林さん/04_RawData")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('2_omoiyari.csv')
df

def show_all(obj):
    for i in obj: print(i)

show_all(df.columns)

df_5 = df.query('type.str.contains("05")')
df_10 = df.query('type.str.contains("10")')
df_20 = df.query('type.str.contains("20")')

tri_num_5 = list(set(df_5['trial']))
tri_num_10 = list(set(df_10['trial']))
tri_num_20 = list(set(df_20['trial']))

df_5_tri = df_5.query("trial == @tri_num_5[0]")
df_10_tri = df_10.query("trial == @tri_num_10[0]")
df_20_tri = df_20.query("trial == @tri_num_20[0]")

df_5_tri.myNextX.plot(); df_5_tri.myNextY.plot(); plt.show()
df_10_tri.myNextX.plot(); df_10_tri.myNextY.plot(); plt.show()
df_20_tri.myNextX.plot(); df_20_tri.myNextY.plot(); plt.show()

for i in range(len(tri_num_5)):
    df_5_tri = df_5.query("trial == @tri_num_5[@i]")
    df_5_tri.myNextX.plot(); df_5_tri.myNextY.plot() 
    plt.show()
    
for i in range(len(tri_num_10)):
    df_10_tri = df_10.query("trial == @tri_num_10[@i]")
    df_10_tri.myNextX.plot(); df_10_tri.myNextY.plot() 
    plt.show()
    
for i in range(len(tri_num_20)):
    df_20_tri = df_20.query("trial == @tri_num_20[@i]")
    df_20_tri.myNextX.plot(); df_20_tri.myNextY.plot() 
    plt.show()
    
for x, y in zip(df_5_tri.myNextX, df_5_tri.myNextY):
    plt.scatter(x, y)
plt.show()

for x, y in zip(df_10_tri.myNextX, df_10_tri.myNextY):
    plt.scatter(x, y)
plt.show()

for x, y in zip(df_20_tri.myNextX, df_20_tri.myNextY):
    plt.scatter(x, y)
plt.show()



