import speedup as su

import time
import numpy as np

arr1 = np.array([1, 2], dtype='float64')
arr2 = np.array([3, 4], dtype='float64')

start = time.perf_counter()
for i in range(10000):
    su.calc_rad(arr1, arr2)
exe_fast = time.perf_counter() - start
print('exe:', exe_fast)

start = time.perf_counter()
for i in range(10000):
    su.slow_calc_rad(arr1, arr2)
exe_slow = time.perf_counter() - start
print('exe:', exe_slow)

print(exe_slow / exe_fast)


def just_np(pos2, pos1):
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]).astype(np.float64)
    

start = time.perf_counter()
for i in range(10000):
    just_np(arr1, arr2)
exe_np = time.perf_counter() - start
print('exe:', exe_np)

print('exe fast', exe_fast)
print('exe slow', exe_slow)
print('exe np  ', exe_np)
