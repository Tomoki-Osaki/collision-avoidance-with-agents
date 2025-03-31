# cython: language_level=3
# distutils: language=c++

import numpy as np
import cython

cimport numpy as cnp
cnp.import_array()

from libcpp.random cimport uniform_real_distribution
from libcpp.random cimport random_device, mt19937
from libcpp.vector cimport vector
from libc.math cimport sqrt as c_sqrt

ctypedef vector[vector[double]] vector2d

cdef double c_abs(double x):
    if x >= 0:
        return x
    else:
        return -x

cdef double c_norm (double dx,
                     double dy):
    return c_sqrt(dx * dx + dy * dy) 

cdef vector[double] generate_random_numbers(int seed):
    cdef mt19937 gen = mt19937(seed) # メルセンヌ・ツイスター乱数生成器
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](-5.0, 5.0)  # [-5.0, 5.0] の範囲
    
    cdef vector[double] numbers = vector[double](2)
    
    numbers[0] = dist(gen)  
    numbers[1] = dist(gen)
    c_norm(numbers[0], numbers[1])
    c_abs(numbers[0])
    abs(numbers[0])
    np.abs(numbers[0])
    
    return numbers


cdef double[:] generate_agents(int seed):
    cdef double[:] agent
    np.random.seed(seed)
    agent = np.random.uniform(-5., 5., 2)
    np.linalg.norm(agent)
    c_norm(agent[0], agent[1])
    
    return agent

# 2次元配列を作成する関数
cdef vector2d create_2d_array(int rows, int cols, double initial_value):
    cdef vector2d array = vector[vector[double]](rows)  # 行数を設定
    
    for i in range(rows):
        array[i] = vector[double](cols)  # 列数を設定
        for j in range(cols):
            array[i][j] = initial_value  # 初期値を設定
    
    return array