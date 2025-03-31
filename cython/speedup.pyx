# cython: language_level = 3
import cython
from cython import boundscheck, wraparound
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float64_t DTYPE_t

from libcpp import vector

cdef DTYPE_t gas = 100.0
cdef int color = 1

cdef DTYPE_t[:] arr1 = np.array([1., 2.])

cdef class Car:
    cdef DTYPE_t[:] arr2
    
    def __init__(self):
        self.gas = gas
        self.color = color
        arr2 = np.array([1., 2.])
        self.arr2 = arr2
        
    cpdef void disp(self):
        print(self.gas)
        print(self.color)

@cython.cfunc
@boundscheck(False)
@wraparound(False)
cpdef DTYPE_t slow_calc_rad (DTYPE_t[:] pos2, 
                        DTYPE_t[:] pos1): 
    """
    pos1からpos2のベクトルの角度を返す
    ex. calc_rad(pos2=np.array([1.5, 2.5]), pos1=np.array([3.0, 1.0])) -> 2.4
    """    
    cdef DTYPE_t retval
    
    retval = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]).astype(np.float64)
    
    return retval

@cython.cfunc
@boundscheck(False)
@wraparound(False)
cpdef DTYPE_t[:] slow_rotate_vec (DTYPE_t[:] vec,
                            DTYPE_t rad):
    """
    ベクトルをradだけ回転させる (回転行列)
    ex. rotate_vec(vec=np.array([3.0, 5.0]), rad=1.2) -> array([-3.6, 4.6])
    """
    cdef:
        DTYPE_t sin, m_sin, cos
        DTYPE_t[:,:] rotation

    sin = np.sin(rad).astype(np.float64)
    m_sin = -np.sin(rad).astype(np.float64)
    cos = np.cos(rad).astype(np.float64)
    
    rotation = np.array([[cos, m_sin], [sin, cos]], dtype=np.float64)

    retval = np.dot(rotation, vec.T).T.astype(np.float64)

    return retval

from cython import boundscheck, wraparound
from libc.math cimport sin, cos, atan2, fmin, fmax

@cython.cfunc
@boundscheck(False)
@wraparound(False)
cpdef DTYPE_t calc_rad(double[:] pos2, double[:] pos1):
    """
    Return the angle between pos1 and pos2 vectors in radians
    """
    cdef DTYPE_t retval
    
    # Using libc's atan2 for better performance
    retval = atan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
    
    return retval

@cython.cfunc
@boundscheck(False)
@wraparound(False)
cpdef DTYPE_t[:] rotate_vec(DTYPE_t[:] vec, DTYPE_t rad):
    """
    Rotate vector by rad radians using a rotation matrix
    """
    cdef DTYPE_t sin_val, cos_val
    cdef DTYPE_t[:] retval

    # Use libc math functions for sin and cos
    sin_val = sin(rad)
    cos_val = cos(rad)
    
    # Perform matrix multiplication manually (since it's a 2x2 matrix)
    retval = vec.copy()  # Create a copy to store the rotated result
    retval[0] = cos_val * vec[0] - sin_val * vec[1]  # Rotation for X-axis
    retval[1] = sin_val * vec[0] + cos_val * vec[1]  # Rotation for Y-axis

    return retval

cdef double sum_all(double[:] arr):
    cdef double total
    cdef double i
    total = 0
    for i in arr:
        total += i
    return total
     
from cython.parallel import prange

cdef double find_min(double[:] arr):
    cdef int i
    cdef double total
    for i in range(100):
        i += i * 3
    total = sum(arr)
    return min(arr), max(arr)
    
cdef double find_min_np(double[:] arr):
    cdef Py_ssize_t i
    cdef double total, total2, total3
    for i in prange(100, nogil=True):
        i += i * 3
    total = np.sum(arr)
    total2 = sum_all(arr)
    total3 = sum(arr)
    return np.min(arr), np.max(arr)

cdef double find_min_c(double arr, double arr2):
    cdef Py_ssize_t i
    for i in range(100):
        i += i * 3
    return fmin(arr, arr2)
    
import numpy as np
import timeit


array_1 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 100, size=(3000, 2000)).astype(np.intc)
a = 4
b = 3
c = 9

DTYPE = np.intc   # numpy.intc ---- int. Otherwise,they are implicitly typed as Python objects

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.

def compute_cy_i(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):
     
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, :] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result

print(compute_cy_i(array_1, array_2, a, b, c))

compute_cy_i_time = timeit.timeit(lambda: compute_cy_i(array_1, array_2, a, b, c), number=10)/10

print("compute_cy_i execution time:", compute_cy_i_time)