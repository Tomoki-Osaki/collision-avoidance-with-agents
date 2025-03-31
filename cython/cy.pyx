import numpy as np
cimport cython

DTYPE = np.intc

cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
def compute(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):
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
            result[x, y] = tmp + c
            
    return result
    

cpdef int myfunction(int x, int y=2):
    cdef int a = x - y
    return a + x * y


cdef double _helper(double a):
    return a + 1
    

cdef class A:
    cdef:
        double _scale
    cdef public:
        int x, y, a, b
    cdef readonly:
        float read_only
    
    def __init__(self, int a=3, int b=0):
        self.a = a
        self.b = b
        self._scale = 2.0
        self.read_only = 1.0
        
    cpdef void disp(self):
        print('a', self.a)
        print('b', self.b)
        
    cpdef double foo(self, double x):
        return (x + _helper(1.0)) * self._scale
         

cdef class Particle:
    cdef readonly:
        double mass, position, velocity
    
    def __init__(self, m, p, v):
        self.mass = m
        self.position = p
        self.velocity = v
    
    cpdef double get_momentum(self):
        return self.mass * self.velocity
