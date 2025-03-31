import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef int AGENT_SIZE = 1

ctypedef np.float64_t DTYPE_t 

cdef DTYPE_t[:, :] calc_distance_all_agents(
    int num_agents,
    DTYPE_t[:, :, :] all_agents
):
    cdef:
        DTYPE_t[:, :] dist_all = np.empty((num_agents, num_agents), dtype=np.float64)
        int i, j
        DTYPE_t dx, dy, dist

    for i in range(num_agents):
        for j in range(num_agents):
            dx = all_agents[i, 1, 0] - all_agents[j, 1, 0]
            dy = all_agents[i, 1, 1] - all_agents[j, 1, 1]
            dist = sqrt(dx * dx + dy * dy) - 2 * AGENT_SIZE
            dist_all[i, j] = dist

    return dist_all
