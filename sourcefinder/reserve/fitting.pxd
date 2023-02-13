# cython: language_level=3

import numpy as np
cimport numpy as np
import cython

@cython.locals(peak = cython.float, ratio = cython.float, total = cython.float, x = np.ndarray,
               y = np.ndarray, xbar = cython.float, ybar = cython.float, xxbar = cython.float,
               yybar = cython.float, xybar = cython.float, working1 = cython.float, working2 = cython.float,
               semiminor = cython.float, semimajor = cython.float, semimajor_tmp = cython.float, semiminor_tmp =
               cython.float, theta = cython.float)
cpdef dict moments(np.ndarray[np.npy_float64, ndim = 2] data, float fudge_max_pix_factor, float beamsize,
                   float threshold)