"""
Author: Jessica O'Dell
Editor: Logan Blue
Date: 09/13/2019, 2/15/2020
This function does the translation of area curve to reflection 
coefficient series and vice versa. 
"""

#pylint: disable=bad-whitespace,invalid-name
#pylint: disable=trailing-whitespace

import numpy as np


R_LIMIT = 0.99
DENOM_EPS = 1e-6

def areaSolver(r_N, A_0, r_limit=R_LIMIT, denom_eps=DENOM_EPS):
    """
    This function converts the series of reflection coefficients into the 
    estimated cross sectional area for a given tube series based on a given 
    starting cross sectional area. 
    """
    if not np.isfinite(A_0) or A_0 <= 0:
        return []
    A_0 = float(A_0)
    A_list = []
    for pos in range(0, len(r_N)):
        r_k = float(r_N[pos])
        if not np.isfinite(r_k):
            return []
        r_k = float(np.clip(r_k, -r_limit, r_limit))
        denom = 1.0 - r_k
        if abs(denom) < denom_eps:
            denom = denom_eps if denom >= 0 else -denom_eps
        next_A = (A_0 * (r_k + 1.0)) / denom
        if not np.isfinite(next_A) or next_A <= 0:
            return []
        next_A = float(next_A)
        A_list.append(next_A)
        A_0 = next_A
    return A_list

def reflectionSolver(a_n, denom_eps=DENOM_EPS, r_limit=R_LIMIT):
    """
    This fucntion converts a series of cross sectional area into the corresponding
    reflection coefficient series. 
    """
    r_series = []
    for i in range(0, len(a_n) - 1):
        top = a_n[i+1] - a_n[i]
        bottom = a_n[i+1] + a_n[i]
        if abs(bottom) < denom_eps:
            bottom = denom_eps if bottom >= 0 else -denom_eps
        r_val = top / bottom
        if not np.isfinite(r_val):
            r_val = 0.0
        r_series.append(float(np.clip(r_val, -r_limit, r_limit)))

    return r_series
