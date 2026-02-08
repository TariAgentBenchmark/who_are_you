"""
Author: Logan Blue
Date: August 16, 2019

This script implements a gradient descent search for an equation that
represents the area between two curves.
"""

import sys
import copy
import math
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
from numba import cuda, float64, complex128, int64
import new_gpu as gpu

import pdb
import matplotlib.pyplot as plt
import time

#Pylint flags
#pylint: disable=trailing-whitespace,consider-using-enumerate

def _df_gpu(truth, freq, FS, r_series_guess, max_iterations):
    """Function to calculate the slope of the area function evaluated at 
    a specific coeff value. This slope is calculated using an iterative method
    to prevent needing any knowledge about the underlying equations defining the 
    area function. These implementation of this function is done on a cuda gpu
    to hopefully give a large speed up. 
    baseband    thread      blocks
    4000        64          32
    8575        320         8
    """
    #fixed values
    r_size = int(len(r_series_guess))
    omega_size = int(len(freq))
    if r_size == 0 or omega_size == 0:
        return np.asarray(r_series_guess, dtype=np.float64), np.asarray([], dtype=np.float64)

    threads_per_block = omega_size
    # Keep one r-value per block for better occupancy while avoiding idle blocks.
    blocks_per_grid = max(1, min(r_size, 32))

    #set up on the device memory for read only memory segments
    r_vals = cuda.to_device(np.asarray(r_series_guess, dtype=np.float64))
    omega_vals = cuda.to_device(np.asarray(freq, dtype=np.float64))
    targets = cuda.to_device(np.asarray(truth, dtype=np.float64))
    fft_curve = cuda.device_array(shape=(omega_size,), dtype=np.float64)
    grad = cuda.device_array(shape=(r_size,), dtype=np.float64)
    sub_matrix = cuda.device_array(shape=(r_size, omega_size), dtype=np.float64)

    #call gpu kernel: output --> slope, area_curve
    #print("\n\n\n" + str(type(gpu.grad_calc)) + "\n\n\n")
    gpu.grad_calc[blocks_per_grid, threads_per_block](
        r_vals, omega_vals, targets, fft_curve, sub_matrix, FS, max_iterations, grad
    )
    cuda.synchronize()
    r_series_guess = r_vals.copy_to_host()
    fft_area = fft_curve.copy_to_host()
    #slope = slope.copy_to_host()
    #area_curve = area_curve.copy_to_host()
    #return slope, area_curve, r_found
    return r_series_guess, fft_area

def descent(truth, freq, FS, guess, max_iteration=1500):
    """Main function to be called for performing the gradient descent technique. """
    #single call to gpu per descent operation (low I/O time)
    #curr_grad, curve_area, r_series_found = _df_gpu(truth, freq, guess, max_iteration)
    r_series_found, curve_area = _df_gpu(truth, freq, FS, guess, max_iteration)

    #error function
    #rms = math.sqrt(mean_squared_error(truth, curve_area))
    #====Code for testing.py to graph my fit====
    #print("Final Area function: ", self._funct(xs.get()))
    #print("Area curve length: ", len(curve_area))
    #print("Freq range: ", freq[0], " --- ", freq[-1])
    #print("R series:\n", r_series_found)
    #save area_curve data for examination
    #plt.clf()
    #plt.plot(freq, curve_area, label='Model')
    #plt.plot(freq, truth, linestyle='dashed', label='Target')
    #plt.legend()
    #plt.savefig('./test_fit.png')
    #pdb.set_trace()

    #return list(map(lambda x: round(x, 5), r_series_found)), rms, curve_area
    return list(map(lambda x: round(x, 5), r_series_found))
