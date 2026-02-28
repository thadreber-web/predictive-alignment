"""
Copyright 2025 Toshitake Asabuki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file recept in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numba
import pylab as pl
from tqdm import tqdm

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
params = {
    'backend': 'ps',
    'axes.labelsize': 10,
    'text.fontsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [10 / 2.54, 6 / 2.54]
}

# Network size
N = 500

# Learning rate
eps = 10 ** (-3)

@numba.njit(parallel=True, fastmath=True, nogil=True)
def learning(w, g_V_star, PSP_star, g_V_som, mask):
    """
    Update recurrent weight matrix using the learning rule.
    """
    for i in numba.prange(len(w[:, 0])):
        for l in numba.prange(len(w[0, :])):
            delta = (-(g_V_star[i]) + g_V_som[i]) * PSP_star[l]
            w[i, l] += eps * delta
            w[i, l] *= mask[i, l]
    return w

@numba.njit(parallel=True, fastmath=True, nogil=True)
def learning_readout(w, y, r, z):
    """
    Update readout weight using the learning rule.
    """
    for i in numba.prange(N):
        delta = (-(y) + z) * r[i]
        w[i] += eps * delta
    return w

# Time parameters
dt = 1
tau = 10

# Initialize recurrent weight matrix M
M = np.zeros((N, N))
p_connect_M = 1
g = 0.5
scale = 1.0 / np.sqrt(p_connect_M * N)

mask = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if np.random.rand() < p_connect_M:
            M[i, j] = np.random.randn() * g * scale
            mask[i, j] = 1

# Feedback weight
w_fb = 3 * 2 * (np.random.rand(N) - 0.5)

x = np.random.randn(N)
r = np.tanh(x)

# Strong, fixed recurrent matrix
M_chaos = np.zeros((N, N))
p_connect_chaos = 0.1
scale_chaos = 1.0 / np.sqrt(p_connect_chaos * N)
g_chaos = 1.2
for i in range(N):
    for j in range(N):
        if np.random.rand() < p_connect_chaos:
            M_chaos[i, j] = np.random.randn() * g_chaos * scale_chaos

w_chaos = np.random.randn(N) / np.sqrt(N)
w_readout = np.random.randn(N) / np.sqrt(N)

# Training
simtime_len = 200 * 1000
y = 0

for i in tqdm(range(simtime_len), desc="[training]"):
    # ---------------------
    # Example targets
    # ---------------------
    #### discontinuous
    ###
    # T = 60
    # z = 1.5*2*((np.sin(2*np.pi*i/T)>=0)-0.5)
    #########

    #### Sawtooth
    # T = 100
    # z =1.5*signal.sawtooth(2*np.pi*i/T, width=0.8)
    # M = np.outer(w_fb,w_readout)

    # Periodic
    T = 150
    z = (np.sin(2 * np.pi * i / T)
         + np.sin(2 * 2 * np.pi * i / T)
         + np.sin(3 * 2 * np.pi * i / T)) * 1

    ### Sinwave
    # T = 6*tau  ## short
    # T = 200*tau ## long
    # T = 10*tau  # normal
    # z = 1.5*np.sin(2*np.pi*i/T)
    # ---------------------

    # Recurrent + chaos contribution
    M_term = M.dot(r)
    Chaos_term = M_chaos.dot(r)

    # Update state
    x = (1.0 - dt / tau) * x + (M_term + Chaos_term) / tau

    # Compute feedback
    FB = w_fb.dot(y)

    # Update recurrent weights
    M = learning(M, M_term, r, FB + 1 * Chaos_term, mask)

    # Update firing rates
    r = np.tanh(x)

    # Readout
    y = w_readout.dot(r)

    # Update readout weights
    w_readout = learning_readout(w_readout, y, r, z)

# Testing
simtime_len2 = 600
x_list_testing = np.zeros((N, simtime_len2))
y_list = np.zeros(simtime_len2)
target_list = np.zeros(simtime_len2)

for i in tqdm(range(simtime_len2), desc="[testing]"):
    # Recurrent update
    x = (1.0 - dt / tau) * x + (M + M_chaos).dot(r) / tau
    r = np.tanh(x)

    # Readout
    y = w_readout.dot(r)

    # Target (same as training examples)
    z = (np.sin(2 * np.pi * (i + simtime_len) / T)
         + np.sin(2 * 2 * np.pi * (i + simtime_len) / T)
         + np.sin(3 * 2 * np.pi * (i + simtime_len) / T)) * 1

    # Record
    x_list_testing[:, i] = r
    y_list[i] = y
    target_list[i] = z

# Plot results
fig, ax = plt.subplots(figsize=(6, 2))
pl.plot(target_list, c='gray', label='Target')
pl.plot(y_list, c='orange', label='Readout')
plt.legend()
plt.tight_layout()
plt.savefig('trained-output.pdf', format='pdf', dpi=350)
plt.show()
