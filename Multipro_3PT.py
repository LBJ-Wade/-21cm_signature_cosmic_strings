#!/usr/bin/env python
# coding: utf-8

# In[8]:

import time
import numpy as np
import matplotlib.pyplot as plt
import math
import multiprocessing
import warnings

patch_size = 512
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
c = angle_per_pixel
N = 512


def power_spectrum(k, alpha=2, sigma=1.):
    warnings.filterwarnings("ignore")
    out = 1 / k ** alpha * 1 / sigma ** 2
    out[k < 0.01] = 1 / (0.01) ** alpha
    return out


def signal_ft(k1, k2):
    return 0.1* (
            1 / (math.pi * k1) * 1 / (math.pi * k2) * np.sin(math.pi * k1 * 1.) *
            np.sin(math.pi * k2 * 1.))


def sort_ft(field):
    dummy = np.array(np.zeros((len(field), len(field[0]))), dtype=complex)
    N = len(field)
    for k in range(0, int(N/2)):
        for l in range(0, int(N/2)):
            dummy[-k+int(N/2)-1][l+int(N/2)] = field[k][l]
    for k in range(0, int(N/2)):
        for l in range(int(N/2), N):
            dummy[-k+int(N/2)-1][l-int(N/2)] = field[k][l]
    for k in range(int(N/2), N):
        for l in range(0, int(N/2)):
            dummy[N-(k-int(N/2))-1][l+int(N/2)] = field[k][l]
    for k in range(int(N/2), N):
        for l in range(int(N/2), N):
            dummy[N-(k-int(N/2))-1][l-int(N/2)] = field[k][l]
    return dummy


def multiprocessing_fun(j, threepoint_average_r, threepoint_average_i, threepoint_average_signal_r, threepoint_average_signal_i):
    np.random.seed(j)
    grf = np.random.normal(0, 1, (N, N))
    kx, ky = np.meshgrid(2 * math.pi * np.fft.fftfreq(N, c), 2 * math.pi * np.fft.fftfreq(N, c))
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    for i in range(0, N):
        ky[0][i] = 0.001
    for i in range(0, N):
        kx[i][0] = 0.001
    ft_sig = signal_ft(kx, ky)
    ft_signal = (ft_sig + np.fft.fft2(grf) * power_spectrum(mag_k, 2, 1) ** .5 )
    ft = (np.fft.fft2(grf) * power_spectrum(mag_k, 2, 1) ** .5)
    ft_ordered = sort_ft(ft)
    ft_ordered_signal = sort_ft(ft_signal)
    threepoint = 0
    threepoint_signal = 0
    for k in range(0, N):
        for l in range(0, N):
            threepoint += ft_ordered[k][l] * ft_ordered[N - k - 1][N - l - 1] * ft_ordered[N - l - 1][k]
            threepoint_signal += ft_ordered_signal[k][l] * ft_ordered_signal[N - k - 1][N - l - 1] * \
                                 ft_ordered_signal[N - l - 1][k]
    threepoint_average_r[j] = (threepoint / N ** 2).real
    threepoint_average_i[j] = (threepoint / N ** 2).imag
    threepoint_average_signal_r[j] = (threepoint_signal / N ** 2).real
    threepoint_average_signal_i[j] = (threepoint_signal / N ** 2).imag


def combine_complex(a, b):
    dummy = []#np.array(np.zeros(len(a)), dtype=complex)
    for i in range(0, len(a)):
        if np.abs(a[i]+1j*b[i])<30000:
            dummy.append(a[i]+1j*b[i])
    return dummy


n = 500000
parts = 5000
bins = 300

threepoint_average_r = multiprocessing.Array('d', range(n))
threepoint_average_i = multiprocessing.Array('d', range(n))
threepoint_average_signal_r = multiprocessing.Array('d', range(n))
threepoint_average_signal_i = multiprocessing.Array('d', range(n))

threepoint_average = []#np.ndarray(np.zeros(n), dtype=complex)
threepoint_average_signal = []#np.ndarray(np.zeros(n), dtype=complex)
for k in range(0, parts):
    processes = []
    for i in range(int(k*n/parts), int((k+1)*n/parts)):
        p = multiprocessing.Process(target=multiprocessing_fun, args=(i, threepoint_average_r, threepoint_average_i, threepoint_average_signal_r, threepoint_average_signal_i))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    for process in processes:
        process.terminate()
    del processes
threepoint_average = np.array(combine_complex(np.array(threepoint_average_r), np.array(threepoint_average_i)))
threepoint_average_signal = np.array(combine_complex(np.array(threepoint_average_signal_r), np.array(threepoint_average_signal_i)))

print(np.abs(np.mean(threepoint_average)))
print(np.abs(np.mean(threepoint_average_signal)))
plt.hist(np.array(threepoint_average).real, range=(-15000, 15000), bins=100)
plt.savefig('test_3PF.png', dpi=400)
plt.clf()
plt.hist(np.array(threepoint_average_signal).real,range = (-15000, 15000), bins=100)
plt.savefig('test_3PF_with_sign.png', dpi=400)




#plt.xlabel('degree')
#plt.ylabel('degree')
#my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
#plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
#plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
#plt.imshow(np.fft.ifft2(ft).real, interpolation='none')
#plt.show()
#plt.savefig('test_GRF_power.png', dpi=400)