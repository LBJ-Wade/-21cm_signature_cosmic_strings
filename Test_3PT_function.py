#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import math
patch_size = 512
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
c = 2 * math.pi * angle_per_pixel
N = 512
def power_spectrum(k, alpha=2, sigma=1.):
    out = 1 / k ** alpha * 1 / sigma ** 2
    out[k < 0.01] = 1 / (0.01) ** alpha
    return out


signal = np.zeros((N, N))
for a in range(206, 306):
    for b in range(206, 306):
        signal[a][b] = 0.01
pspec_signal = np.abs(np.fft.fft2(signal))**2 / N ** 2

n = 100
sigma1 = 1
bins = 300
'''chi2 = np.zeros(n)
chi2_check = np.zeros(n)
chi2_real = np.zeros(n)'''
threepoint_average = np.array(np.zeros(n), dtype=complex)
threepoint_average_signal = np.array(np.zeros(n), dtype=complex)
for j in range(0, len(threepoint_average)):
    grf = np.random.normal(0, sigma1, (N, N))
    #grf2 = np.random.normal(0, sigma1, (N, N))
    kx, ky = np.meshgrid(np.fft.fftfreq(N, c), np.fft.fftfreq(N, c))
    mag_k = np.sqrt(kx**2 + ky**2)
    pspec_noise = power_spectrum(mag_k, 2, 1)
    #wien_fn = pspec_signal / (pspec_noise + pspec_signal)
    ft = (np.fft.fft2(grf) * power_spectrum(mag_k, 2, 1) ** .5 )#+ np.fft.fft2(signal))
    ft_signal = (np.fft.fft2(grf) * power_spectrum(mag_k, 2, 1) ** .5 + np.fft.fft2(signal))
    ft_ordered = np.array(np.zeros((N, N)),dtype=complex)
    ft_ordered_signal = np.array(np.zeros((N, N)),dtype=complex)
    for k in range(0, N/2):
        for l in range(0, N/2):
            ft_ordered[-k+N/2-1][l+N/2] = ft[k][l]
            ft_ordered_signal[-k + N / 2 - 1][l + N / 2] = ft_signal[k][l]
    for k in range(0, N/2):
        for l in range(N/2, N):
            ft_ordered[-k+N/2-1][l-N/2] = ft[k][l]
            ft_ordered_signal[-k + N / 2 - 1][l - N / 2] = ft_signal[k][l]
    for k in range(N/2, N):
        for l in range(0, N/2):
            ft_ordered[N-(k-N/2)-1][l+N/2] = ft[k][l]
            ft_ordered_signal[N - (k - N / 2) - 1][l + N / 2] = ft_signal[k][l]
    for k in range(N/2, N):
        for l in range(N/2, N):
            ft_ordered[N-(k-N/2)-1][l-N/2] = ft[k][l]
            ft_ordered_signal[N - (k - N / 2) - 1][l - N / 2] = ft_signal[k][l]
    '''print(ft_ordered.min())
    print(np.abs(ft).min())
    plt.imshow(np.abs(ft_ordered))
    plt.colorbar()
    plt.show()
    plt.imshow(ft)
    plt.colorbar()
    plt.show()''' #test successful reordered

    threepoint = 0
    threepoint_signal = 0
    for k in range(0, N):
        for l in range(0, N):
            threepoint += ft_ordered[k][l]*ft_ordered[N-k-1][N-l-1]*ft_ordered[N-l-1][k]
            threepoint_signal += ft_ordered_signal[k][l] * ft_ordered_signal[N - k - 1][N - l - 1] * ft_ordered_signal[N - l - 1][k]
    threepoint_average[j]=threepoint/N**2
    threepoint_average_signal[j] = threepoint_signal/ N ** 2
    #print(threepoint/N**2)
    #print("...............")
print(np.mean(threepoint_average))
print(np.abs(np.mean(threepoint_average)-np.mean(threepoint_average_signal)))






#plt.xlabel('degree')
#plt.ylabel('degree')
#my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
#plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
#plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
#plt.imshow(np.fft.ifft2(ft).real, interpolation='none')
#plt.show()
#plt.savefig('test_GRF_power.png', dpi=400)