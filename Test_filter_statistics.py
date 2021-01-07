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
        signal[a][b] = 0.1
pspec_signal = np.abs(np.fft.fft2(signal))**2 / N ** 2

n = 100
sigma1 = 1
bins = 300
chi2 = np.zeros(n)
chi2_check = np.zeros(n)
chi2_real = np.zeros(n)
for j in range(0, len(chi2)):
    grf = np.random.normal(0, sigma1, (N, N))
    kx, ky = np.meshgrid(np.fft.fftfreq(N, c), np.fft.fftfreq(N, c))
    mag_k = np.sqrt(kx**2 + ky**2)
    pspec_noise = power_spectrum(mag_k, 2, 1)
    wien_fn = pspec_signal / (pspec_noise + pspec_signal)

    ft = (np.fft.fft2(grf) * power_spectrum(mag_k, 2, 1) ** .5 + np.fft.fft2(signal))
    print(ky)
    plt.imshow(np.abs(ft))
    plt.show()
    ft_filtered = ft * wien_fn
    ift_filtered = np.fft.ifft2(ft_filtered).real

    pspec = np.abs(ft_filtered) ** 2 / N ** 2
    pspec_check = np.abs(ft) ** 2 / N ** 2 #unfiltered
    pspec_noise = power_spectrum(mag_k, 2, 1) * wien_fn**2 #filtered power spectrum
    k_bins = np.linspace(0.1, 0.95 * mag_k.max(), bins)
    k_bin_cents = k_bins[:-1] + (k_bins[1:] - k_bins[:-1])/2
    digi = np.digitize(mag_k, k_bins) - 1

    binned_ps = []
    for i in range(0, digi.max()):
        binned_ps.append(np.mean(pspec[digi == i]))
    binned_ps = np.array(binned_ps)

    binned_ps_check = []
    for i in range(0, digi.max()):
        binned_ps_check.append(np.mean(pspec_check[digi == i]))
    binned_ps_check = np.array(binned_ps_check)

    binned_ps_noise = []
    for i in range(0, digi.max()):
        binned_ps_noise.append(np.mean(pspec_noise[digi == i]))
    binned_ps_noise = np.array(binned_ps_noise)
    '''plt.plot(k_bin_cents, binned_ps)
    plt.plot(k_bin_cents, power_spectrum(k_bin_cents, 2, 1))
    plt.xlim(1.5, 2)
    plt.ylim(0, 10)
    plt.show()'''
    chi2[j] = np.sum(binned_ps / binned_ps_noise)  # Up to normalisation
    chi2_check[j] = np.sum(binned_ps_check / power_spectrum(k_bin_cents, 2, 1))
    for a in range(0, N):
        for b in range(0, N):
            chi2_real[j] = chi2_real[j] + (ift_filtered[a][b])**2


print(np.mean(chi2)/bins)
print(np.mean(chi2_check/bins))
print(np.mean(chi2_real)/N**2)



#plt.xlabel('degree')
#plt.ylabel('degree')
#my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
#plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
#plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
#plt.imshow(np.fft.ifft2(ft).real, interpolation='none')
#plt.show()
#plt.savefig('test_GRF_power.png', dpi=400)