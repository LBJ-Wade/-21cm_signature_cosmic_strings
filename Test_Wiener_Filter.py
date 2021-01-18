#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import math
patch_size = 512
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
wake_size = [1.,1.]
c = angle_per_pixel
N = 512


def power_spectrum(k, alpha=2., sigma=1.):
    out = 1/k**alpha * 1/sigma**2
    out[k < 0.01] = 1/(0.01)**alpha
    return out


bins = 300
grf = np.random.normal(0, 1, size = (patch_size, patch_size)) + 1j * np.random.normal(0, 1, size = (patch_size, patch_size))
kx, ky = np.meshgrid(np.fft.fftfreq(N, c), np.fft.fftfreq(N, c))
mag_k = np.sqrt(kx**2 + ky**2)
signal = np.zeros((N, N))
for i in range(306, 407):
    for j in range(306, 407):
        signal[i][j] = 0.2
kx[0][0] = 0.001
for i in range(0, N):
    ky[0][i] = 0.001
for i in range(0, N):
    kx[i][0] = 0.001
ft_signal_square = 0.2 * (1/(math.pi * kx) * 1/(math.pi * ky) * np.sin(math.pi * kx * wake_size[0]) * np.sin(math.pi * ky * wake_size[1]))
ft_signal_square[0][0]=0.01
print(kx)
plt.imshow(np.abs(ft_signal_square))
plt.colorbar()
plt.show()
plt.imshow(np.abs(np.fft.fft2(signal)))
plt.colorbar()
plt.show()
grf_w_power_spec_noise = np.fft.ifft2(np.fft.fft2(grf) * power_spectrum(mag_k)**.5 + ft_signal_square)
ft_all = (np.fft.fft2(grf) * power_spectrum(mag_k)**.5 + ft_signal_square)
pspec = np.abs(ft_all)**2/N**2
k_bins = np.linspace(0.1, 0.95*mag_k.max(), bins)
k_bin_cents = k_bins[:-1] + (k_bins[1:] - k_bins[:-1])/2
digi = np.digitize(mag_k, k_bins) - 1
plt.xlabel('degree')
plt.ylabel('degree')
my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.imshow(grf_w_power_spec_noise.real, interpolation='none')
plt.colorbar()
plt.show()
#plt.savefig('test_GRF_power.png', dpi=400)

#Wiener-filter: (n times)
pspec_signal = np.abs(ft_signal_square)**2/N**2
pspec_noise = power_spectrum(mag_k)#/(2 * math.pi)**2
wien_fn = pspec_signal/(pspec_noise + pspec_signal)
wien_fn2 = pspec_signal/pspec_noise
#plt.imshow(pspec_noise.real)
#plt.colorbar()
#plt.show()
wien_real = np.fft.ifft2(wien_fn * ft_all).real
wien_real2 = np.fft.ifft2(wien_fn2 * ft_all).real

plt.xlabel('degree')
plt.ylabel('degree')
my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.imshow(wien_real, interpolation='none')
plt.colorbar()
plt.show()
plt.imshow(wien_real2, interpolation='none')
plt.colorbar()
plt.show()




'''binned_ps = []
for i in range(0, digi.max()):
    binned_ps.append(np.mean(pspec[digi == i]))
binned_ps = np.array(binned_ps)
binned_ft = []
for k in range(0, digi.max()):
    binned_ft.append(np.mean(ft[digi == k]))
binned_ft = np.array(binned_ft).real  #instead of np.abs'''

#plt.plot(k_bin_cents, binned_ps)
#plt.plot(k_bin_cents, power_spectrum(k_bin_cents))
#plt.xlim(0, 0.5)
#plt.show()

#plt.plot(k_bin_cents,binned_ft/np.sqrt(power_spectrum(k_bin_cents)))
#plt.show()
#print(np.std(binned_ft/np.sqrt(power_spectrum(k_bin_cents)*N)))





