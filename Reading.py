import numpy as np
import math
import matplotlib.pyplot as plt
'''grf= np.load('grf_LCDAM.npy')
plt.imshow(grf)
plt.colorbar()
plt.show()'''
N = 512
patch_size = N
c = 5./512
z = 30
foregroung_type = 1

def fg_normalize(grf_fg, fg_type):
    if fg_type == 1:
        mean, std = 253*(1420/(1+z)*1/120)**-2.8, 1.3*(1420/(1+z)*1/120)**-2.8
    if fg_type == 2:
        mean, std = 1,1
    if fg_type == 3:
        mean, std = 2.2*(1420/(1+z)*1/120)**-2.15, 0.05*(1420/(1+z)*1/120)**-2.15
    if fg_type == 4:
        mean, std = 1,1
    sum = 0
    for i in range(0, len(grf_fg)):
        for j in range(0, len(grf_fg)):
            sum += np.abs(grf_fg[i, j]) ** 2
    sum = sum - grf_fg[0, 0] ** 2
    for i in range(0, len(grf_fg)):
        for j in range(0, len(grf_fg)):
            grf_fg[i, j] = np.sqrt(patch_size ** 4 * std ** 2 * 1 / sum) * grf_fg[i, j]
    grf_fg[0][0] = mean * patch_size ** 2
    return  grf_fg




#deep21 arXiv:2010.15843       A  beta  alpha Xi   type
#--------------------------------------------------------
#Galactic Synchrotron       1100, 3.3, 2.80, 4.0)   1
#Point Sources                57, 1.1, 2.07, 1.0)   2
#Galactic free-free        0.088, 3.0, 2.15, 32.)   3
#Extragalactic free-free   0.014, 1.0, 2.10, 35.)   4
#https://arxiv.org/pdf/0804.1130.pdf  and   https://arxiv.org/pdf/astro-ph/0408515.pdf -->Implementation
def foregroung(l, fg_type):
    if fg_type == 1:
        A, beta, alpha = 1100., 3.3, 2.80
    if fg_type == 2:
        A, beta, alpha = 57., 1.1, 2.07 #https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/flux-to-brightness.pdf and https://arxiv.org/pdf/0804.1130.pdf extraprolated to relevant freq
    if fg_type == 3:
        A, beta, alpha = 0.088, 3.0, 2.15
    if fg_type == 4:#https://arxiv.org/pdf/astro-ph/0408515.pdf --> uncertainty at least two orders of magnitude
        A, beta, alpha = 0.014, 1.0, 2.10
    dummy = np.zeros((N,N))
    for i in range(0,len(l)):
        for j in range(0,len(l)):
            if l[i][j]<1:
                dummy[i][j] = A * (1100. / (1)) ** beta * (130. ** 2 / 1420. ** 2) ** alpha * (
                            1 + z) ** (2 *alpha)
            else:
                dummy[i][j] = A * (1100. / (l[i][j]+1)) ** beta * (130 ** 2 / 1420 ** 2) ** alpha* (1+z)**(2*alpha)#(1. / (a + 1.) * ((1. + 30) ** (a + 1.) - (1. + 30 -0.008) ** (a + 1.))) ** 2
    return dummy


kx, ky = np.meshgrid(2 * math.pi * np.fft.fftfreq(N, c),
                         2 * math.pi * np.fft.fftfreq(N, c))
mag_k = np.sqrt(kx ** 2 + ky ** 2)
grf = np.random.normal(0., 1., size = (patch_size, patch_size))
#grf = np.random.normal(15.64, 63, size = (patch_size, patch_size))
l = 360 * mag_k/ (2 * math.pi)
fg = foregroung(l, foregroung_type)
grf_fg = np.fft.fft2(grf)*fg**0.5*1e-3 #in Kelvin
grf_norm_fg = fg_normalize(grf_fg, foregroung_type)
print(np.std(np.fft.ifft2(    grf_norm_fg   ).real))
plt.imshow(np.fft.ifft2(    grf_norm_fg   ).real)
plt.colorbar()
plt.show()
