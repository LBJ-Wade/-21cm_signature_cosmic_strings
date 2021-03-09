#!/usr/bin/env python
# coding: utf-8

# In[8]:

import numpy as np
import matplotlib.pyplot as plt
import math
import multiprocessing
import warnings
from scipy.optimize import curve_fit



#define constants according to arXiv: 1006.2514v3
#according to The Astrophysical Journal, 622:1356-1362, 2005 April 1, Table 2. Units [cm^3 s^-1]
def deexitation_crosssection(t_k):
    if t_k < 1:
        return 1.38e-13
    if 1 < t_k and t_k <= 2:
        return 1.38e-13 + (t_k - 1) * (1.43 - 1.38) * 1e-13
    if 2 < t_k and t_k <= 4:
        return 1.43e-13 + (t_k - 2) / 2 * (2.71 - 1.43) * 1e-13
    if 4 < t_k and t_k <= 6:
        return 2.71e-13 + (t_k - 4) / 2 * (6.6 - 2.71) * 1e-13
    if 6 < t_k and t_k <= 8:
        return 6.60e-13 + (t_k - 6) / 2 * (1.47e-12 - 6.6e-13)
    if 8 < t_k and t_k <= 10:
        return 1.47e-12 + (t_k - 8) / 2 * (2.88 - 1.47) * 1e-12
    if 10 < t_k and t_k <= 15:
        return 2.88e-12 + (t_k - 10) / 5 * (9.10 - 2.88) * 1e-12
    if 15 < t_k and t_k <= 20:
        return 9.1e-12 + (t_k - 15) / 5 * (1.78e-11 - 9.10 * 1e-12)
    if 20 < t_k and t_k <= 25:
        return 1.78e-11 + (t_k - 20) / 5 * (2.73 - 1.78) * 1e-11
    if 25 < t_k and t_k <= 30:
        return 2.73e-11 + (t_k - 25) / 5 * (3.67 - 2.73) * 1e-11
    if 30 < t_k and t_k <= 40:
        return 3.67e-11 + (t_k - 30) / 10 * (5.38 - 3.67) * 1e-11
    if 40 < t_k and t_k <= 50:
        return 5.38e-11 + (t_k - 40) / 10 * (6.86 - 5.38) * 1e-11
    if 50 < t_k and t_k <= 60:
        return 6.86e-11 + (t_k - 50) / 10 * (8.14 - 6.86) * 1e-11
    if 60 < t_k and t_k <= 70:
        return 8.14e-11 + (t_k - 60) / 10 * (9.25 - 8.14) * 1e-11
    if 70 < t_k and t_k <= 80:
        return 9.25e-11 + (t_k - 70) / 10 * (1.02e-10 - 9.25 * 1e-11)
    if 80 < t_k and t_k <= 90:
        return 1.02e-10 + (t_k - 80) / 10 * (1.11 - 1.02) * 1e-10
    if 90 < t_k and t_k <= 100:
        return 1.11e-10 + (t_k - 90) / 10 * (1.19 - 1.11) * 1e-10
    if 100 < t_k and t_k <= 200:
        return 1.19e-10 + (t_k - 100) / 100 * (1.75 - 1.19) * 1e-10
    if 200 < t_k and t_k <= 300:
        return 1.75e-10 + (t_k - 200) / 100 * (2.09 - 1.75) * 1e-10
    else:
        print('T_K is out of scope for the deexcitation fraction')
        return 0



patch_size = 512
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
c = angle_per_pixel
N = patch_size
z = 12

#redshift string formation
z_i = 3000
#frequency bin: 10kHz = 0.01 MHz
delta_f = 0.05
#thickness redshift bin (assuming we look at f in [f_0, f_0 + delta_f])
delta_z = -delta_f/(1420)*(z+1)
#redshift of center of wake
z_wake = z+delta_z/2
#string tension in units of [10^-6]
gmu_6 = 0.3
#string speed
vsgammas_square = 1./3
#temperature of HI atoms inside the wake [Kl
T_K = 20 * gmu_6**2 * vsgammas_square * (z_i+1.)/(z_wake+1)
#CMB temperature [K]
T_gamma = 2.725*(1+z_wake)
#temperature of the cosmic gas for z<150
T_g = 0.02*(1+z)**2
#background numberdensity hydrogen [cm^-3]
nback=1.9e-7 *(1.+z_wake)**3
#collision coeficcient hydrogen-hydrogen (density in the wake is 4* nback, Delta E for hyperfine is 0.068 [K], A_10 = 2.85e-15 [s^-1])
#xc = 4*nback*deexitation_crosssection(T_K)* 0.068/(2.85e-15 *T_gamma)
deex_fit = (1e-13*(27598 - 8.5 + (-13797.5 - 13799.5)/(1 + (T_K/24.0322)**2.28305)**0.0136134))
xc = 4*nback*deex_fit* 0.068/(2.85e-15 *T_gamma)
print('We make use of an fit function for the deexcitation cross section')
#fraction of baryonc mass comprised of HI. Given that we consider redshifts of the dark ages, we can assume that all the
#hydrogen of the universe is neutral and we assume the mass fraction of baryoni is:
xHI = 0.75
#background temperature [mK] [formula according to arXiv:2010.15843] (assume Omega_b, h, Omega_Lambda, Omega_m as in arXiv: 1405.1452[they use planck collaboration 2013b best fit])
T_back = (0.19055) * (0.049*0.67*(1.+z_wake)**2 * xHI)/np.sqrt(0.267*(1.+z_wake)**3 + 0.684)
#another option is to follow arXiv:1401.2095 (OmegaHI = 0.62*1e-3, rest from Roberts paper arXiv:1006.2514)   in mK !!!!
T_back2 = 0.1* 0.62*1e-3/(0.33*1e-4) *np.sqrt((0.26 + (1+z_wake)**-3 * (1-0.26-0.042))/0.29)**-1 * (1+z_wake)**0.5/2.5**0.5
theta1 =  math.pi*0.32 #angle 1 in z-space
theta2 = 0 #angle 2 in z-space

"""Including Diffusion effects."""
if T_K > 3 * T_g:
    # T_b.append(1e3* 0.07  *xc[j]/(xc[j]+1.)*(1-T_gamma/T_K[j])*np.sqrt(1.+z_wake)*(2*np.sin(theta1)**2)**-1)
    T_b = (17 * xc / (xc + 1) * (1 - T_gamma / T_K) * 4 * np.sqrt(1. + z_wake) * ( 2 * np.sin(theta1) ** 2) ** -1)
else:
    if T_K < T_g:
        T_b = (17 * xc / (xc + 1) * (1 - T_gamma / (3 * T_g)) * (1 + T_K / T_g) * np.sqrt(1. + z_wake) * (2 * np.sin(theta1) ** 2) ** -1 * T_g / T_K)
    else:
        T_b = (17 * xc / (xc + 1) * (1 - T_gamma / (3 * T_g)) * (1 + T_K / T_g) * np.sqrt(1. + z_wake) * (2 * np.sin(theta1) ** 2) ** -1)

wake_brightness = T_b* 1e3 #in mK
wake_thickness = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * (z_wake + 1.)**0.5 *2.*np.sin(theta1)**2*1/np.cos(theta1)
#wake_thickness = -delta_z
rot_angle_uv =0# math.pi/4 #rotation angle in the uv plane
wake_size_angle = [1., 1.] #in degree
shift_wake_angle = [0, 0]
print('3PF evaluation!')
print('Frequency interval: f in ['+ str(1420/(1.+z))+', '+str(1420/(1.+z)+delta_f) +'] MHz.')
print('Therefore, dz = '+ str(delta_z)+' and we cover ['+ str(z)+', '+ str(z+delta_z)+ '].')
print('The string wake brightness temperature at '+str(z)+' is '+str(T_b)+' mK.')
#wakes extend in frequency space is [v_0 + wakethickness/2, v_0 - wakethickness/2]
print('The wakes thickness in redshift space is given by dz_wake = '+str(wake_thickness))


#https://arxiv.org/pdf/0804.1130.pdf
'''def foreground_power_spectrum(k, A_pure, beta, a, Xi, sigma): # Xi):
    #an example from arXiv:2010.15843 (deep21)
    sec_small = sorted(k[int(patch_size/2)])[1]
    k[int(patch_size/2)][int(patch_size/2)] = sec_small
    lref = 1100.
    A = A_pure #*1e-3
    vref = 130.  # MHz
    if k[1].ndim == 0:
        ps = np.zeros(len(k))
        for i in range(0, len(k)):
            l = 360 * k[i] / (2 * math.pi)
            l_bottom = math.floor(l)
            l_top = l_bottom + 1
            delta_l = -l + l_top
            if l_bottom == 0:
                if l < 0.01:
                    ps[i] = A * (lref / 1) ** beta * (vref ** 2 / 1420 ** 2) ** a * (1. / (a + 1.) * ((1. + z) ** (a + 1.) - (1. + z + delta_z) ** (a + 1.))) ** 2
                else:
                    ps[i] = A * (lref / 1) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a + 1.) * ((1. + z)**(a + 1.) - (1. + z + delta_z)**(a+1.))) ** 2
            else:
                ps[i] = A * (lref / l_top) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+ z + delta_z)**(a+1.)))**2 + delta_l * (A * (lref / l_bottom) ** beta * (vref ** 2 / 1420**2) ** a - A * (
                            lref / l_top) ** beta * (vref ** 2 / 1420**2) ** a) * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+ z + delta_z)**(a+1.)))**2  # exp()
                #ps[i] = A * (lref / l_top) ** beta * (vref ** 2 / 1420 ** 2) ** a *((1+z_wake)**2)**a + delta_l * (
                #                    A * (lref / l_bottom) ** beta * (vref ** 2 / 1420 ** 2) ** a - A * (
                #                    lref / l_top) ** beta * (vref ** 2 / 1420 ** 2) ** a) *((1+z_wake)**2)**a

        return ps*1/sigma**2
    else:
        ps = np.zeros((len(k[1]), len(k[1])))
        for i in range(0, len(k[1])):
            for j in range(0, len(k[1])):
                l = 360 * k[i][j]/(2 * math.pi)
                l_bottom = math.floor(l)
                l_top = l_bottom + 1
                delta_l = l - l_bottom
                if l_bottom == 0:
                    if l < 0.01:
                        ps[i][j] = A * (lref / 1) ** beta * (vref ** 2 / 1420 ** 2) ** a *(1. / (a + 1.) * ((1. + z) ** (a + 1.) - (1. + z + delta_z) ** (a + 1.))) ** 2 #((1+z_wake)**2)**a
                    else:
                        ps[i][j] = A * (lref/ 1)**beta * (vref**2/1420**2)**a * (1./(a+1.) * ((1.+z)**(a+1.) - (1. + z + delta_z)**(a+1.)))**2 #((1+z_wake)**2)**a
                else:
                    ps[i][j] = A * (lref / l_top) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+ z + delta_z)**(a+1.)))**2 + delta_l * (A * (lref/l_bottom)**beta * (vref**2/(1420**2))**a - A * (lref/l_top)**beta * (vref**2/(1420**2))**a) * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+z+delta_z)**(a+1.)))**2  #exp()
                    #ps[i][j] = A * (lref / l_top) ** beta * (vref ** 2 / 1420 ** 2) ** a *((1+z_wake)**2)**a + delta_l * (
                    #                       A * (lref / l_bottom) ** beta * (vref ** 2 / (1420 ** 2)) ** a - A * (
                    #                           lref / l_top) ** beta * (vref ** 2 / (1420 ** 2)) ** a) *((1+z_wake)**2)**a
        #print(np.mean(ps)**0.5)
        return ps/sigma**2'''

def fg_normalize(grf_fg, fg_type):
    if fg_type == 1:
        mean, std, std_eff = 253 * (1420 / (1 + z_wake) * 1 / 120) ** -2.8, 1.3 * (
                    1420 / (1 + z_wake) * 1 / 120) ** -2.8, 69*(angle_per_pixel/(5/512))**(-3.3/2)
    if fg_type == 2:
        mean, std, std_eff = 38.6 * (1420 / (1 + z_wake) * 1 / 151) ** -2.07, 2.3 * (
                    1420 / (1 + z_wake) * 1 / 151) ** -2.07, 1410*(angle_per_pixel/(5/512))**(-1.1/2)
    if fg_type == 3:
        mean, std, std_eff = 2.2 * (1420 / (1 + z_wake) * 1 / 120) ** -2.15, 0.05 * (
                    1420 / (1 + z_wake) * 1 / 120) ** -2.15, 415*(angle_per_pixel/(5/512))**(-3.0/2)
    if fg_type == 4:
        mean, std, std_eff = 1e-4 * (1420 / (1 + z_wake) * 1 / (2 * 1e3)) ** -2.1, 1e-5 * (
                    1420 / (1 + z_wake) * 1 / (2 * 1e3)) ** -2.1, 81*(angle_per_pixel/(5/512))**(-1.0/2)
    if fg_type == 6:
        mean, std, std_eff = -2.72477, 0.0000508 * (1 + 30) / (1 + z), 189 * (1 + 30) / (1 + z)*(angle_per_pixel/(5/512))**(-2.0/2)
    sum = 0
    for i in range(0, len(grf_fg)):
        for j in range(0, len(grf_fg)):
            sum += np.abs(grf_fg[i, j]) ** 2
    sum = sum - grf_fg[0, 0] ** 2
    norm = np.sqrt(patch_size ** 4 * std ** 2 * 1 / sum).real
    grf_fg = norm * grf_fg
    grf_fg[0][0] = mean * patch_size ** 2
    return grf_fg #, std_eff, norm



def LCDM(l):
    dummy = np.zeros((N, N))
    for i in range(0,len(l)):
        for j in range(0,len(l)):
            l_bottom = math.floor(l[i][j])
            l_top = l_bottom + 1
            delta_l = l[i, j] - l_bottom
            if l_bottom == 0:
                if l[i][j] < 0.1:
                    dummy[i][j] = LCDM_ps[0]
                else:
                    dummy[i][j] = LCDM_ps[l_bottom] + delta_l * (LCDM_ps[l_top] - LCDM_ps[l_bottom])
            else:
                dummy[i][j] = LCDM_ps[l_bottom] + delta_l * (LCDM_ps[l_top] - LCDM_ps[l_bottom])
    return dummy


#deep21 arXiv:2010.15843       A  beta  alpha Xi   type
#--------------------------------------------------------
#Galactic Synchrotron       1100, 3.3, 2.80, 4.0)   1
#Point Sources                57, 1.1, 2.07, 1.0)   2
#Galactic free-free        0.088, 3.0, 2.15, 32.)   3
#Extragalactic free-free   0.014, 1.0, 2.10, 35.)   4
#https://arxiv.org/pdf/0804.1130.pdf  and   https://arxiv.org/pdf/astro-ph/0408515.pdf -->Implementation
def foreground(l, fg_type):
    if fg_type == 1:
        A, beta, alpha = 1100., 3.3, 2.80
    if fg_type == 2:                    #https://arxiv.org/pdf/astro-ph/0408515.pdf and https://iopscience.iop.org/article/10.1086/588628/pdf
        A, beta, alpha = 57., 1.1, 2.07
    if fg_type == 3:
        A, beta, alpha = 0.088, 3.0, 2.15
    if fg_type == 4:                    #https://arxiv.org/pdf/astro-ph/0408515.pdf and https://iopscience.iop.org/article/10.1086/421241/pdf --> uncertainty at least two orders of magnitude
        A, beta, alpha = 0.014, 1.0, 2.10
    if fg_type == 6:
        return LCDM(l)
    dummy = np.zeros((N,N))
    for i in range(0,len(l)):
        for j in range(0,len(l)):
            alpha_random = alpha#np.random.normal(alpha, 0.1)
            if l[i][j]<1:
                dummy[i][j] = A * (1100. / (1)) ** beta * (130. ** 2 / 1420. ** 2) ** alpha * (
                            1 + z) ** (2 *alpha_random)
            else:
                l_bottom = np.floor(l[i,j])
                l_top = l_bottom+1
                delta_l = l[i,j]-l_bottom
                dummy[i][j] = A * (1100. / (l_bottom)) ** beta * (130 ** 2 / 1420 ** 2) ** alpha_random* (1+z)**(2*alpha_random) + delta_l*(A * (1100. / (l_top)) ** beta * (130 ** 2 / 1420 ** 2) ** alpha_random* (1+z)**(2*alpha_random)-A * (1100. / (l_bottom)) ** beta * (130 ** 2 / 1420 ** 2) ** alpha_random* (1+z)**(2*alpha_random))
    return dummy



def power_spectrum(k, alpha=2, sigma=1.):
    warnings.filterwarnings("ignore")
    out = 10000 / k ** alpha * 1 / sigma ** 2
    out[k < 0.01] = 1 / (0.01) ** alpha
    return out

def Pinst(l, token):
    if token == 0:
        dummy = np.zeros((len(l), len(l[1])))
        u = l / (2 * np.pi)
        for i in range(0, len(u)):
            for k in range(0, len(u[1])):
                index = 0
                for j in range(0, len(Inter_ps_u)):
                    if u[i, k] > Inter_ps_u.max():
                        index = np.pi
                        break
                    if u[i, k] < Inter_ps_u[j]:
                        index = j
                        break
                    else:
                        continue
                if index == np.pi:
                    # dummy[i,k] = -1000
                    continue
                if index - 1 < 0:
                    u_down = 0
                else:
                    u_down = Inter_ps_u[index - 1]
                u_up = Inter_ps_u[index]
                if index - 1 < 0:
                    dummy[i, k] = Inter_ps[index] - (Inter_ps[index + 1] - Inter_ps[index]) / (
                                Inter_ps_u[index + 1] - Inter_ps_u[index]) * (u_up - u[i, k])
                else:
                    dummy[i, k] = Inter_ps[index - 1] + (Inter_ps[index] - Inter_ps[index - 1]) / (
                                Inter_ps_u[index] - Inter_ps_u[index - 1]) * (u[i, k] - u_down)
        return dummy
    else:
        dummy = np.zeros((len(l), len(l[1])))
        u = l / (2 * np.pi)
        for i in range(0, len(u)):
            for k in range(0, len(u[1])):
                index = 0
                for j in range(0, len(Inter_ps_u)):
                    if u[i, k] > Inter_ps_u.max():
                        index = np.pi
                        break
                    if u[i, k] < Inter_ps_u[j]:
                        index = j
                        break
                    else:
                        continue
                if index == np.pi:
                    # dummy[i,k] = -1000
                    continue
                if index - 1 < 0:
                    u_down = 0
                else:
                    u_down = Inter_ps_u[index - 1]
                u_up = Inter_ps_u[index]
                if index - 1 < 0:
                    dummy[i, k] = Inter_ps[index] - (Inter_ps[index + 1] - Inter_ps[index]) / (
                            Inter_ps_u[index + 1] - Inter_ps_u[index]) * (u_up - u[i, k])
                else:
                    dummy[i, k] = Inter_ps[index - 1] + (Inter_ps[index] - Inter_ps[index - 1]) / (
                            Inter_ps_u[index] - Inter_ps_u[index - 1]) * (u[i, k] - u_down)

        dummy_full = np.zeros((N, N))
        for i in range(0, len(dummy[1])):
            for j in range(0, len(dummy)):
                dummy_full[j, i] = dummy[j, i]

        for i in range(0, len(dummy[1])+1):
            for j in range(0, len(dummy)):
                dummy_full[N - j - 1, N - i - 1] = dummy_full[j, i]
        dummy_full = np.fft.fftshift(dummy_full)

        for a in range(int(N/2.),N):
            for b in range(int(N/4.), int(3*N/4.)):
                dummy_full[b,a] = dummy_full[b+1,a]

        return np.fft.fftshift(dummy_full)

def GRF_inst():
    real_part = np.sqrt(0.5 * Pl) * np.random.normal(loc=0., scale=1., size=l_inst.shape) * lpix / (2.0 * np.pi)
    imaginary_part = np.sqrt(0.5 * Pl) * np.random.normal(loc=0., scale=1., size=l_inst.shape) * lpix / (2.0 * np.pi)

    # Get map in real space and return
    ft_map = (real_part + imaginary_part * 1.0j) * l_inst.shape[0] ** 2

    ft_map[0, 0] = 0.0

    grf_inst = np.zeros((N, N)) + 1J * np.zeros((N, N))
    for i in range(0, len(ft_map[1])):
        for j in range(0, len(ft_map)):
            grf_inst[j, i] = ft_map[j, i]

    for i in range(0, len(ft_map[1])):
        for j in range(0, len(ft_map)):
            grf_inst[N - j - 1, N - i - 1] = ft_map[j, i]

    grf_inst = np.fft.fftshift(grf_inst)

    for a in range(int(N / 2.), N):
        for b in range(int(N / 4.), int(3 * N / 4.)):
            grf_inst[b, a] = grf_inst[b + 1, a]

    return np.fft.fftshift(grf_inst)  # np.fft.irfft2(ft_map).real


def signal_ft(size, anglewake, angleperpixel, shift, background_on):
    #coordinate the dimensions of wake and shift
    patch = np.zeros((size, size))
    if background_on == True:
        patch = np.ones((size, size)) * T_back * -delta_z
    patch_rotated = np.zeros((size, size))
    #dz_wake = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * (z_wake + 1.)**0.5 *2.*np.sin(theta1)**2/np.cos(theta1)
    #df_wake = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * 1/(z_wake + 1.)**0.5 * 1420.*2.*np.sin(theta1)**2/np.cos(theta1) # MHz. THe 2sin^2 theta cancels when multiplied with T_b
    shift_pixel = np.zeros(2)
    shift_pixel[0] = int(np.round(shift[0]/angleperpixel))
    shift_pixel[1] = int(np.round(shift[1]/angleperpixel))
    wakesize_pixel = [int(np.round(np.cos(0)*anglewake[0]/angleperpixel)), int(np.round(anglewake[1]/angleperpixel))] #theta1 term added depending on the direction of rot
    i_x = int(size/2.+shift_pixel[0]-wakesize_pixel[0]/2.)
    f_x = int(size/2.+shift_pixel[0]+wakesize_pixel[0]/2.+1)
    i_y = int(size/2.+shift_pixel[1]-wakesize_pixel[1]/2.)
    f_y = int(size/2.+shift_pixel[1]+wakesize_pixel[1]/2.+1)
    for i in range(i_x, f_x):
        for j in range(i_y, f_y):
            #patch[i, j] = 1e3 * 1/(2.*np.sin(theta1)**2) * df_wake/delta_f * (i-i_x)*1./(f_x-i_x) * T_b # according to https://arxiv.org/pdf/1403.7522.pdf
            patch[i, j] += wake_thickness*T_b  #(1e3*0.07 * (2*np.sin(theta1)**2)**-1* xc/(xc+1.)*2./3*((1 + z_wake + dz_wake/2. * (i-i_x)*1./(f_x-i_x))**1.5-(1 + z_wake - dz_wake/2. * (i-i_x)*1./(f_x-i_x))**1.5) - 1e3*0.07*  (2*np.sin(theta1)**2)**-1 * xc/(xc+1.)*2.725/(20 * gmu_6**2 * vsgammas_square * (z_i+1.)) * 2/7. * ((1 + z_wake + dz_wake/2. * (i-i_x)*1./(f_x-i_x))**3.5-(1 + z_wake - dz_wake/2. * (i-i_x)*1./(f_x-i_x))**3.5)) #in mK
    if rot_angle_uv!=0:
        for k in range(i_x, f_x):
            for l in range(i_y, f_y):
                patch_rotated[int(np.floor(math.cos(rot_angle_uv) * (k - size/2) - math.sin(rot_angle_uv) * (l - size/2))) + size/2][
                    int(np.floor(math.sin(rot_angle_uv) * (k - size/2) + math.cos(rot_angle_uv) * (l - size/2))) + size/2] = patch[k][l]
        for p in range(1, size-1):
            for q in range(1, size-1):
                if np.abs(patch_rotated[p][q - 1] + patch_rotated[p - 1][q] + patch_rotated[p + 1][q] + patch_rotated[p][q + 1]) > 2 * max(np.abs(
                       [patch_rotated[p][q - 1], patch_rotated[p - 1][q], patch_rotated[p + 1][q], patch_rotated[p][q + 1]])):
                    a = np.array([patch_rotated[p][q - 1], patch_rotated[p - 1][q], patch_rotated[p + 1][q], patch_rotated[p][q + 1]])
                    patch_rotated[p][q] = np.mean(a[np.nonzero(a)])
        return np.fft.fft2(patch_rotated)
    return np.fft.fft2(patch)

'''def sort_ft(field):
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
    return dummy'''


def multiprocessing_fun(j, threepoint_average_r, threepoint_average_i, threepoint_average_signal_r, threepoint_average_signal_i, fg_type):
    np.random.seed(j*13)
    grf = np.fft.fft2(np.random.normal(0, 1, size = (patch_size, patch_size)))
    if foreg_type == 5:
        grf_II = np.random.normal(0., 1., size=(patch_size, patch_size))
        grf_III = np.random.normal(0., 1., size=(patch_size, patch_size))
        grf_IV = np.random.normal(0., 1., size=(patch_size, patch_size))
        grf_LCDM = np.random.normal(0., 1., size=(patch_size, patch_size))
    #grf2 = np.fft.fft2(np.random.normal(0, 1, size = (patch_size, patch_size)))
    #kx, ky = np.meshgrid(np.fft.fftshift(2 * math.pi * np.fft.fftfreq(N, c)), np.fft.fftshift( 2 * math.pi * np.fft.fftfreq(N, c)))
    kx, ky = np.meshgrid(2 * math.pi * np.fft.fftfreq(N, c),
                         2 * math.pi * np.fft.fftfreq(N, c))
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    l = 360 * mag_k / (2 * math.pi)
    ft_sig = np.fft.fftshift(signal_ft(patch_size, wake_size_angle,  angle_per_pixel, shift_wake_angle, False))
    #print(l[0])
    if fg_type == 1:
        pspectrum = foreground(l, 1)
    if fg_type == 2:
        pspectrum = foreground(l, 2)
    if fg_type == 3:
        pspectrum = foreground(l, 3)
    if fg_type == 4:
        pspectrum = foreground(l, 4)
    if fg_type == 6:
        pspectrum = foreground(l, 6)
    epsilon_fgr = eps_fg
    if foreg_type == 1:
        filter_function = ft_sig / (ft_sig + np.fft.fftshift(pspectrum))
    if foreg_type == 2:
        filter_function = ft_sig / (ft_sig + np.fft.fftshift(pspectrum))
    if foreg_type == 5:
        filter_function = ft_sig / (ft_sig + np.fft.fftshift(foreground(l, 1) + foreground(l, 2) + foreground(l, 3) + foreground(l, 4) + foreground(l, 6) + eps_noise*Pl1))
    if foreg_type == 3:
        filter_function = ft_sig / (ft_sig + np.fft.fftshift(pspectrum))
    if foreg_type == 4:
        filter_function = ft_sig / (ft_sig + np.fft.fftshift(pspectrum))
    if foreg_type == 6:
        filter_function = ft_sig / (ft_sig + np.fft.fftshift(pspectrum))
    if foreg_type == 5:
        grf_fg = np.fft.fft2(grf) * foreground(l, 1) ** 0.5 * 1e-3  # in Kelvin
        grf_fg_II = np.fft.fft2(grf_II) * foreground(l, 2) ** 0.5 * 1e-3
        grf_fg_III = np.fft.fft2(grf_III) * foreground(l, 3) ** 0.5 * 1e-3
        grf_fg_IV = np.fft.fft2(grf_IV) * foreground(l, 4) ** 0.5 * 1e-3
        grf_fg_LCDM = np.fft.fft2(grf_LCDM) * foreground(l, 6) ** 0.5 * 1e-3
        del grf, grf_II, grf_III, grf_IV
        del l, mag_k, kx, ky
    else:
        grf_fg = grf * pspectrum ** 0.5 * 1e-3  # in Kelvin

    '''
    Interfeometer noise
    '''
    real_part = np.sqrt(0.5 * Pl1) * np.random.normal(loc=0., scale=1., size=(N, N))
    imaginary_part = np.sqrt(0.5 * Pl1) * np.random.normal(loc=0., scale=1., size=(N, N))
    noise_ps_inst = (real_part + imaginary_part * 1.0j)
    ps_inst_shift = np.fft.fftshift(Pl1)
    '''
    create the middle bin
    '''
    alphas = random_alpha()
    '''
    create the other bins
    '''
    grf_all, redshifts = random_bins(fg_normalize(grf_fg, 1), fg_normalize(grf_fg_II, 2), fg_normalize(grf_fg_III, 3), fg_normalize(grf_fg_IV, 4), alphas, z_bins)
    del alphas
    '''
    '''
    if foreg_type==5:
        grf_norm_fg = (grf_all + fg_normalize(grf_fg_LCDM, 6) + eps_noise*noise_ps_inst*1e-3)* 1e3 * -delta_z  #noise_ps_inst*1e-3
        #fg_normalize(grf_fg, 1) + fg_normalize(grf_fg_II, 2) + fg_normalize(grf_fg_III, 3) + fg_normalize(grf_fg_IV, 4)
    else:
        grf_norm_fg = np.fft.fftshift(fg_normalize(grf_fg, fg_type)*1e3*-delta_z*epsilon_fgr)

    if foreg_type==5:
        grf_norm_fg_new = remove_fg(grf_norm_fg, redshifts)


    ft_signal = (ft_sig + grf_norm_fg) * filter_function
    ft = grf_norm_fg * filter_function

    #TODO: Foreground removal and shift
    reduc = 1#e1
    ft_ordered = ft*reduc
    ft_ordered_signal = ft_signal*reduc
    threepoint = 0
    threepoint_signal = 0
    for k in range(1, N):
        for l in range(1, N):
            #if l==256 and k==256:
            if 254<l<258 and 254<k<258:
                continue
            if ps_inst_shift[k,l]==0:
                continue
            threepoint += ft_ordered[k][l] * ft_ordered[N - k][N - l] * ft_ordered[N - l][k]
            threepoint_signal += ft_ordered_signal[k][l] * ft_ordered_signal[N - k][N - l] * ft_ordered_signal[N - l][k]
    threepoint_average_r[j] = (threepoint / (N-1) ** 2).real
    threepoint_average_i[j] = (threepoint / (N-1) ** 2).imag
    threepoint_average_signal_r[j] = (threepoint_signal / (N-1) ** 2).real
    threepoint_average_signal_i[j] = (threepoint_signal / (N-1) ** 2).imag


def combine_complex(a, b):
    dummy = []
    for i in range(0, len(a)):
        if np.abs(a[i]+1j*b[i]) < 50000:
            dummy.append(a[i]+1j*b[i])
    return dummy

def rfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    n_half = n // 2 + 1
    results = np.arange(0, n_half, dtype=int)
    return results * val


'''
removal
'''

def random_alpha():
    alphas = np.array([np.ones((N, N))*2.8, np.ones((N, N))*2.07, np.ones((N, N))*2.15, np.ones((N, N))*2.1])
    for m in range(0, N):
        for n in range(0, N):
            alpha1_plus = -np.random.normal(0, 0.1)
            alpha2_plus = -np.random.normal(0, 0.1)
            alpha3_plus = -np.random.normal(0, 0.1)
            alpha4_plus = -np.random.normal(0, 0.1)
            alphas[0, m, n] += -alpha1_plus
            alphas[1, m, n] += -alpha2_plus
            alphas[2, m, n] += -alpha3_plus
            alphas[3, m, n] += -alpha4_plus
    return alphas

def random_bins(fg1, fg2, fg3, fg4, alphaa, number_z_bins):
    dummy = []
    all_grf_bins = []
    for i in range(1, int(number_z_bins/2)+1):
        dummy.append(z_wake + i*delta_z)
        dummy.append(z_wake - i*delta_z)
    dummy.append(z_wake)
    redshifts = np.sort(np.array(dummy)) # Middle of the redshift bins
    for j in range(0, len(redshifts)):
        grf_bin_j = np.zeros((N, N))+1J*np.zeros((N, N))
        for m in range(0, len(grf_bin_j)):
            for n in range(0, len(grf_bin_j)):
                grf_bin_j[m,n] = fg1[m, n]*(1420/(1+z_wake)*1/120)**2.8*(1420/(1+redshifts[j])*1/120)**-alphaa[0,m,n] + fg2[m, n]*(1420 / (1 + z_wake) * 1 / 151)**2.07*(1420/(1+redshifts[j])*1/151)**-alphaa[1,m,n] + fg3[m, n]*(1420 / (1 + z_wake) * 1 / 120)**2.15*(1420/(1+redshifts[j])*1/120)**-alphaa[2,m,n] + fg4[m, n]*(1420 / (1 + z_wake) * 1 / (2 * 1e3))**2.1*(1420/(1+redshifts[j])*1/(2* 1e3))**-alphaa[3,m,n]
        all_grf_bins.append(grf_bin_j)
    all_grf_bins=np.array(all_grf_bins)
    return all_grf_bins, redshifts

def fit_function(z, a, b):
    return b*(1+z)**a

def remove_fg(all_fields, redshift):
    real_all_fields = []
    for i in range(0, len(all_fields)):
        real_all_fields.append(np.fft.ifft2(all_fields[i]).real)
    real_all_fields = np.array(real_all_fields)
    grf_mid = real_all_fields[int(z_bins/2)+1]
    dummy_x = redshift
    dummy_y = np.zeros(len(redshift))
    for m in range(0, N):
        for n in range(0, N):
            for o in range(0, len(dummy_x)):
                dummy_y[o] = real_all_fields[o, m, n]
            plt.plot(dummy_x, dummy_y, 'o')
            plt.show()
            pars, cov = curve_fit(f=fit_function, xdata=dummy_x, ydata=dummy_y, bounds=(0, 5))
            grf_mid[m,n] = grf_mid[m, n] - fit_function(z_wake, pars[0], pars[1])
            if m==0 and n==0:
                plt.plot(dummy_x, dummy_y, 'o')
                plt.plot(dummy_x, fit_function(dummy_x, pars[0], pars[1]))
                plt.show()

    return 0





'''program start'''

n = 1
parts = 1
z_bins = 10
foreg_type = 5
eps_fg = 0.09
eps_noise = 1#0.1
#print('Without noise ps, only pixel restriction')
print('N = '+str(n))
print('angle = '+ str(patch_angle)+' with '+str(N)+' pixel')
print('foreground removal '+ str(eps_fg))
print('noise removal ' + str(eps_noise))
print('gradient included: no')
print('G\mu = ' + str(gmu_6))
print('NOISE I')
#print('NO FILTER!')

threepoint_average_r = multiprocessing.Array('d', range(n))
threepoint_average_i = multiprocessing.Array('d', range(n))
threepoint_average_signal_r = multiprocessing.Array('d', range(n))
threepoint_average_signal_i = multiprocessing.Array('d', range(n))
LCDM_ps = np.load('angular_ps_12.npy')
Inter_ps = np.load('pinst_12_MWA_II.npy')
Inter_ps_u = np.load('u_cut.npy')
shape = [N, N]
lpix = 360.0 / patch_angle
lx = rfftfreq(shape[0]) * shape[0] * lpix
ly = np.fft.fftfreq(shape[0]) * shape[0] * lpix
l_inst = np.sqrt(lx[np.newaxis, :] ** 2 + ly[:, np.newaxis] ** 2)
Pl = Pinst(l_inst, 0)
Pl1 = Pinst(l_inst, 1)


threepoint_average = []#np.ndarray(np.zeros(n), dtype=complex)
threepoint_average_signal = []#np.ndarray(np.zeros(n), dtype=complex)
for k in range(0, parts):
    processes = []
    for i in range(int(k*n/parts), int((k+1)*n/parts)):
        p = multiprocessing.Process(target=multiprocessing_fun, args=(i, threepoint_average_r, threepoint_average_i, threepoint_average_signal_r, threepoint_average_signal_i, foreg_type))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    for process in processes:
        process.terminate()
    del processes
threepoint_average = np.array(combine_complex(np.array(threepoint_average_r), np.array(threepoint_average_i)))
threepoint_average_signal = np.array(combine_complex(np.array(threepoint_average_signal_r), np.array(threepoint_average_signal_i)))
print('Without signal: ')
print('\mu = '+ str(np.abs(np.mean(threepoint_average))))
print('\sigma = '+ str(np.abs(np.std(threepoint_average))))
print('With signal: ')
print('\mu =' + str(np.abs(np.mean(threepoint_average_signal))))
print('\sigma =' + str(np.abs(np.std(threepoint_average_signal))))
plt.hist(np.array(threepoint_average).real, bins=100)
plt.vlines(0, 0, 4000, colors='r')
#plt.savefig('test_3PF.png', dpi=400)
plt.clf()
plt.hist(np.array(threepoint_average_signal).real, bins=100)
plt.vlines(0, 0, 4000, colors='r')
#plt.savefig('test_3PF_with_sign.png', dpi=400)






#Plots
'''plt.xlabel('degree')
plt.ylabel('degree')
my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.imshow(np.fft.ifft2(ft).real, interpolation='none')
plt.show()
plt.savefig('test_GRF_power.png', dpi=400)'''