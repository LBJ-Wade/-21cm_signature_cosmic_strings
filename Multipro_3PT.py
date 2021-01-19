#!/usr/bin/env python
# coding: utf-8

# In[8]:

import numpy as np
import matplotlib.pyplot as plt
import math
import multiprocessing
import warnings



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
patch_angle = 20. #in degree
angle_per_pixel = patch_angle/patch_size
c = angle_per_pixel
N = 512
z = 30

#redshift string formation
z_i = 1000
#frequency bin: 10kHz = 0.01 MHz
delta_f = 0.01
#thickness redshift bin (assuming we look at f in [f_0, f_0 + delta_f])
delta_z = -delta_f/(1420)*(z+1)
#redshift of center of wake
z_wake = z+delta_z/2
#string tension in units of [10^-6]
gmu_6 = 0.3
#string speed
vsgammas_square = 1./3
#temperature of HI atoms inside the wake [K]
T_K = 20 * gmu_6**2 * vsgammas_square * (z_i+1.)/(z_wake+1)
#CMB temperature [K]
T_gamma = 2.725*(1+z_wake)
#background numberdensity hydrogen [cm^-3]
nback=1.9e-7 *(1.+z_wake)**3
#collision coeficcient hydrogen-hydrogen (density in the wake is 4* nback, Delta E for hyperfine is 0.068 [K], A_10 = 2.85e-15 [s^-1])
xc = 4*nback*deexitation_crosssection(T_K)* 0.068/(2.85e-15 *T_gamma)
#fraction of baryonc mass comprised of HI. Given that we consider redshifts of the dark ages, we can assume that all the
#hydrogen of the universe is neutral and we assume the mass fraction of baryoni is:
xHI = 0.75
#background temperature [mK] [formula according to arXiv:2010.15843] (assume Omega_b, h, Omega_Lambda, Omega_m as in arXiv: 1405.1452[they use planck collaboration 2013b best fit])
T_back = (0.19055) * (0.049*0.67*(1.+z_wake)**2 * xHI)/np.sqrt(0.267*(1.+z_wake)**3 + 0.684)
#another option is to follow arXiv:1401.2095 (OmegaHI = 0.62*1e-3, rest from Roberts paper arXiv:1006.2514)   in mK !!!!
T_back2 = 0.1* 0.62*1e-3/(0.33*1e-4) *np.sqrt((0.26 + (1+z_wake)**-3 * (1-0.26-0.042))/0.29)**-1 * (1+z_wake)**0.5/2.5**0.5
theta1 = 0# math.pi*0.4 #angle 1 in z-space
theta2 = 0 #angle 2 in z-space
T_b = 1e3* 0.07  *xc/(xc+1.)*(1-T_gamma/T_K)*np.sqrt(1.+z_wake)#*(2*np.sin(theta1)**2)**-1, theta1=0
wake_brightness = T_b* 1e3 #in mK
wake_thickness = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * (z_wake + 1.)**0.5 *1/np.cos(theta1) #2.*np.sin(theta1)**21/np.cos(theta1), theta1=0
print('3PF evaluation!')
print('Frequency interval: f in ['+ str(1420/(1.+z))+', '+str(1420/(1.+z)+delta_f) +'] MHz.')
print('Therefore, dz = '+ str(delta_z)+' and we cover ['+ str(z)+', '+ str(z+delta_z)+ '].')
print('The string wake brightness temperature at '+str(z)+' is '+str(T_b)+' mK.')
#wakes extend in frequency space is [v_0 + wakethickness/2, v_0 - wakethickness/2]
print('The wakes thickness in redshift space is given by dz_wake = '+str(wake_thickness))


def foreground_power_spectrum(k, A_pure, beta, a, Xi, sigma): # Xi):
    #an example from arXiv:2010.15843 (deep21)
    lref = 1100.
    A = A_pure #*1e-6
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
                    ps[i] = 1/delta_z**2 *A * (lref / 1.) ** beta * (vref ** 2 / 1420 ** 2) ** a * (1. / (a + 1.) * ((1. + z) ** (a + 1.) - (1. + z + delta_z) ** (a + 1.))) ** 2
                else:
                    ps[i] = 1/delta_z**2*A * (lref / 1.) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a + 1.) * ((1. + z)**(a + 1.) - (1. + z + delta_z)**(a+1.))) ** 2
            else:
                ps[i] = 1/delta_z**2*A * (lref / l_top) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+ z + delta_z)**(a+1.)))**2 + delta_l * (A * (lref / l_bottom) ** beta * (vref ** 2 / 1420**2) ** a - A * (
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
                        ps[i][j] = 1/delta_z**2*A * (lref / 1.) ** beta * (vref ** 2 / 1420 ** 2) ** a *(1. / (a + 1.) * ((1. + z) ** (a + 1.) - (1. + z + delta_z) ** (a + 1.))) ** 2 #((1+z_wake)**2)**a
                    else:
                        ps[i][j] = 1/delta_z**2*A * (lref/1.)**beta * (vref**2/1420**2)**a * (1./(a+1.) * ((1.+z)**(a+1.) - (1. + z + delta_z)**(a+1.)))**2 #((1+z_wake)**2)**a
                else:
                    ps[i][j] = 1/delta_z**2*A * (lref / l_top) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+ z + delta_z)**(a+1.)))**2 + delta_l * (A * (lref/l_bottom)**beta * (vref**2/(1420**2))**a - A * (lref/l_top)**beta * (vref**2/(1420**2))**a) * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+z+delta_z)**(a+1.)))**2  #exp()
                    #ps[i][j] = A * (lref / l_top) ** beta * (vref ** 2 / 1420 ** 2) ** a *((1+z_wake)**2)**a + delta_l * (
                    #                       A * (lref / l_bottom) ** beta * (vref ** 2 / (1420 ** 2)) ** a - A * (
                    #                           lref / l_top) ** beta * (vref ** 2 / (1420 ** 2)) ** a) *((1+z_wake)**2)**a
        #print(np.mean(ps)**0.5)
        return ps/sigma**2




def power_spectrum(k, alpha=2, sigma=1.):
    warnings.filterwarnings("ignore")
    out = 10000 / k ** alpha * 1 / sigma ** 2
    out[k < 0.01] = 1 / (0.01) ** alpha
    return out


def signal_ft(k1, k2): #TODO: do it right
    return  -1/delta_z*wake_thickness*T_b*(
            1 / (math.pi * (k1+0.001)) * 1 / (math.pi * (k2+0.001)) * np.sin(math.pi * (k1+0.001) * 1.) *
            np.sin(math.pi * (k2+0.001) * 1.))


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
    np.random.seed(j)
    grf = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (patch_size, patch_size)) + 1j * np.random.normal(0, 1, size = (patch_size, patch_size)))
    kx, ky = np.meshgrid(np.fft.fftshift(2 * math.pi * np.fft.fftfreq(N, c)), np.fft.fftshift( 2 * math.pi * np.fft.fftfreq(N, c)))
    #print(kx[0])
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    mag_k[256][256] = 0.1
    ft_sig = signal_ft(kx, ky)
    if fg_type == 1:
        pspectrum = foreground_power_spectrum(mag_k, 1100, 3.3, 2.80, 4.0, 1)
    if fg_type == 2:
        pspectrum = foreground_power_spectrum(mag_k, 57, 1.1, 2.07, 1.0, 1)
    if fg_type == 3:
        pspectrum = foreground_power_spectrum(mag_k, 0.088, 3.0, 2.15, 32., 1)
    if fg_type == 4:
        pspectrum = foreground_power_spectrum(mag_k, 0.014, 1.0, 2.10, 35., 1)
    ft_signal = (ft_sig + grf * pspectrum ** .5)
    ft = (grf * pspectrum ** .5)
    ft_ordered = ft
    ft_ordered_signal = ft_sig
    threepoint = 0
    threepoint_signal = 0
    for k in range(1, N):
        for l in range(1, N):
            threepoint += ft_ordered[k][l] * ft_ordered[N - k -1][N - l -1] * ft_ordered[N - l -1][k]
            threepoint_signal += ft_ordered_signal[k][l] * ft_ordered_signal[N - k -1][N - l -1] * ft_ordered_signal[N - l - 1][k]
    threepoint_average_r[j] = (threepoint / N ** 2).real
    threepoint_average_i[j] = (threepoint / N ** 2).imag
    threepoint_average_signal_r[j] = (threepoint_signal / N ** 2).real
    threepoint_average_signal_i[j] = (threepoint_signal / N ** 2).imag


def combine_complex(a, b):
    dummy = []
    for i in range(0, len(a)):
        if np.abs(a[i]+1j*b[i]) < 50000:
            dummy.append(a[i]+1j*b[i])
    return dummy


n = 1
parts = 1
bins = 300
foreg_type = 2

threepoint_average_r = multiprocessing.Array('d', range(n))
threepoint_average_i = multiprocessing.Array('d', range(n))
threepoint_average_signal_r = multiprocessing.Array('d', range(n))
threepoint_average_signal_i = multiprocessing.Array('d', range(n))

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
print(np.abs(np.mean(threepoint_average)))
print('With signal: ')
print(np.abs(np.mean(threepoint_average_signal)))
plt.hist(np.array(threepoint_average).real, bins=100)
plt.savefig('test_3PF.png', dpi=400)
plt.clf()
plt.hist(np.array(threepoint_average_signal).real, bins=100)
plt.savefig('test_3PF_with_sign.png', dpi=400)






#Plots
'''plt.xlabel('degree')
plt.ylabel('degree')
my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.imshow(np.fft.ifft2(ft).real, interpolation='none')
plt.show()
plt.savefig('test_GRF_power.png', dpi=400)'''