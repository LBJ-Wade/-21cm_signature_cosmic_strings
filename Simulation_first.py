'''First draft of the simulations in the context of a masters thesis
with Prof. Robert Brandenberger'''
'''This programm is meant to be used as a foundation for further simulations'''

'''Copyrights to David Maibach'''

import numpy as np
import math
import matplotlib.pyplot as plt

#define constants according to arXiv: 1006.2514v3
#according to The Astrophysical Journal, 622:1356-1362, 2005 April 1, Table 2. Units [cm^3 s^-1]
def deexitation_crosssection(t_k):
    if 0<t_k and t_k<1.5:
        return 1.38e-13
    if 1.5 < t_k and t_k < 3:
        return 1.43e-13
    if 3 < t_k and t_k < 5:
        return 2.71e-13
    if 5 < t_k and t_k < 7:
        return 6.60e-13
    if 7 < t_k and t_k < 9:
        return 1.47e-12
    if 9 < t_k and t_k < 12.5:
        return 2.88e-12
    if 12.5 < t_k and t_k < 17.5:
        return 9.1e-12
    if 17.5 < t_k and t_k < 22.5:
        return 1.78e-11
    if 22.5 < t_k and t_k < 27.5:
        return 2.73e-11
    if 27.5 < t_k and t_k < 35:
        return 3.67e-11
    if 35 < t_k and t_k < 45:
        return 5.38e-11
    if 45 < t_k and t_k < 55:
        return 6.86e-11
    if 55 < t_k and t_k < 65:
        return 8.14e-11
    if 65 < t_k and t_k < 75:
        return 9.25e-11
    if 75 < t_k and t_k < 85:
        return 1.02e-10
    if 85 < t_k and t_k < 95:
        return 1.11e-10
    if 95 < t_k and t_k < 150:
        return 1.19e-10
    if 150 < t_k and t_k < 250:
        return 1.75e-10
    if 250 < t_k and t_k < 350:
        return 2.09e-10
    else:
        print('T_K is out of scope for the deexcitation fraction')
        return 0
#redshift probing
z = 100
#redshift string formation
z_i = 1000
#thickness redshift bin
delta_z = 0
#string tension in units of [10^-6]
gmu_6 = 0.3
#string speed
vsgammas_square = 1./3
#temperature of HI atoms inside the wake [K]
T_K = 20*gmu_6*vsgammas_square*(z_i+1.)/(z+1)
#CMB temperature [K]
T_gamma = 2.725*(1+z)
#background numberdensity hydrogen [cm^-3]
nback=1.9e-7 *(1.+z)**3
#collision coeficcient hydrogen-hydrogen (density in the wake is 4* nback, Delta E for hyperfine is 0.068 [K], A_10 = 2.85e-15 [s^-1])
xc = 4*nback*deexitation_crosssection(T_K)* 0.068/(2.85e-15 *T_gamma)
#wake brightness temperature [K]
T_b = 0.07* xc/(xc+1.)*(1-T_gamma/T_K)*np.sqrt(1.+z)
#fraction of baryonc mass comprised of HI
xHI = 0 #TODO: find out xHI
#background temperature [K] (assume Omega_b, h, Omega_Lambda, Omega_m as in arXiv: 1405.1452[they use planck collaboration 2013b best fit])
T_back = 0.19055e-3 * (0.049*0.67*(1.+z)**2 * xHI)/np.sqrt(0.316*(1.+z)**3 + 0.684)
#TODO:Average over a redshift bin


#define quantities of noise and the patch of the sky
#patch properties
patch_size = 256
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
#wake properties
wake_brightness = 1
wake_size_angle = 1 #in degree
shift_wake_angle = [0, 0]
#Gaussian noise properties
sigma_noise = 1
mean_noise = 1
power_law = -2.

#define function for a string signal (assuming it is about wake_size_angle deg x wake_size_angle deg in size)
def stringwake_PS(size, intensity, anglewake, angleperpixel, shift):
    #coordinate the dimensions of wake and shift
    patch = np.zeros((size, size))
    shift_pixel = np.zeros(2)
    shift_pixel[0] = int(np.round(shift[0]/angleperpixel))
    shift_pixel[1] = int(np.round(shift[1]/angleperpixel))
    wakesize_pixel = int(np.round(anglewake/angleperpixel))
    print(size/2+shift_pixel[0]-wakesize_pixel/2)
    print(size/2+shift_pixel[0]+wakesize_pixel/2+1)
    for i in range(int(size/2+shift_pixel[0]-wakesize_pixel/2), int(size/2+shift_pixel[0]+wakesize_pixel/2+1)):#Todoo: make sure its an integer is not necessary because integer division
        for j in range(int(size/2+shift_pixel[1]-wakesize_pixel/2), int(size/2+shift_pixel[1]+wakesize_pixel/2+1)):
            patch[i, j] = intensity
    return patch


#define function for generating a Fourierspace in every point of the patch we are looking at
#def fftIndgen(n):
#    a = range(0, n/2+1)
#    b = range(1, n/2)              #####not the right method if we want to fit frequency and space
#    b.reverse()
#    b = [-i for i in b]
#    return a + b


#define function that generates a gaussian random field of size patch_size
def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100, sigma = 1, mean = 0, angleperpixel = 1 ):
    #create a two dimensional projected power spectrum
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    #create the noise
    noise = np.fft.fft2(np.random.normal(mean, sigma, size = (size, size)))
    #calculate its amplitude. Note that in this form the k modes are defined as in units [1/degree]
    amplitude = np.zeros((size, size))
    for i, kx in enumerate(np.fft.fftfreq(size, angle_per_pixel*2*math.pi)):
        for j, ky in enumerate(np.fft.fftfreq(size, angle_per_pixel*2*math.pi)):
            amplitude[i, j] = Pk2(kx, ky)
    #add your signal
    return np.fft.ifft2(noise * amplitude + np.fft.fft2(stringwake_PS(patch_size, wake_brightness, wake_size_angle,
                                                                   angle_per_pixel, shift_wake_angle)))

#define a function for your chi^2 -statistics
def chi_square():
    return 0

#Plot the GRF map for a given size for different power spectra
for alpha in [-4.0, -3.0, -0.0]:
    out = gaussian_random_field(Pk = lambda k: k**alpha, size = patch_size, sigma = sigma_noise, mean = mean_noise, angleperpixel = angle_per_pixel)
    plt.figure()
    plt.xlabel('degree')
    plt.ylabel('degree')
    my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
    plt.xticks([0, 51, 102, 128, 154, 205, 255], my_ticks)
    plt.yticks([0, 51, 102, 128, 154, 205, 255], my_ticks)
    plt.imshow(out.real, interpolation='none')
    plt.show()
    #plt.savefig('test_GRF_with'+str(np.abs(alpha))+'.png', dpi=400)

