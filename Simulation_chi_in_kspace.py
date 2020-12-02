'''First draft of the simulations in the context of a masters thesis
with Prof. Robert Brandenberger'''
'''This programm is meant to be used as a foundation for further simulations'''

'''Copyrights to David Maibach'''

import numpy as np
import math
import matplotlib.pyplot as plt
'''Section 1: Define constants, dimensions, signal brightness, and noise properties'''
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
#redshift interval probing #TODO: average all redshift dependent quantities over the redshift bin
z = 30
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
#fraction of baryonc mass comprised of HI. Given that we consider redshifts of the dark ages, we can assume that all the
#hydrogen of the universe is neutral and we assume the mass fraction of baryoni is:
xHI = 0.75
#background temperature [K] (assume Omega_b, h, Omega_Lambda, Omega_m as in arXiv: 1405.1452[they use planck collaboration 2013b best fit])
T_back = (0.19055e-3) * (0.049*0.67*(1.+z)**2 * xHI)/np.sqrt(0.267*(1.+z)**3 + 0.684)

#define quantities of noise and the patch of the sky
#patch properties
patch_size = 64
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
#Gaussian noise properties, alpha noise according to arXiv:2010.15843
alpha_noise = 0.0475
sigma_noise = T_back*alpha_noise
mean_noise = T_back
power_law = -0.0
#wake properties
wake_brightness = sigma_noise #T_b
wake_size_angle = 1 #in degree
shift_wake_angle = [0, 0]






'''Section 2: We define functions that define our signal, and the gausian random field'''
#define function for a string signal (assuming it is about wake_size_angle deg x wake_size_angle deg in size)


def power_spectrum(k, alpha=-2.):
    out = k**alpha
    out[k == 0] = 0.
    return out


def mag_k(k_x, k_y):
    return np.sqrt(k_x**2+k_y**2)

#define our signal in a real space patch with matched dimensions
def stringwake_ps(size, intensity, anglewake, angleperpixel, shift):
    #coordinate the dimensions of wake and shift
    patch = np.zeros((size, size))
    shift_pixel = np.zeros(2)
    shift_pixel[0] = int(np.round(shift[0]/angleperpixel))
    shift_pixel[1] = int(np.round(shift[1]/angleperpixel))
    wakesize_pixel = int(np.round(anglewake/angleperpixel))
    for i in range(int(size/2+shift_pixel[0]-wakesize_pixel/2), int(size/2+shift_pixel[0]+wakesize_pixel/2+1)):#Todoo: make sure its an integer is not necessary because integer division
        for j in range(int(size/2+shift_pixel[1]-wakesize_pixel/2), int(size/2+shift_pixel[1]+wakesize_pixel/2+1)):
            patch[i, j] = intensity
    return patch


#define function that generates a gaussian random field of size patch_size
def gaussian_random_field(size = 100, sigma = 1, mean = 0, alpha = -1.0 ):
    #create the noise
    noise_real = np.random.normal(mean, sigma, size = (size, size))
    noise = np.fft.fft2(noise_real)
    #calculate its amplitude. Note that in this form the k modes are defined as in units [1/degree]
    amplitude = np.zeros((size, size))
    for i, kx in enumerate(np.fft.fftfreq(size, angle_per_pixel*2*math.pi)):
        for j, ky in enumerate(np.fft.fftfreq(size, angle_per_pixel*2*math.pi)):
            amplitude[i, j] = np.sqrt(power_spectrum(mag_k(kx, ky), alpha))
    return np.fft.ifft2(noise * amplitude)


#define function that generates a gaussian random field of size patch_size with my signal
def gaussian_random_field_with_signal(size = 100, sigma = 1, mean = 0, angleperpixel = 1. , alpha =-1.0):
    #create the noise
    noise = np.fft.fft2(np.random.normal(mean, sigma, size = (size, size)))
    #calculate its amplitude. Note that in this form the k modes are defined as in units [1/degree]
    amplitude = np.zeros((size, size))
    for i, kx in enumerate(np.fft.fftfreq(size, angleperpixel*2*math.pi)):
        for j, ky in enumerate(np.fft.fftfreq(size, angleperpixel*2*math.pi)):
            amplitude[i, j] = np.sqrt(power_spectrum(mag_k(kx, ky), alpha))
    #add your signal
    return np.fft.ifft2(noise * amplitude + np.fft.fft2(stringwake_ps(patch_size, wake_brightness, wake_size_angle,
                                                                      angleperpixel, shift_wake_angle)))


'''Section 3: Define a methode that calculates the chi^2 in fourier space'''
def chi_square(data_sample_real):
    data_ft = np.fft.fft2(data_sample_real)
    data_power_spectrum = np.abs(data_ft)**2





'''Section 4: We apply the methods introduced before in various ways.'''

#calculate the chi^2 statistic for n datasamples in fourier space
#number of sample
n = 10
memory_chi = np.zeros(n)
memory_chi_signal = np.zeros(n)
#calculate the chi_square statistics n times
for h in range(0, n):
    alpha_fun = power_law
    out = gaussian_random_field(size=patch_size, sigma=sigma_noise, mean=mean_noise, alpha=alpha_fun)
    out1 = gaussian_random_field_with_signal(size=patch_size, sigma=sigma_noise, mean=mean_noise, angleperpixel=angle_per_pixel,
                                             alpha=alpha_fun)

print(np.mean(memory_chi))
print(np.mean(memory_chi_signal))

#Plot the GRF map for a given size for different power spectra
'''for alpha in [-2.0]:
    out = gaussian_random_field_with_signal(Pk = lambda k: k**alpha, size = patch_size, sigma = sigma_noise, mean = mean_noise, angleperpixel = angle_per_pixel, alpha=alpha)
    plt.figure()
    plt.xlabel('degree')
    plt.ylabel('degree')
    my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
    plt.xticks([0, 51, 102, 128, 154, 205, 255], my_ticks)
    plt.yticks([0, 51, 102, 128, 154, 205, 255], my_ticks)
    plt.imshow(out.real, interpolation='none')
    plt.show()
    #plt.savefig('test_GRF_with'+str(np.abs(alpha))+'.png', dpi=400)'''

