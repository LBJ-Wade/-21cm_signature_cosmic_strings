'''First draft of the simulations in the context of a masters thesis
with Prof. Robert Brandenberger'''
'''This programm is meant to be used as a foundation for further simulations'''

'''Copyrights to David Maibach'''

import numpy as np
import math
import PyCosmo as pyco
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import multiprocessing
cosmo = pyco.Cosmo()
cosmo.set(pk_type = 'BBKS')

'''Section 1: Define constants, dimensions, signal brightness, and noise properties'''
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

#redshift interval probing
z = 30
#redshift string formation
z_i = 1000
#frequency bin: 15kHz = 0.015 MHz
delta_f = 0.015
#thickness redshift bin (assuming we look at f in [f_0, f_0 + delta_f])
delta_z = -delta_f/(1420)*(z+1)     #(1420/(z+1.)+delta_f)*(z+1)
print('Frequency interval: f in ['+ str(1420/(1.+z))+', '+str(1420/(1.+z)+delta_f) +'] MHz.')
print('Therefore, dz = '+ str(delta_z)+' and we cover ['+ str(z)+', '+ str(z+delta_z)+ '].')
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


#define quantities of noise and the patch of the sky
#patch properties
patch_size = 512
patch_angle = 5. #in degree
angle_per_pixel = patch_angle/patch_size
#Gaussian noise properties, alpha noise according to arXiv:2010.15843
alpha_noise = 0.0475
sigma_noise = T_back*alpha_noise
mean_noise = T_back
power_law = -2.0
#not so improtant ..


#wake properties
theta1 = 0#math.pi*0.32 #angle 1 in z-space
theta2 = 0 #angle 2 in z-space
#wake brightness temperature [K]
T_b = 1e3* 0.07  *xc/(xc+1.)*(1-T_gamma/T_K)*np.sqrt(1.+z_wake)#*(2*np.sin(theta1)**2)**-1
wake_brightness = T_b#in mK
wake_size_angle = [1., 1.] #in degree
shift_wake_angle = [0, 0]
rot_angle_uv =0# math.pi/4 #rotation angle in the uv plane

wake_thickness = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * (z_wake + 1.)**0.5 *1/np.cos(theta1)#2.*np.sin(theta1)**2
print('The string wake brightness temperature at '+str(z)+' is '+str(T_b)+' mK.')
#wakes extend in frequency space is [v_0 + wakethickness/2, v_0 - wakethickness/2]
print('The wakes thickness in redshift space is given by dz_wake = '+str(wake_thickness))
#delta_z = 2*wake_thickness
#print('The redshift bin thickness is '+str(delta_z))




'''Section 2: We define functions that define our signal, and the gausian random field'''
#define function for a string signal (assuming it is about wake_size_angle deg x wake_size_angle deg in size)


def power_spectrum(k, alpha=-2., sigma=1.):
    out = k**alpha * 1/sigma**2
    out[k < 0.01] =  (0.01) ** alpha
    return out

'''LCDM POWER SPECTRUM'''

def ps(k, l):
    return cosmo.lin_pert.powerspec_a_k(a=1/(1+z), k=np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))


def multi_fn(j, dummy):
    dummy[j] = T_back2**2 * 1/(math.pi * cosmo.background.dist_rad_a(1/(1+z)) * cosmo.background.dist_rad_a(1/(1+z+delta_z)) ) * integrate.quad(lambda k: np.cos((cosmo.background.dist_rad_a(1/(1+z)) - cosmo.background.dist_rad_a(1/(1+z+delta_z)))*k) * ps(k, j), 0, 50)[0]


def angular_ps(l_max):
    dummy = multiprocessing.Array('d', int(l_max)+1)
    for k in range(0, int((l_max+1)/1000)):
        processes = []
        for i in range(k*1000, (k+1)*1000):
            p = multiprocessing.Process(target=multi_fn, args=(i, dummy))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
        for process in processes:
            process.terminate()
        del processes
    processes = []
    for i in range(int((l_max+1)/1000) * 1000, int(l_max)+1):
        p = multiprocessing.Process(target=multi_fn, args=(i, dummy))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    for process in processes:
        process.terminate()
    del processes

    return np.array(dummy)


def def_ang_ps(k, init_angular):
    if k[1].ndim == 0:
        ps_CDM = np.zeros(len(k))
        for i in range(0, len(k)):
            l = 360 * k[i][j] / (2 * math.pi)
            l_bottom = math.floor(l)
            l_top = l_bottom + 1
            delta_l = l - l_bottom
            if l_bottom == 0:
                if l < 0.01:
                    ps_CDM[i] = init_angular[0]
                else:
                    ps_CDM[i] = init_angular[l_bottom] + delta_l * (init_angular[l_top] - init_angular[l_bottom])
            else:
                ps_CDM[i] = init_angular[l_bottom] + delta_l * (init_angular[l_top] - init_angular[l_bottom])
        return ps_CDM
    else:
        ps_CDM = np.zeros((len(k[1]), len(k[1])))
        for i in range(0, len(k[1])):
            for j in range(0, len(k[1])):
                l = 360 * k[i][j] / (2 * math.pi)
                l_bottom = math.floor(l)
                l_top = l_bottom + 1
                delta_l = l - l_bottom
                if l_bottom == 0:
                   if l < 0.01:
                        ps_CDM[i][j] = init_angular[0]
                   else:
                       ps_CDM[i][j] = init_angular[l_bottom]+ delta_l*(init_angular[l_top]-init_angular[l_bottom])
                else:
                     ps_CDM[i][j] = init_angular[l_bottom]+ delta_l*(init_angular[l_top]-init_angular[l_bottom])
        return ps_CDM

'''######################'''

def instrumental_ps(): #TODO: Implement as additional Gaussian  noise
    #TODO: Blow of implementieren
    return 1


#deep21 arXiv:2010.15843       A  beta  alpha Xi   type
#--------------------------------------------------------
#Galactic Synchrotron       1100, 3.3, 2.80, 4.0)   1
#Point Sources                57, 1.1, 2.07, 1.0)   2
#Galactic free-free        0.088, 3.0, 2.15, 32.)   3
#Extragalactic free-free   0.014, 1.0, 2.10, 35.)   4
def foreground_power_spectrum(k, A_pure, beta, a, Xi, sigma): # Xi):
    #an example from arXiv:2010.15843 (deep21)
    lref = 1100.
    A = A_pure*1e-6
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
                    ps[i] = 1/delta_z**2*A * (lref / 1) ** beta * (vref ** 2 / 1420 ** 2) ** a * (1. / (a + 1.) * ((1. + z) ** (a + 1.) - (1. + z + delta_z) ** (a + 1.))) ** 2
                else:
                    ps[i] = 1/delta_z**2*A * (lref / 1) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a + 1.) * ((1. + z)**(a + 1.) - (1. + z + delta_z)**(a+1.))) ** 2
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
                        ps[i][j] = 1/delta_z**2*A * (lref / 1) ** beta * (vref ** 2 / 1420 ** 2) ** a *(1. / (a + 1.) * ((1. + z) ** (a + 1.) - (1. + z + delta_z) ** (a + 1.))) ** 2 #((1+z_wake)**2)**a
                    else:
                        ps[i][j] = 1/delta_z**2*A * (lref/1)**beta * (vref**2/1420**2)**a * (1./(a+1.) * ((1.+z)**(a+1.) - (1. + z + delta_z)**(a+1.)))**2 #((1+z_wake)**2)**a
                else:
                    ps[i][j] = 1/delta_z**2*A * (lref / l_top) ** beta * (vref ** 2 / 1420**2) ** a * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+ z + delta_z)**(a+1.)))**2 + delta_l * (A * (lref/l_bottom)**beta * (vref**2/(1420**2))**a - A * (lref/l_top)**beta * (vref**2/(1420**2))**a) * (1./(a+1.) * ((1.+z)**(a+1.) - (1.+z+delta_z)**(a+1.)))**2  #exp()
                    #ps[i][j] = A * (lref / l_top) ** beta * (vref ** 2 / 1420 ** 2) ** a *((1+z_wake)**2)**a + delta_l * (
                    #                       A * (lref / l_bottom) ** beta * (vref ** 2 / (1420 ** 2)) ** a - A * (
                    #                           lref / l_top) ** beta * (vref ** 2 / (1420 ** 2)) ** a) *((1+z_wake)**2)**a
        #print(np.mean(ps)**0.5)
        return ps/sigma**2


#define our signal in a real space patch w
# ith matched dimensions .... Not in use when considereing k in [1/degree]
def stringwake_ps(size, anglewake, angleperpixel, shift, background_on):
    #coordinate the dimensions of wake and shift
    patch = np.zeros((size, size))
    if background_on == True:
        patch = np.ones((size, size)) * T_back
    patch_rotated = np.zeros((size, size))
    dz_wake = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * (z_wake + 1.)**0.5 *2.*np.sin(theta1)**2/np.cos(theta1)
    df_wake = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * 1/(z_wake + 1.)**0.5 * 1420.*2.*np.sin(theta1)**2/np.cos(theta1) # MHz. THe 2sin^2 theta cancels when multiplied with T_b
    #print(-dz_wake/delta_z)
    #print(df_wake/delta_f)
    shift_pixel = np.zeros(2)
    shift_pixel[0] = int(np.round(shift[0]/angleperpixel))
    shift_pixel[1] = int(np.round(shift[1]/angleperpixel))
    wakesize_pixel = [int(np.round(np.cos(theta1)*anglewake[0]/angleperpixel)), int(np.round(anglewake[1]/angleperpixel))] #theta1 term added depending on the direction of rot
    i_x = int(size/2.+shift_pixel[0]-wakesize_pixel[0]/2.)
    f_x = int(size/2.+shift_pixel[0]+wakesize_pixel[0]/2.+1)
    i_y = int(size/2.+shift_pixel[1]-wakesize_pixel[1]/2.)
    f_y = int(size/2.+shift_pixel[1]+wakesize_pixel[1]/2.+1)
    #print(df_wake/delta_f)
    for i in range(i_x, f_x):
        for j in range(i_y, f_y):
            #patch[i, j] = 1e3 * 1/(2.*np.sin(theta1)**2) * df_wake/delta_f * (i-i_x)*1./(f_x-i_x) * T_b # according to https://arxiv.org/pdf/1403.7522.pdf
            patch[i, j] += (1e3*0.07 * (2*np.sin(theta1)**2)**-1* xc/(xc+1.)*2./3*((1 + z_wake + dz_wake/2. * (i-i_x)*1./(f_x-i_x))**1.5-(1 + z_wake - dz_wake/2. * (i-i_x)*1./(f_x-i_x))**1.5) - 1e3*0.07*  (2*np.sin(theta1)**2)**-1 * xc/(xc+1.)*2.725/(20 * gmu_6**2 * vsgammas_square * (z_i+1.)) * 2/7. * ((1 + z_wake + dz_wake/2. * (i-i_x)*1./(f_x-i_x))**3.5-(1 + z_wake - dz_wake/2. * (i-i_x)*1./(f_x-i_x))**3.5)) #in mK
    #print(str(patch[f_x-1,f_y-1])+ ' signal ')
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
        return patch_rotated
    return patch #we assume here, the wake is centered in the middle of the redshift bin


#define function that generates a gaussian random field of size patch_size
def gaussian_random_field(size = 100, sigma = 1., mean = 0, alpha = -1.0):
    #create the noise
    noise = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    kx, ky = np.meshgrid(2*math.pi*np.fft.fftfreq(size, angle_per_pixel), 2*math.pi*np.fft.fftfreq(size, angle_per_pixel))
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    grf = noise * power_spectrum(mag_k, alpha, sigma)**0.5
    for i in range(0, size):
        ky[0][i] = 0.001
    for i in range(0, size):
        kx[i][0] = 0.001
    ft_signal = -1/delta_z*wake_thickness * T_b * (
                1 / (math.pi * kx * 180./math.pi) * 1 / (math.pi * ky* 180./math.pi) * np.sin(math.pi * kx * wake_size_angle[0]) *
                np.sin(math.pi * ky * wake_size_angle[1]))
    ft_signal = ft_signal+T_back
    return grf, mag_k, ft_signal


#define function that generates a gaussian random field of size patch_size with my signal
def gaussian_random_field_with_signal(size = 100, sigma = 1., mean = 0., angleperpixel = 1. , alpha =-1.0):
    #create the noise
    noise = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    kx, ky = np.meshgrid(2*math.pi*np.fft.fftfreq(size, angle_per_pixel),
                         2*math.pi*np.fft.fftfreq(size, angle_per_pixel))
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    for i in range(0, size):
        ky[0][i] = 0.001
    for i in range(0, size):
        kx[i][0] = 0.001
    ft_signal = -1/delta_z*wake_thickness * T_b * (1 / (math.pi * kx* 180./math.pi) * 1 / (math.pi * ky* 180./math.pi) * np.sin(math.pi * kx * wake_size_angle[0]) *
                               np.sin(math.pi * ky * wake_size_angle[1]))
    ft_signal = ft_signal + T_back

    grf = noise * power_spectrum(mag_k, alpha, sigma) ** 0.5 + ft_signal #np.fft.fft2(stringwake_ps(patch_size,
                                                                                           #    wake_size_angle,
                                                                                            #   angleperpixel,
                                                                                             #  shift_wake_angle, True))).real
    return grf, mag_k, ft_signal


#define a function the generates a foreground
def grf_foreground(type, size, sigma):
    noise = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    noise1 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    noise2 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    noise3 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    kx, ky = np.meshgrid(2*math.pi*np.fft.fftfreq(size, angle_per_pixel),
                         2*math.pi*np.fft.fftfreq(size, angle_per_pixel))
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    for i in range(0, size):
        ky[0][i] = 0.001
    for i in range(0, size):
        kx[i][0] = 0.001
    ft_signal = -1/delta_z*wake_thickness * T_b * (
                1 / (math.pi * kx* 180./math.pi) * 1 / (math.pi * ky* 180./math.pi) * np.sin(math.pi * kx * wake_size_angle[0]) *
                np.sin(math.pi * ky * wake_size_angle[1]))
    ft_signal = ft_signal + T_back
    if type == 1:
        grf = noise * foreground_power_spectrum(mag_k, 1100, 3.3, 2.80, 4.0, sigma)**0.5
        return grf, mag_k, ft_signal
    if type == 2:
        grf = noise * foreground_power_spectrum(mag_k, 57, 1.1, 2.07, 1.0, sigma)**0.5
        return grf, mag_k, ft_signal
    if type == 3:
        grf = noise * foreground_power_spectrum(mag_k, 0.088, 3.0, 2.15, 32., sigma) ** 0.5
        return grf, mag_k, ft_signal
    if type == 4:
        grf = noise * foreground_power_spectrum(mag_k, 0.014, 1.0, 2.10, 35., sigma) ** 0.5
        return grf, mag_k, ft_signal
    if type == 5:
        grf = (noise * foreground_power_spectrum(mag_k, 0.014, 1.0, 2.10, 35., sigma) ** 0.5 + noise1 *
                                    foreground_power_spectrum(mag_k, 0.088, 3.0, 2.15, 32., sigma) ** 0.5 + noise2 *
                           foreground_power_spectrum(mag_k, 57, 1.1, 2.07, 1.0, sigma) ** 0.5 + noise3 *
                           foreground_power_spectrum(mag_k, 1100, 3.3, 2.80, 4.0, sigma) ** 0.5)
        return grf, mag_k, ft_signal



#define a function the generates a foreground with the string signal included
def grf_foreground_signal(type, size, sigma):
    noise = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    noise1 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    noise2 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    noise3 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (size, size)) + 1j * np.random.normal(0, 1, size = (size, size)))
    kx, ky = np.meshgrid(2*math.pi*np.fft.fftfreq(size, angle_per_pixel),
                         2*math.pi*np.fft.fftfreq(size, angle_per_pixel))
    mag_k = np.sqrt(kx ** 2 + ky ** 2)
    for i in range(0, size):
        ky[0][i] = 0.001
    for i in range(0, size):
        kx[i][0] = 0.001
    ft_signal = -1/delta_z*wake_thickness * T_b * (
                1 / (math.pi * kx* 180./math.pi) * 1 / (math.pi * ky* 180./math.pi) * np.sin(math.pi * kx * wake_size_angle[0]) *
                np.sin(math.pi * ky * wake_size_angle[1]))
    ft_signal = ft_signal + T_back
    if type == 1:
        grf = noise * foreground_power_spectrum(mag_k, 1100, 3.3, 2.80, 4.0, sigma)**0.5 + ft_signal#np.fft.fft2(stringwake_ps(patch_size,
                                                                                           #    wake_size_angle,
                                                                                            #   angleperpixel,
                                                                                             #  shift_wake_angle, True))).real
        return grf, mag_k, ft_signal
    if type == 2:
        grf = noise * foreground_power_spectrum(mag_k, 57, 1.1, 2.07, 1.0, sigma)**0.5 + ft_signal #np.fft.fft2(stringwake_ps(patch_size,
                                                                                           #    wake_size_angle,
                                                                                            #   angleperpixel,
                                                                                             #  shift_wake_angle, True))).real
        return grf, mag_k, ft_signal
    if type == 3:
        grf = noise * foreground_power_spectrum(mag_k, 0.088, 3.0, 2.15, 32., sigma) ** 0.5 + ft_signal #np.fft.fft2(stringwake_ps(patch_size,
                                                                                           #    wake_size_angle,
                                                                                            #   angleperpixel,
                                                                                             #  shift_wake_angle, True))).real
        return grf, mag_k, ft_signal
    if type == 4:
        grf =  noise * foreground_power_spectrum(mag_k, 0.014, 1.0, 2.10, 35., sigma) ** 0.5 + ft_signal #np.fft.fft2(stringwake_ps(patch_size,
                                                                                           #    wake_size_angle,
                                                                                            #   angleperpixel,
                                                                                             #  shift_wake_angle, True))).real
        return grf, mag_k, ft_signal
    if type == 5:
        grf =  (noise * foreground_power_spectrum(mag_k, 0.014, 1.0, 2.10, 35., sigma) ** 0.5 + noise1 *
                                    foreground_power_spectrum(mag_k, 0.088, 3.0, 2.15, 32., sigma) ** 0.5 + noise2 *
                           foreground_power_spectrum(mag_k, 57, 1.1, 2.07, 1.0, sigma)**0.5 + noise3 *
                           foreground_power_spectrum(mag_k, 1100, 3.3, 2.80, 4.0, sigma)**0.5 + ft_signal) #np.fft.fft2(stringwake_ps(patch_size,
                                                                                           #    wake_size_angle,
                                                                                            #   angleperpixel,
                                                                                             #  shift_wake_angle, True))).real
        return grf, mag_k, ft_signal


'''Section 3.1: Define a methode that calculates the chi^2 in fourier space'''


def chi_square(data_sample_real, magnitude_k, fft_signal, alpha, foreground_type, filter):
    bins = 300
    data_ft = data_sample_real
    data_ps = np.abs(data_ft)**2/patch_size**2
    k_bins = np.linspace(0.1, 0.95*magnitude_k.max(), bins)
    k_bin_cents = k_bins[:-1] + (k_bins[1:] - k_bins[:-1])/2
    digi = np.digitize(magnitude_k, k_bins) - 1
    filter_stuff = wien_filter(foreground_type, magnitude_k, fft_signal)
    data_filter = filter_stuff[1] * data_ft
    data_ps_filter = np.abs(data_filter)**2/patch_size**2
    pspec_noise = filter_stuff[0] * filter_stuff[1]**2
    binned_ps = []
    for k in range(0, digi.max()):
        binned_ps.append(np.mean(data_ps[digi == k]))
    binned_ps = np.array(binned_ps).real
    ####   for filtered data
    binned_ps_filtered = []
    for k in range(0, digi.max()):
        binned_ps_filtered.append(np.mean(data_ps_filter[digi == k]))
    binned_ps_filtered = np.array(binned_ps_filtered).real
    binned_ps_noise = []
    for i in range(0, digi.max()):
        binned_ps_noise.append(np.mean(pspec_noise[digi == i]))
    binned_ps_noise = np.array(binned_ps_noise)
    ###############
    if filter == 1:
        return np.sum(binned_ps_filtered/(binned_ps_noise*bins))
    if foreground_type == 0:
        return np.sum(binned_ps/(power_spectrum(k_bin_cents, alpha)*bins))
    if foreground_type == 1:
        return np.sum(binned_ps/(foreground_power_spectrum(k_bin_cents, 1100, 3.3, 2.80, 4.0, 1)*bins))
    if foreground_type == 2:
        return np.sum(binned_ps/(foreground_power_spectrum(k_bin_cents, 57, 1.1, 2.07, 1.0, 1)*bins))
    if foreground_type == 3:
        return np.sum(binned_ps/(foreground_power_spectrum(k_bin_cents, 0.088, 3.0, 2.15, 32., 1)*bins))
    if foreground_type == 4:
        return np.sum(binned_ps/(foreground_power_spectrum(k_bin_cents, 0.014, 1.0, 2.10, 35., 1)*bins))
    if foreground_type == 5:
        return np.sum(binned_ps/(foreground_power_spectrum(k_bin_cents, 1100, 3.3, 2.80, 4.0, 1)*bins +
                                 foreground_power_spectrum(k_bin_cents, 57, 1.1, 2.07, 1.0, 1)*bins +
                                 foreground_power_spectrum(k_bin_cents, 0.088, 3.0, 2.15, 32., 1)*bins +
                                 foreground_power_spectrum(k_bin_cents, 0.014, 1.0, 2.10, 35., 1)*bins))
    if foreground_type == 6:
        return np.sum(binned_ps / (def_ang_ps(k_bin_cents, init_angular) * bins))

'''Section 3.2: define the filter funcitons'''


def wien_filter(foreground_comp, k, fft_s):
    pspec_signal = np.abs(fft_s)**2/patch_size**2
    pspec_noise = 0
    if foreground_comp == 0:
        pspec_noise = power_spectrum(k, power_law, sigma_noise)
    if foreground_comp == 1:
        pspec_noise = foreground_power_spectrum(k, 1100, 3.3, 2.80, 4.0, 1)
    if foreground_comp == 2:
        pspec_noise = (foreground_power_spectrum(k, 57, 1.1, 2.07, 1.0, 1))
    if foreground_comp == 3:
        pspec_noise = foreground_power_spectrum(k, 0.088, 3.0, 2.15, 32., 1)
    if foreground_comp == 4:
        pspec_noise = foreground_power_spectrum(k, 0.014, 1.0, 2.10, 35., 1)
    if foreground_comp == 5:
        pspec_noise = foreground_power_spectrum(k, 1100, 3.3, 2.80, 4.0, 1) + foreground_power_spectrum(k, 57, 1.1, 2.07, 1.0, 1) + foreground_power_spectrum(k, 0.088, 3.0, 2.15, 32., 1) + foreground_power_spectrum(k, 0.014, 1.0, 2.10, 35., 1)
    if foreground_comp == 6:
        pspec_noise = ps_LCDM
    return pspec_noise, pspec_signal/(pspec_noise + pspec_signal)


def matched_filter(foreground_comp, k, fft_s):
    pspec_signal = np.abs(fft_s)**2/patch_size**2
    pspec_noise = 0
    if foreground_comp == 0:
        pspec_noise = power_spectrum(k, power_law, sigma_noise)
    if foreground_comp == 1:
        pspec_noise = foreground_power_spectrum(k, 1100, 3.3, 2.80, 4.0, 1)
    if foreground_comp == 2:
        pspec_noise = (foreground_power_spectrum(k, 57, 1.1, 2.07, 1.0, 1))
    if foreground_comp == 3:
        pspec_noise = foreground_power_spectrum(k, 0.088, 3.0, 2.15, 32., 1)
    if foreground_comp == 4:
        pspec_noise = foreground_power_spectrum(k, 0.014, 1.0, 2.10, 35., 1)
    if foreground_comp == 5:
        pspec_noise = foreground_power_spectrum(k, 1100, 3.3, 2.80, 4.0, 1) + foreground_power_spectrum(k, 57, 1.1, 2.07, 1.0, 1) + foreground_power_spectrum(k, 0.088, 3.0, 2.15, 32., 1) + foreground_power_spectrum(k, 0.014, 1.0, 2.10, 35., 1)
    if foreground_comp == 6:
        pspec_noise = ps_LCDM
    return pspec_noise, pspec_signal/pspec_noise


'''Section 4: We apply the methods introduced before in various ways.'''


#calculate chi^2 for LCDM-noise
'''
kx, ky = np.meshgrid( 2 * math.pi * np.fft.fftfreq(patch_size, angle_per_pixel),
                         2 * math.pi * np.fft.fftfreq(patch_size, angle_per_pixel))
mag_k = np.sqrt(kx ** 2 + ky ** 2)
init_angular = angular_ps(180*mag_k.max()/math.pi+1)                #critical
ps_LCDM = def_ang_ps(mag_k, init_angular)
for i in range(0, patch_size):
    ky[0][i] = 0.001
for i in range(0, patch_size):
    kx[i][0] = 0.001
ft_signal = wake_thickness * T_b * (
        1 / (math.pi * kx * 180. / math.pi) * 1 / (math.pi * ky * 180. / math.pi) * np.sin(
    math.pi * kx * wake_size_angle[0]) *
        np.sin(math.pi * ky * wake_size_angle[1]))
K = 100
chi_list_signal = []
chi_list = []
#check, if the result is achieved by random fluctuations
chi_list2 = []
#check for improvement via filtration
chi_filtered = []
for l in range(0, K):
    #out = gaussian_random_field(size=patch_size, sigma=sigma_noise, mean=mean_noise, alpha=power_law)
    #return grf, mag_k, ft_signal
    grf = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (patch_size, patch_size)) + 1j * np.random.normal(0, 1, size = (patch_size, patch_size))) * ps_LCDM ** 0.5
    grf2 = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (patch_size, patch_size)) + 1j * np.random.normal(0, 1, size = (patch_size, patch_size)))* ps_LCDM ** 0.5
    grf_with_signal = 1/np.sqrt(2)*(np.random.normal(0, 1, size = (patch_size, patch_size)) + 1j * np.random.normal(0, 1, size = (patch_size, patch_size)))* ps_LCDM ** 0.5 + ft_signal
    chi_list.append(chi_square(grf, mag_k, ft_signal, power_law, 6, 0))
    chi_list_signal.append(chi_square(grf_with_signal, mag_k, ft_signal, power_law, 6, 0))
    chi_list2.append(chi_square(grf2, mag_k, ft_signal, power_law, 6, 1))
    chi_filtered.append(chi_square(grf_with_signal, mag_k, ft_signal, power_law, 6, 1))
print('LCDM-noise check, with no removal!!!')
print('For without signal: ' + str(np.mean(chi_list)))
print('For with signal: ' + str(np.mean(chi_list_signal)))
print('For without signal and filtered: ' + str(np.mean(chi_list2)))
print('For with signal and filtered: ' + str(np.mean(chi_filtered)))
print('With and without string signal: delta chi^2 = ' + str(np.abs(np.mean(chi_list)-np.mean(chi_list_signal))))
print('With and without string signal and filtered: delta chi^2 = ' + str(np.abs(np.mean(chi_list2)-np.mean(chi_filtered))))
print('With signal, with and without filter: delta chi^2 = ' + str(np.abs(np.mean(chi_filtered)-np.mean(chi_list_signal))))
'''



#calculate the DELTAchi^2 for N datasambles in Fourier space for foregrounds
N = 100
foreground = 1
chi_list_signal = []
chi_list = []
#check, if the result is achieved by random fluctuations
chi_list2 = []
#check for improvement via filtration
chi_filtered = []
for l in range(0, N):
    #out = gaussian_random_field(size=patch_size, sigma=sigma_noise, mean=mean_noise, alpha=power_law)
    out_signal = grf_foreground_signal(foreground, patch_size, 1)
    out_check = grf_foreground(foreground, patch_size, 1)
    out_2 = grf_foreground(foreground, patch_size, 1)
    #if l ==0:
    #    plt.imshow(out_signal[0].real)
    #    plt.colorbar()
    #    plt.show()
    chi_list.append(chi_square(out_check[0], out_check[1], out_check[2], power_law, foreground, 0))
    chi_list_signal.append(chi_square(out_signal[0], out_signal[1], out_signal[2], power_law, foreground, 0))
    chi_list2.append(chi_square(out_2[0], out_2[1], out_2[2], power_law, foreground, 1))
    chi_filtered.append(chi_square(out_signal[0], out_signal[1], out_signal[2], power_law, foreground, 1))
print('Foreground noise check for foreground contaminant: '+str(foreground))
print('For without signal: ' + str(np.mean(chi_list)))
print('For with signal: ' + str(np.mean(chi_list_signal)))
print('For without signal and filtered: ' + str(np.mean(chi_list2)))
print('For with signal and filtered: ' + str(np.mean(chi_filtered)))
print('With and without string signal: delta chi^2 = ' + str(np.abs(np.mean(chi_list)-np.mean(chi_list_signal))))
print('With and without string signal and filtered: delta chi^2 = ' + str(np.abs(np.mean(chi_list2)-np.mean(chi_filtered))))
print('With signal, with and without filter: delta chi^2 = ' + str(np.abs(np.mean(chi_filtered)-np.mean(chi_list_signal))))






#Plots...
'''out = grf_foreground_signal(1, patch_size, 1)

plt.xlabel('degree')
plt.ylabel('degree')
my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.imshow(out[0].real)
plt.colorbar(label = 'mK')
plt.show()'''
#plt.savefig('test_foreground_EG_f_f.png', dpi=400)

#Plot the GRF map for a given size for different power spectra
'''for alpha in [power_law]:
    out = gaussian_random_field_with_signal( size = patch_size, sigma = sigma_noise, mean = mean_noise, angleperpixel=angle_per_pixel, alpha=alpha)
    plt.figure()
    plt.xlabel('degree')
    plt.ylabel('degree')
    my_ticks = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]
    plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
    plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
    plt.imshow(out[0], interpolation='none')
    plt.show()
    #plt.savefig('test_GRF_with'+str(np.abs(alpha))+'.png', dpi=400)'''

