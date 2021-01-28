import numpy as np
import math
import PyCosmo as pyco
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants
import multiprocessing
cosmo = pyco.Cosmo()
cosmo.set(pk_type = 'BBKS')



'''y_unfiltered = [0.10038, 0.0112, 0.0003, 0.00011, 0.00053, 0.0020, 0.0071, 0.0093, 0.0079, 0.0052, 0.0004,0.0005, 0.0055, 0.0094, 0.01487]
y = np.array(y_unfiltered)*10
print(y)
plt.plot([0.9*1e-8, 0.3*1e-7, 0.6*1e-7, 0.9*1e-7, 0.15*1e-6, 0.225*1e-6, 0.3*1e-6, 0.375*1e-6, 0.4*1e-6, 0.45e-6, 0.6*1e-6, 0.7*1e-6, 0.9*1e-6, 1*1e-6, 1.2*1e-6],[1.872, 0.5863, 0.043, 0.0118, 0.0045, 0.1263, 0.3873, 0.4832, 0.4235, 0.2895, 0.0132, 0.0381, 0.2966, 0.4388 ,0.6613],'bs')
plt.plot([0.9*1e-8, 0.3*1e-7, 0.6*1e-7, 0.9*1e-7, 0.15*1e-6, 0.225*1e-6, 0.3*1e-6, 0.375*1e-6, 0.4*1e-6, 0.45e-6, 0.6*1e-6, 0.7*1e-6, 0.9*1e-6, 1*1e-6, 1.2*1e-6],y,'g^')
my_ticks = ['$0.5 \cdot 10^{-7}$' ,'$0.5 \cdot 10^{-6}$','$0.1 \cdot 10^{-5}$']
plt.xticks([ 0.5*1e-7,0.5*1e-6,0.1*1e-5], my_ticks)
plt.xlabel('$G\mu$')
plt.ylabel('$ \delta \chi^2$')
plt.show()

z=[-200.1, -149.5, -37.11, -16.7425, -17.77, -65.58, -126.435, -133.998, -124.965, -100.27, -13.652, 34.0307, 101.819, 126.27, 160.079]
x= np.zeros(len(z))
plt.plot([0.9*1e-8, 0.3*1e-7, 0.6*1e-7, 0.9*1e-7, 0.15*1e-6, 0.225*1e-6, 0.3*1e-6, 0.375*1e-6, 0.4*1e-6, 0.45e-6, 0.6*1e-6, 0.7*1e-6, 0.9*1e-6, 1*1e-6, 1.2*1e-6],z,'ro')
plt.plot([0.9*1e-8, 0.3*1e-7, 0.6*1e-7, 0.9*1e-7, 0.15*1e-6, 0.225*1e-6, 0.3*1e-6, 0.375*1e-6, 0.4*1e-6, 0.45e-6, 0.6*1e-6, 0.7*1e-6, 0.9*1e-6, 1*1e-6, 1.2*1e-6],x,'--')

my_ticks = ['$0.5 \cdot 10^{-7}$' ,'$0.5 \cdot 10^{-6}$','$0.1 \cdot 10^{-5}$']
plt.xticks([ 0.5*1e-7,0.5*1e-6,0.1*1e-5], my_ticks)
plt.xlabel('$G\mu$')
plt.ylabel('$ \delta T_b^{wake} \,\,\,[mK]$')
plt.show()
'''

######################


'''
z = 30
z_wake=z
#redshift string formation
z_i = 1000
#frequency bin: 40kHz = 0.04 MHz
delta_f = 0.04
#thickness redshift bin (assuming we look at f in [f_0, f_0 + delta_f])
delta_z = -delta_f/(1420)*(z+1)
patch_size = 512
patch_angle = 5.
angle_per_pixel = patch_angle/patch_size
T_back2 = 0.1 * 0.62*1e-3/(0.33*1e-4) * np.sqrt((0.26 + (1+z_wake)**-3 * (1-0.26-0.042))/0.29)**-1 * (1+z_wake)**0.5/2.5**0.5


def ps(k, l): #TODO: substitute 10 with the brightness temperature
    return T_back2**2 * (1 + (0.7*(1+z)**3*(cosmo.background.H_a(a=1.)**2/cosmo.background.H_a(a=1./(1+z))**2))**0.55*(k/np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))**2)**2  * cosmo.lin_pert.powerspec_a_k(a=1/(1+z), k=np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))


def multi_fn(j, dummy):
    dummy[j] = 1/(math.pi * cosmo.background.dist_rad_a(1/(1+z)) * cosmo.background.dist_rad_a(1/(1+z+delta_z)) ) * integrate.quad(lambda k: np.cos((cosmo.background.dist_rad_a(1/(1+z)) - cosmo.background.dist_rad_a(1/(1+z+delta_z)))*k) * ps(k, j), 0, 40)[0]


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
    ps_CDM = np.zeros((len(k[1]), len(k[1])))
    for i in range(0, len(k[1])):
        for j in range(0, len(k[1])):
            l = 360 * k[i][j] / (2 * math.pi)
            l_bottom = math.floor(l)
            l_top = l_bottom + 1
            delta_l = l - l_bottom
            if l_bottom == 0:
                if l < 0.1:
                    ps_CDM[i][j] = init_angular[0]
                else:
                    ps_CDM[i][j] = init_angular[l_bottom] + delta_l*(init_angular[l_top]-init_angular[l_bottom])
            else:
                ps_CDM[i][j] = init_angular[l_bottom] + delta_l*(init_angular[l_top]-init_angular[l_bottom])
    return ps_CDM

kx, ky = np.meshgrid( 2 * math.pi * np.fft.fftfreq(patch_size, angle_per_pixel),
                         2 * math.pi * np.fft.fftfreq(patch_size, angle_per_pixel))
mag_k = np.sqrt(kx ** 2 + ky ** 2)
init_angular = angular_ps(180*mag_k.max()/math.pi+1)
np.save('angular_ps_30', init_angular)
ps_LCDM = def_ang_ps(mag_k, init_angular)
noise_real = np.random.normal(0, 1, size = (patch_size, patch_size))
noise = np.fft.fft2(noise_real)
grf = np.fft.ifft2(noise * ps_LCDM**0.5).real
np.save('grf_LCDAM', grf)'''




##################



'''
#https://arxiv.org/pdf/1206.6945.pdf
z = 30 #redshift
T_sky = 60 * 1e3 * (1420/((1.+z) * 300))**-2.5 #in mK
T_inst = 100#T_inst= T_receiver is suppressed. https://arxiv.org/pdf/1604.03751.pdf
T_sys = T_sky + T_inst #temperature
N_d = 256 #numper of tiles, 1 tile are 16 antennas, effective area per tile see below, for MWA II: https://core.ac.uk/download/pdf/195695824.pdf
N_p = 0 #number of pointings: N_p Omega_p = 4 Pi f_sky
A_e = 21.5 #effective total dish area, source: The EoR sensitivity of the Murchison Widefield Array
D_min =14 #smallest baseling in m
D_max = 5300 #longest baseline in m for MWA II: "The Phase II Murchison Widefield Array: Design overview"
t_tot = 1000*3600 #total integration time: 1000h
d_nu = 0.01*1e6 #bandwidth in that channel: 10kHz
#FOV: 25° -->?
#D_max: 5300

def n_u(k): #trivial approximation for unknown baseline distributions
    return N_d * (N_d-1)/(2 * math.pi * ((D_max/((1420.*1e6/(1+z))**-1 * scipy.constants.c))**2 - (D_min/((1420.*1e6/(1+z))**-1 * scipy.constants.c))**2))


#test intererometer noise power spectrum
def pspec_inter(k):
    return (((1420.*1e6/(1+z))**-1 * scipy.constants.c)**2 * T_sys**2 * N_p)/(n_u(k) * t_tot * d_nu * A_e**2)'''


###############################



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




N = 512
patch_size = N
c = 5./512
angle_per_pixel =c
z = 25
####################
foreground_type = 5
####################
T_back2 = 0.1 * 0.62*1e-3/(0.33*1e-4) *np.sqrt((0.26 + (1+z)**-3 * (1-0.26-0.042))/0.29)**-1 * (1+z)**0.5/2.5**0.5
z_i = 1000
#frequency bin: 10kHz = 0.01 MHz
delta_f = 0.02
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
theta1 =  math.pi*0.32 #angle 1 in z-space
theta2 = 0 #angle 2 in z-space
T_b = 1e3* 0.07  *xc/(xc+1.)*(1-T_gamma/T_K)*np.sqrt(1.+z_wake)*(2*np.sin(theta1)**2)**-1
wake_brightness = T_b* 1e3 #in mK
wake_thickness = 24 * math.pi/15 * gmu_6 * 1e-6 * vsgammas_square**0.5 * (z_i+1)**0.5 * (z_wake + 1.)**0.5 *2.*np.sin(theta1)**2*1/np.cos(theta1)
rot_angle_uv =0# math.pi/4 #rotation angle in the uv plane
wake_size_angle = [1., 1.] #in degree
shift_wake_angle = [0, 0]


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




def fg_normalize(grf_fg, fg_type):#TODO: Integrate over redshift bin
    if fg_type == 1:
        mean, std, std_eff = 253*(1420/(1+z_wake)*1/120)**-2.8, 1.3*(1420/(1+z_wake)*1/120)**-2.8, 69
    if fg_type == 2:
        mean, std, std_eff = 38.6*(1420/(1+z_wake)*1/151)**-2.07, 2.3*(1420/(1+z_wake)*1/151)**-2.07, 1410
    if fg_type == 3:
        mean, std, std_eff = 2.2*(1420/(1+z_wake)*1/120)**-2.15, 0.05*(1420/(1+z_wake)*1/120)**-2.15, 415
    if fg_type == 4:
        mean, std, std_eff = 1e-4*(1420/(1+z_wake)*1/(2*1e3))**-2.1, 1e-5*(1420/(1+z_wake)*1/(2*1e3))**-2.1, 81
    sum = 0
    for i in range(0, len(grf_fg)):
        for j in range(0, len(grf_fg)):
            sum += np.abs(grf_fg[i, j]) ** 2
    sum = sum - grf_fg[0, 0] ** 2
    norm = np.sqrt(patch_size ** 4 * std ** 2 * 1 / sum).real
    grf_fg = norm * grf_fg
    grf_fg[0][0] = mean * patch_size ** 2
    return grf_fg, std_eff, norm




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
    if l[1].ndim == 0:
       dummy = np.zeros(len(l))
       for i in range(0, len(dummy)):
           if l[i] < 1:
               dummy[i] = A * (1100. / (30)) ** beta * (130. ** 2 / 1420. ** 2) ** alpha * (1 + z_wake) ** (2 * alpha)
           else:
               dummy[i] = A * (1100. / (l[i] + 1)) ** beta * (130 ** 2 / 1420 ** 2) ** alpha * (1 + z_wake) ** (
                           2 * alpha)  # (1. / (a + 1.) * ((1. + 30) ** (a + 1.) - (1. + 30 -0.008) ** (a + 1.))) ** 2
       return dummy
    else:
        dummy = np.zeros((N, N))
        for i in range(0,len(l)):
            for j in range(0,len(l)):
                if l[i][j]<1:
                    dummy[i][j] = A * (1100. / (30)) ** beta * (130. ** 2 / 1420. ** 2) ** alpha * (1 + z_wake) ** (2 * alpha)
                else:
                    dummy[i][j] = A * (1100. / (l[i][j]+1)) ** beta * (130 ** 2 / 1420 ** 2) ** alpha* (1+z_wake)**(2*alpha)#(1. / (a + 1.) * ((1. + 30) ** (a + 1.) - (1. + 30 -0.008) ** (a + 1.))) ** 2
        return dummy


def rfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    n_half = n // 2 + 1
    results = np.arange(0, n_half, dtype=int)
    return results * val


def LCDM(l):
    dummy = np.zeros((l.shape[0], l.shape[1]))
    for i in range(0, len(dummy)):
        for j in range(0, len(dummy[0])):
            l_bottom = math.floor(l[i][j])
            l_top = l_bottom + 1
            delta_l = l[i,j] - l_bottom
            if l_bottom == 0:
                if l[i][j] < 0.1:
                    dummy[i][j] = LCDM_ps[0]
                else:
                    dummy[i][j] = LCDM_ps[l_bottom] + delta_l * (LCDM_ps[l_top] - LCDM_ps[l_bottom])
            else:
                dummy[i][j] = LCDM_ps[l_bottom] + delta_l * (LCDM_ps[l_top] - LCDM_ps[l_bottom])
    return dummy


def GRF_generator(ang, shape, seed=None):
    """
    Generates a GRF with given power_func, shape (must be square), and ang (deg)
    """

    if seed is not None:
        np.random.seed(seed)

    lpix = 360.0 / ang
    lx = rfftfreq(shape[0]) * shape[0] * lpix
    ly = np.fft.fftfreq(shape[0]) * shape[0] * lpix

    # Compute the multipole moment of each FFT pixel
    l = np.sqrt(lx[np.newaxis, :] ** 2 + ly[:, np.newaxis] ** 2)
    #grf1 = np.random.normal(2.7, 0.4, size=(l.shape[0], l.shape[1]))
    #grf2 = np.random.normal(2.55, 0.1, size=(l.shape[0], l.shape[1]))
    #grf3 = np.abs(np.random.normal(1, 0.25, size=(l.shape[0], l.shape[1])))
    Pl = LCDM(l)

    real_part = np.sqrt(0.5* Pl) * np.random.normal(loc=0., scale=1., size=l.shape) * lpix / (2.0 * np.pi)
    imaginary_part = np.sqrt(0.5*Pl) * np.random.normal(loc=0., scale=1., size=l.shape) * lpix / (2.0 * np.pi)

    # Get map in real space and return
    ft_map = (real_part + imaginary_part*1.0j) * l.shape[0] ** 2

    ft_map[0, 0] = 0.0

    return np.fft.irfft2(ft_map).real


def GRF_spec(kappa, l_edges, ang):
    """
    calculates the power spectrum of a square convergence maps with size ang (deg) binned in l_edges
    """

    # Get the pixels
    n_pix = np.shape(kappa)[0]

    # get the fourier transform
    fft_abs = np.abs(np.fft.fft2(kappa)) ** 2

    # Get physical pixsize
    lpix = 360.0 / ang

    # Get the norm
    norm = ((ang * np.pi / 180.0) / n_pix ** 2) ** 2

    # Get the modes
    lx = np.array([min(i, n_pix - i) * lpix for i in range(n_pix)])
    ly = np.array([i * lpix for i in range(n_pix)])

    l = np.sqrt(lx[:, np.newaxis] ** 2 + ly[np.newaxis, :] ** 2)

    # Cycle through bins
    power_spectrum = []
    for bins in range(len(l_edges) - 1):
        Pl = np.where(np.logical_and(l > l_edges[bins], l <= l_edges[bins + 1]), fft_abs, 0.0)
        Pl = np.sum(Pl) / Pl[Pl != 0].size
        power_spectrum.append(Pl)

    # normalize
    power_spectrum = norm * np.asarray(power_spectrum)

    # return l, Pl
    return l_edges[:-1] + np.diff(l_edges) / 2.0, power_spectrum


kx, ky = np.meshgrid(2 * math.pi * np.fft.fftfreq(N, c),
                             2 * math.pi * np.fft.fftfreq(N, c))
mag_k = np.sqrt(kx ** 2 + ky ** 2)
grf = np.random.normal(0., 1., size = (patch_size, patch_size))
LCDM_ps = np.load('angular_ps_25.npy')
#plt.imshow((GRF_generator(5, [512,512])-2.752*(1+z)*1e3)/(1+z)*-delta_z + np.fft.ifft2(signal_ft(patch_size, wake_size_angle,  angle_per_pixel, shift_wake_angle, False)).real)
plt.imshow((np.fft.ifft2(LCDM(180*mag_k/np.pi)**0.5*np.fft.fft2(grf)).real*360/(5.*2*np.pi)*N-2.752*(1+z)*1e3)/(1+z))
plt.colorbar()
plt.show()
plt.imshow((GRF_generator(5, [512,512])-2.752*(1+z)*1e3)/(1+z))
plt.xlabel('degree')
plt.ylabel('degree')
my_ticks = ['$-2.5\degree$', '$-1.5\degree$', '$-0.5\degree$', '$0\degree$', '$0.5\degree$', '$1.5\degree$', '$2.5\degree$']
plt.xticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
plt.yticks([0,  102,  204,  256, 308, 410, 511], my_ticks)
cbar = plt.colorbar()
cbar.set_label('$ T_b \,\,\,[$'+'mK'+'$]$', rotation=270, labelpad=20, size=11 )
plt.show()
print(np.mean(GRF_generator(5, [512,512])))
print(np.mean(np.fft.ifft2(LCDM(180*mag_k/np.pi)**0.5*np.fft.fft2(grf)).real*360/(5.*2*np.pi)*N))





######################



'''
x = np.array([0.216518,     0.313392,       0.469813,        0.708374,      1.06183,     1.59342,       2.39161,        3.58803,       5.38264,           8.07420, 12.1110,14])
y = np.array([2.5775220e+13, 7.0157119e+13, 9.7912021e+13, 2.5893861e+14, 6.9986468e+14, 2.3962459e+15, 6.8343556e+15, 1.3656459e+16, 2.4723744e+16, 3.8448373e+16, 9.1288065e+16, 1.5238e+17])

fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)
z=np.linspace(0.0001,15, 1000)
a=np.linspace(0.2, 15, 100)
axs[1].plot(x, y*10**-5, label='$P_{inst}$')
axs[1].plot(a, foreground(180*a/np.pi, 1)*1e1 , label='$P_{fg1}\cdot 10^1$')
axs[1].plot(a, foreground(180*a/np.pi, 2)*1e7 , label='$P_{fg2}\cdot 10^7$')
axs[1].plot(a, foreground(180*a/np.pi, 3)*0.5e6 , label='$P_{fg2}\cdot 0.5\cdot 10^6$')
axs[1].plot(a, foreground(180*a/np.pi, 4)*1e11 , label='$P_{fg2}\cdot 10^{11}$')

axs[1].vlines(0.216518, 2.5775220e+8, 1e13)
#axs[1].vlines(27.2505,1.0787888e+13, 3e13)
axs[1].set(ylabel= '$P(k)\,\,\,$[mK'+'$^2$]')
axs[1].set(xlabel='k $\,\,\,$ [1/degree]')
axs[0].plot(z, -88.5*1/(np.pi*z) *np.sin(np.pi*z), label='$\delta T_{signal}$')
axs[0].set(ylabel= '$\delta T_b^{wake}(k)\,\,\,$[mK]')
axs[1].axvspan( 0.216518, 12.1110, color='lightgrey')
axs[0].axvspan( 0.216518, 12.1110, color='lightgrey')
axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
plt.show()'''