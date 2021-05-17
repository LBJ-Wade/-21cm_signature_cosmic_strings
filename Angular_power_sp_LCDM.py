import numpy as np
import math
import PyCosmo as pyco
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants
import multiprocessing
cosmo = pyco.Cosmo()
cosmo.set(pk_type = 'BBKS')





z = 12
z_wake=z
#redshift string formation
z_i = 1000
#frequency bin: 40kHz = 0.04 MHz
delta_f = 0.02
#thickness redshift bin (assuming we look at f in [f_0, f_0 + delta_f])
delta_z = -delta_f/(1420)*(z+1)
patch_size = 512
patch_angle = 5.
angle_per_pixel = patch_angle/patch_size
T_back2 = 0.1 * 0.62*1e-3/(0.33*1e-4) * np.sqrt((0.26 + (1+z_wake)**-3 * (1-0.26-0.042))/0.29)**-1 * (1+z_wake)**0.5/2.5**0.5


def ps(k, l):
    return T_back2**2 * (1 + (0.7*(1+z)**3*(cosmo.background.H_a(a=1.)**2/cosmo.background.H_a(a=1./(1+z))**2))**0.55*(k/np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))**2)**2  * cosmo.lin_pert.powerspec_a_k(a=1/(1+z), k=np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))


def multi_fn(j, dummy):
    dummy[j] = 1/(math.pi * cosmo.background.dist_rad_a(1/(1+z)) * cosmo.background.dist_rad_a(1/(1+z+delta_z)) ) * integrate.quad(lambda k: 1 * ps(k, j), 0, 40)[0] #np.cos((cosmo.background.dist_rad_a(1/(1+z)) - cosmo.background.dist_rad_a(1/(1+z+delta_z)))*k)


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
print(mag_k[0])
k=np.linspace(1,10000,1000)#=np.sort(mag_k[1])
plt.loglog(k, ps(0,k))
#plt.xscale('log')
plt.show()
init_angular = angular_ps(180*mag_k.max()/math.pi+1)
np.save('angular_ps_12', init_angular)
ps_LCDM = def_ang_ps(mag_k, init_angular)




##################




'''
z = 30 #redshift
T_sky = 60 * 1e3 * (1420/((1.+z) * 300))**-2.5 #in mK
T_inst = 0#T_inst= T_receiver is suppressed. https://arxiv.org/pdf/1604.03751.pdf
T_sys = T_sky + T_inst #temperature
N_d = 256 #numper of tiles, 1 tile are 16 antennas, effective area per tile see below, for MWA II: https://core.ac.uk/download/pdf/195695824.pdf
N_p = 0 #number of pointings: N_p Omega_p = 4 Pi f_sky
A_e = 14.5 #effective total dish area, source: The EoR sensitivity of the Murchison Widefield Array
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
    return (((1420.*1e6/(1+z))**-1 * scipy.constants.c)**2 * T_sys**2 * N_p)/(n_u(k) * t_tot * d_nu * A_e**2)


def rfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    n_half = n // 2 + 1
    results = np.arange(0, n_half, dtype=int)
    return results * val


def foregroung(l):
    dummy = np.zeros((l.shape[0], l.shape[1]))
    for i in range(0,len(dummy)):
        for j in range(0,len(dummy[0])):
            if l[i][j]<30:
                dummy[i][j] = 1100 * (1100. / (30)) ** 3.3 * (130. ** 2 / 1420. ** 2) ** 2.8 * (
                            1 + 30) ** (2 *2.8)  # (1. / (a + 1.) * ((1. + 30) ** (a + 1.) - (1. + 30 -0.008) ** (a + 1.))) ** 2
            else:
                dummy[i][j] = 1100 * (1100. / (l[i][j]+1)) ** 3.3 * (130. ** 2 / 1420. ** 2) ** 2.8 * (1+30)**(2*2.8)#(1. / (a + 1.) * ((1. + 30) ** (a + 1.) - (1. + 30 -0.008) ** (a + 1.))) ** 2
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
    Pl = foregroung(l )#grf2, grf1, grf3)

    real_part = np.sqrt(0.5* Pl) * np.random.normal(loc=10., scale=1., size=l.shape) * lpix / (2.0 * np.pi)
    imaginary_part = np.sqrt(0.5*Pl) * np.random.normal(loc=10., scale=1., size=l.shape) * lpix / (2.0 * np.pi)

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


a = (300*1e-6)*1/0.75 * (1.2*150*1e-3/115)**-2* ((25)**2/100)**-1 *(1420/(1+z)/120)**-2.07
print(a)
print(57 * (1100. / (30)) ** 1.1 * (130. ** 2 / 1420. ** 2) ** 2.07 * (
                            1 + z) ** (2 *2.07))'''