"""
angular power spectrum of Lambda-CDM fluctuation
"""

import numpy as np
import math
import PyCosmo as pyco
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants
import multiprocessing
cosmo = pyco.Cosmo()
cosmo.set(pk_type = 'BBKS')

"""
redshift of the wake
"""

z = 12
z_wake=z

"""
redshift string formation
"""

z_i = 1000

"""
frequency bin: 40kHz = 0.04 MHz
"""

delta_f = 0.02

"""
thickness redshift bin (assuming we look at f in [f_0, f_0 + delta_f])
"""

delta_z = -delta_f/(1420)*(z+1)

"""
patch config
"""

patch_size = 512
patch_angle = 5.
angle_per_pixel = patch_angle/patch_size

"""
background temperature, see https://arxiv.org/abs/1401.2095 (C5)
"""

T_back2 = 0.1 * 0.62*1e-3/(0.33*1e-4) * np.sqrt((0.26 + (1+z_wake)**-3 * (1-0.26-0.042))/0.29)**-1 * (1+z_wake)**0.5/2.5**0.5

"""
primordial power spectrum, (C4) in https://arxiv.org/abs/1401.2095
"""

def ps(k, l):
    return T_back2**2 * (1 + (0.7*(1+z)**3*(cosmo.background.H_a(a=1.)**2/cosmo.background.H_a(a=1./(1+z))**2))**0.55*(k/np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))**2)**2  * cosmo.lin_pert.powerspec_a_k(a=1/(1+z), k=np.sqrt(k**2+l**2/((cosmo.background.dist_rad_a(1/(1+z)) + cosmo.background.dist_rad_a(1/(1+z+delta_z)))/2.)**2))

"""
Integrating over line of sight (angular power spectrum), (C6)
"""

def multi_fn(j, dummy):
    dummy[j] = 1/(math.pi * cosmo.background.dist_rad_a(1/(1+z)) * cosmo.background.dist_rad_a(1/(1+z+delta_z)) ) * integrate.quad(lambda k: 1 * ps(k, j), 0, 40)[0] #np.cos((cosmo.background.dist_rad_a(1/(1+z)) - cosmo.background.dist_rad_a(1/(1+z+delta_z)))*k)


"""
angular power spectrum for a range of l-modes
"""

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


"""
Transform list of l-modes and angular power spectrum into a patch
"""

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



"""
initializing k-space grid corresponding to the patch
"""

kx, ky = np.meshgrid( 2 * math.pi * np.fft.fftfreq(patch_size, angle_per_pixel),
                         2 * math.pi * np.fft.fftfreq(patch_size, angle_per_pixel))

mag_k = np.sqrt(kx ** 2 + ky ** 2)

"""
create and save power spectrum
"""

init_angular = angular_ps(180*mag_k.max()/math.pi+1)

np.save('angular_ps_12', init_angular)

ps_LCDM = def_ang_ps(mag_k, init_angular)

