#!/usr/bin/env python
# coding: utf-8

# In[59]:


from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from functools import partial
from itertools import combinations
import numpy as np
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import pchip
from scipy.integrate import simps
import scipy.constants
import matplotlib.pyplot as plt

from astropy import coordinates as coords
from astropy import units

from hirax_tools.utils import HARTRAO_COORD, pointing, lmn_coordinates


class HIRAXArrayConfig(object):
    """
    Class for representing HIRAX array configurations
    """
    @classmethod
    def from_n_elem_compact(cls, Ndish, spacing,
                            altitude=90, azimuth=180, Ddish=6,
                            reference_coordinate=None):
        """
        Creates a HIRAXArrayConfig object assuming a compact square array
        given a number of dishes and a grid spacing.

        Parameters
        ----------
        Ndish : int
            Number of dishes. Should be a square otherwise it will be square-rooted and
            rounded down.
        spacing : number, `~astropy.units.Quantity`
            Dish spacing in meters unless a Quantity object with equivalent units.
        altitude : number, `~astropy.coordinates.Angle`
            Array pointing altitude in degrees or an Angle object. Default: 90 deg
        azimuth : number, `~astropy.coordinates.Angle`
            Array pointing azimuth in degrees or an Angle object. Default: 180 deg
        Ddish : number, `~astropy.units.Quantity`
            Dish diameter in meters unless a Quantity object with equivalent units.
            Default: 6 m
        reference coordinate : `~astropy.coordinates.EarthLocation` object or None
            The reference coordinate for the array, defining the location of the
            reference (SE) dish. Only necessary for pointing calculations. If None
            certain methods will raise ValueError exceptions.
            Default: None

        """
        spacing = units.Quantity(spacing, unit=units.m)
        Nside = np.ceil(np.sqrt(Ndish)).astype(int)
        # Reference dish is the SE dish
        d_ew = -spacing*(np.arange(Ndish) % Nside)
        d_ns = spacing*(np.arange(Ndish) // Nside)
        return cls(d_ew, d_ns, altitude=altitude, azimuth=azimuth, Ddish=Ddish,
                   reference_coordinate=reference_coordinate)

    @classmethod
    def from_n_elem_footprint(cls, Ndish, box_size,
                              altitude=90, azimuth=180, Ddish=6,
                              reference_coordinate=None):
        """
        Creates a HIRAXArrayConfig object assuming a compact square array
        given a number of dishes and a footprint size.

        Parameters
        ----------
        Ndish : int
            Number of dishes. Should be a square otherwise it will be square-rooted and
            rounded down.
        box_size : number, `~astropy.units.Quantity`
            Length of square side of array footprint in meters unless a Quantity
            object with equivalent units.
        altitude : number, `~astropy.coordinates.Angle`
            Array pointing altitude in degrees or an Angle object. Default: 90 deg
        azimuth : number, `~astropy.coordinates.Angle`
            Array pointing azimuth in degrees or an Angle object. Default: 180 deg
        Ddish : number, `~astropy.units.Quantity`
            Dish diameter in meters unless a Quantity object with equivalent units.
            Default: 6 m
        reference coordinate : `~astropy.coordinates.EarthLocation` object or None
            The reference coordinate for the array, defining the location of the
            reference (SE) dish. Only necessary for pointing calculations. If None
            certain methods will raise ValueError exceptions.
            Default: None
        """
        Nside = np.ceil(np.sqrt(Ndish)).astype(int)
        box_size = units.Quantity(box_size, unit=units.m)
        d_ew, d_ns = np.meshgrid(
            np.linspace(-box_size/2, box_size/2, Nside),
            np.linspace(-box_size/2, box_size/2, Nside))
        return cls(d_ew.ravel(), d_ns.ravel(), altitude=altitude, azimuth=azimuth, Ddish=Ddish,
                   reference_coordinate=reference_coordinate)

    def __init__(self, d_ew, d_ns, altitude=90, azimuth=180, Ddish=6,
                 reference_coordinate=None, sort_coords=False):
        """
        Initialise a HIRAXArrayConfig object, specifying the relative EW and NS
        coordinates of each dish in the array.

        Parameters
        ----------
        d_ew : array-like, `~astropy.units.Quantity`
            Relative East-West coordinates of dishes comprising the array in meters unless
            a Quantity object with equivalent units.
        d_ns : array-like, `~astropy.units.Quantity`
            Relative North-South coordinates of dishes comprising the array in meters unless
            a Quantity object with equivalent units.
        altitude : number, `~astropy.coordinates.Angle`
            Array pointing altitude in degrees or an Angle object. Default: 90 deg
        azimuth : number, `~astropy.coordinates.Angle`
            Array pointing azimuth in degrees or an Angle object. Default: 180 deg
        Ddish : number, `~astropy.units.Quantity`
            Dish diameter in meters unless a Quantity object with equivalent units.
            Default: 6 m
        reference coordinate : `~astropy.coordinates.EarthLocation` object or None
            The reference coordinate for the array, defining the location of the
            reference (SE) dish. Only necessary for pointing calculations. If None
            certain methods will raise ValueError exceptions.
            Default: None
        sort_coords : boolean
            If True, sort input dish coordinates from East to West, then North to
            South. If False, leave in input order.
            Default: False
        """
        self.Ddish = units.Quantity(Ddish, unit=units.m)
        self.d_ew = units.Quantity(d_ew, unit=units.m)
        self.d_ns = units.Quantity(d_ns, unit=units.m)
        self.n_dish = len(self.d_ns) # Could do some checks here
        self.altitude = coords.Angle(altitude, unit=units.degree)
        self.azimuth = coords.Angle(azimuth, unit=units.degree)
        self.baseline_pairs = np.array([(a, b) for a, b in combinations(range(len(self.d_ew)), 2)])
        self.reference_coordinate = reference_coordinate
        if sort_coords:
            self.sort_coords()

    def sort_coords(self):
        sort_inds = np.lexsort((-self.d_ew, self.d_ns))
        self.d_ew = self.d_ew[sort_inds]
        self.d_ns = self.d_ns[sort_inds]

    def uv_redundancy(self, frequency=600*units.MHz, sort='ascending',
                      tolerance=10):
        """
        Return the unique u and v coordinates from the given array configuration
        along with their degree of redundancy, ie. the number of baselines that have
        those same u and v separation.

        Parameters
        ----------
        frequency : number or `~astropy.units.Quantity`, optional
            frequency in MHz or Quantity object with equivalent units.
            Default: 600 MHz
        sort : string, 'ascending' or 'descending', optional
            How to sort output by redundancy.
            Default: 'ascending'
        tolerance : int, number of significant figures to use for redundancy
            calculation.
            Default : 10
        Returns
        -------
        tuple
            (unique_u, unique_v, redundant_counts) the unique u and v coordinates and the
            counts of the number of baselines contributing to each of these unique coordinates.
        """

        full_u, full_v, _ = self.uvw_coords(frequency=frequency)
        uv_for_unique = np.round(full_u, tolerance) + 1j*np.round(full_v, tolerance)

        uniq_uv, inverse, redun_counts = np.unique(uv_for_unique, return_counts=True, return_inverse=True)
        uniq_u, uniq_v = uniq_uv.real, uniq_uv.imag

        sort_inds = np.argsort(redun_counts)
        if sort.lower() == 'descending':
            sort_inds = sort_inds[::-1]
        elif sort.lower() != 'ascending':
            raise ValueError("'sort' must be either 'ascending' or 'descending'.")
        return uniq_u[sort_inds], uniq_v[sort_inds], redun_counts[sort_inds], inverse[sort_inds]

    def uvw_coords(self, frequency=600*units.MHz, baseline_pairs=None):
        """
        Return the u, v, w coordinates of all or a subset of baseline pairs
        calculated at a specific frequency.

        Parameters
        ----------
        frequency : number or `~astropy.units.Quantity`, optional
            frequency in MHz or Quantity object with equivalent units.
            Default: 600 MHz
        baseline_pairs : array-like (optional)
            Npairs x 2 array of indices for pairs of dishes to calculate u, v, w
            coordinates for. If None (default), calculate u, v, w coordinates for
            all combinations of dishes.

        Returns
        -------
        tuple
            (u, v, w) coordinate arrays, length Npairs (default Ndish*(Ndish-1)/2)
        """
        frequency = units.Quantity(frequency, unit=units.MHz)
        lamb = frequency.to('m', equivalencies=units.spectral())
        if baseline_pairs is None:
            baseline_pairs = self.baseline_pairs
        else:
            baseline_pairs = np.atleast_2d(baseline_pairs)

        du = ((self.d_ew[baseline_pairs[:, 0]] - self.d_ew[baseline_pairs[:, 1]])/lamb).to('').value
        dv = ((self.d_ns[baseline_pairs[:, 0]] - self.d_ns[baseline_pairs[:, 1]])/lamb).to('').value
        u = du*np.cos(self.azimuth.rad)
        v = dv*np.sin(self.altitude.rad)
        w = np.sqrt((du*np.sin(self.azimuth.rad))**2 + (dv*np.cos(self.altitude.rad))**2)
        return u, v, w

    def fov(self, frequency=600*units.MHz):
        """
        Return the field of view of the instrument in steradians

        FoV = (wavelength/Ddish)**2

        Parameters
        ----------
        frequency : number or `~astropy.units.Quantity` (optional)
            frequency in MHz or Quantity object with equivalent units.
            Default: 600 MHz
        Returns
        -------
        `~astropy.units.Quantity`
            Quantity object with the FoV in steradians.
        """
        frequency = units.Quantity(frequency, unit=units.MHz)
        lamb = frequency.to('m', equivalencies=units.spectral())
        return ((lamb/self.Ddish)**2).to('sr', equivalencies=units.dimensionless_angles())

    def u_range(self, frequency=600*units.MHz):
        """
        Return the valid range of u/v magnitudes with an appropriate spacing
        defined by the FoV.

        Parameters
        ----------
        frequency : number or `~astropy.units.Quantity` (optional)
            frequency in MHz or Quantity object with equivalent units.
            Default: 600 MHz
        Returns
        -------
        tuple : 
            (u_min, u_max, du), the minimum, maximum and step size for the u/v magnitude range
        """
        frequency = units.Quantity(frequency, unit=units.MHz)
        # definitely faster ways to do this for less general arrays...
        us, vs, _ = self.uvw_coords(frequency=frequency)
        mag_us = np.sqrt(us**2 + vs**2)
        return np.min(mag_us), np.max(mag_us), 1/(self.fov(frequency=frequency).value**.5)

    def baseline_density_spline(self, frequency=600*units.MHz):
        """
        Calculate a spline representation of the radial u/v magnitude density, n(|u|).

        Parameters
        ----------
        frequency : number or `~astropy.units.Quantity` (optional)
            frequency in MHz or Quantity object with equivalent units.
            Default: 600 MHz
        Returns
        -------
        `~scipy.interpolation.InterpolatedUnivariateSpline`
           spline object for the baseline density as a function of u/v magnitude
        """
        frequency = units.Quantity(frequency, unit=units.MHz)
        us, vs, _ = self.uvw_coords(frequency=frequency)

        # Add reflected points to make symmetrical
        us = np.concatenate((us, -us))
        vs = np.concatenate((vs, -vs))

        mag_us = np.sqrt(us**2 + vs**2)

        umin, umax, du = self.u_range(frequency=frequency)
        us = np.arange(umin, umax, du)
        hist_bins = np.arange(umin-du/2, umax+du/2, du)
        hist, _ = np.histogram(mag_us, bins=hist_bins)

        hist = np.array(hist)/2. # Because we double counted

        density = hist/(2*np.pi*us*du)
        plt.plot(us, density)

        s_spline = 20
        k_spline = 2
        spl = spline(us, density, ext=1, k=k_spline, s=s_spline)
        plt.plot(us, spl(us))
        #plt.ylim(0, 0.1)
        plt.show()

        '''spl = spline(us, density, k=k_spline, s=s_spline)
        plt.plot(us, density)
        plt.plot(us, spl(us))
        #plt.ylim(-0.1,0.1)
        plt.show()'''


        return spline(us, density, ext=1, k=k_spline, s=s_spline)

    def nu(self, fid_freq=600*units.MHz, normalize=True):
        """
        Return the frequency dependent baseline density, n(u)

        Parameters
        ----------
        fid_freq : number or `~astropy.units.Quantity` (optional)
            The frequency at which to calculate the baseline density.
            The results should not depend on this strongly.
            Default: 600 MHz
        normalize : boolean (optional)
            If True (default), normalize the output so that the integral of
            the discretized baseline density equals the number of baseline
        Returns
        -------
        u : array
           The u values
        n_u : array
            The corresponding n(u) values
        """
        fid_freq = units.Quantity(fid_freq, unit=units.MHz)
        u_min, u_max, du = self.u_range(frequency=fid_freq)
        us = np.arange(u_min, u_max, du) + du
        spl = self.baseline_density_spline(frequency=fid_freq)
        n_u = spl(us)
        '''Non-negativity condition'''
        for i in range(1500, len(n_u)):
            n_u[i] = n_u[1500]

        if normalize:
            N = len(self.d_ew)
            nbl = N*(N-1)/2
            norm = nbl/simps(spl(us)*2*np.pi*us, us)
        else:
            norm = 1

        return us, norm*n_u


def rfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    n_half = n // 2 + 1
    results = np.arange(0, n_half, dtype=int)
    return results * val


def Pinst( n_u):
    return (((1420.*1e6/(1+z))**-1 * scipy.constants.c)**2 * T_sys**2 * N_p)/((n_u) * t_tot * d_nu * A_e**2)



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

    Pl = Pinst(l)

    real_part = np.sqrt(0.5* Pl) * np.random.normal(loc=0., scale=1., size=l.shape) * lpix / (2.0 * np.pi)
    imaginary_part = np.sqrt(0.5*Pl) * np.random.normal(loc=0., scale=1., size=l.shape) * lpix / (2.0 * np.pi)

    # Get map in real space and return
    ft_map = (real_part + imaginary_part*1.0j) * l.shape[0] ** 2

    ft_map[0, 0] = 0.0

    return np.fft.irfft2(ft_map).real


#https://arxiv.org/pdf/1206.6945.pdf
z = 12 #redshift
T_sky = 60 * 1e3 * (1420/((1.+z) * 300))**-2.5 #in mK
T_inst = 100*1e3 #T_inst= T_receiver is suppressed. https://arxiv.org/pdf/1604.03751.pdf
T_sys = T_sky + T_inst #temperature
N_d = 256 #numper of tiles, 1 tile are 16 antennas, effective area per tile see below, for MWA II: https://core.ac.uk/download/pdf/195695824.pdf
N_p = 100 #number of pointings: N_p Omega_p = 4 Pi f_sky
A_e = 21.5 #effective total dish area, source: The EoR sensitivity of the Murchison Widefield Array
t_tot = 1000*3600 #total integration time: 100h
d_nu = 0.05*1e6 #bandwidth in Hz in that channel: 50kHz

#read in MWA config, Phase I + II
antennafile = open('256T_update.txt', 'r')
dist_ns = []
dist_ew = []
for line in antennafile.readlines():
    fields = line.split(',')
    dist_ns.append(float(fields[2]))
    dist_ew.append(float(fields[3]))
antennafile.close()

# Make an array lyout of a 32x32 element array.
array_conf = HIRAXArrayConfig(d_ns=dist_ns, d_ew=dist_ew, Ddish=2*np.sqrt(A_e/np.pi))

u, nu = array_conf.nu(fid_freq=1420./(1+z))

'''
introducing a cut off for the spline
'''
marker =0
nu_cut = []
u_cut = []
for i in range(0,len(nu)):
    if marker ==0:
        if nu[i]>0:
            nu_cut.append(nu[i])
            u_cut.append(u[i])
            continue
        else:
            marker=1
    else:
        continue

'''
final result
'''

u_cut = np.array(u_cut)
nu_cut = np.array(nu_cut)
print(u_cut)

'''
save the result
'''
np.save('nu_cut', nu_cut)
np.save('u_cut', u_cut)

ps = Pinst(nu_cut)
np.save('pinst_12_MWA_II', ps)
