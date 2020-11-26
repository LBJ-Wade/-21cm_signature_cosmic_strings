'''First draft of the simulations in the context of a masters thesis
with Prof. Robert Brandenberger'''

'''Copyrights to David Maibach'''

import numpy as np
import math
import matplotlib.pyplot as plt
#define constants

#define quantities of noise and the patch of the sky
patch_size = 256
power_law = -2.0
wake_brightness = 1
patch_angle = 5
angle_per_pixel = patch_angle/patch_size
wake_size_angle = 1
shift_wake_angle = [0, 0]

#define function for a string signal (assuming it is about 1 deg x 1 deg in size)
def stringwake(size,intensity, anglewake, angleperpixel,shift):
    #coordinate the dimensions of wake and shift
    patch = np.zeros((size,size))
    shift_pixel = shift/angleperpixel#TODO: make sure its an integer
    wakesize_pixel = anglewake/angleperpixel #TODO: make sure its an even integer
    for i in range(size/2+shift_pixel[0]-wakesize_pixel/2,size/2+shift_pixel[0]+wakesize_pixel+1):
        for j in range(size/2+shift_pixel[1]-wakesize_pixel/2,size/2+shift_pixel[1]+wakesize_pixel+1):
            patch[i, j] = intensity
    return patch


#define function for generating a Fourierspace in every point of the patch we are looking at
def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

#define function that generates a gaussian random field of size patch_size
def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    #create a two dimensional projected power spectrum
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    #create the noise
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    #calculate its amplitude
    amplitude = np.zeros((size, size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)


#Plot the GRF map for a given size for different power spectra
for alpha in [-4.0, -3.0, -2.0]:
    out = gaussian_random_field(Pk = lambda k: k**alpha, size=256)
    plt.figure()
    plt.imshow(out.real, interpolation='none')
    plt.show()
    plt.savefig('test_GRF_with'+str(np.abs(alpha))+'.png', dpi=400)

