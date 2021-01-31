import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy import ndimage, misc
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
z = 30
####################
foreground_type = 1
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
rot_angle_uv =np.pi/4# math.pi/4 #rotation angle in the uv plane
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
                patch_rotated[int(np.floor(math.cos(rot_angle_uv) * (k - size/2) - math.sin(rot_angle_uv) * (l - size/2)) + size/2)][
                    int(np.floor(math.sin(rot_angle_uv) * (k - size/2) + math.cos(rot_angle_uv) * (l - size/2)) + size/2)] = patch[k][l]
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

###################


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


######################
epsilon_fgr = 0.3e-2#with kernel #0.15*1e-2 without kernel
kx, ky = np.meshgrid(2 * math.pi * np.fft.fftfreq(N, c),
                             2 * math.pi * np.fft.fftfreq(N, c))
mag_k = np.sqrt(kx ** 2 + ky ** 2)
grf = np.random.normal(0., 1., size = (patch_size, patch_size))
grf_II = np.random.normal(0., 1., size=(patch_size, patch_size))
grf_III = np.random.normal(0., 1., size=(patch_size, patch_size))
grf_IV = np.random.normal(0., 1., size=(patch_size, patch_size))

fg = foreground(180*mag_k/np.pi, 1)
fg_II = foreground(180*mag_k/np.pi, 2)
fg_III = foreground(180*mag_k/np.pi, 3)
fg_IV = foreground(180*mag_k/np.pi, 4)

grf_fg = np.fft.fft2(grf) * fg ** 0.5 * 1e-3
grf_fg_II = np.fft.fft2(grf_II) * fg_II ** 0.5 * 1e-3
grf_fg_III = np.fft.fft2(grf_III) * fg_III ** 0.5 * 1e-3
grf_fg_IV = np.fft.fft2(grf_IV) * fg_IV ** 0.5 * 1e-3

grf_norm_fg, std_fg, norm = fg_normalize(grf_fg, foreground_type)
grf_norm_fg_IV, std_fg_IV, norm_IV = fg_normalize(grf_fg_IV, foreground_type)
grf_norm_fg_II, std_fg_II, norm_II = fg_normalize(grf_fg_II, foreground_type)
grf_norm_fg_III, std_fg_III, norm_III = fg_normalize(grf_fg_III, foreground_type)

data_ft = (grf_norm_fg + grf_norm_fg_II + grf_norm_fg_III + grf_norm_fg_IV)*1e3*-delta_z*epsilon_fgr
data_real = np.fft.ifft2(data_ft).real
############
data_real_norm = (data_real-data_real.min())*255/(data_real.max()-data_real.min())
############
for i in range(0, len(data_real_norm)):
    for j in range(0, len(data_real_norm)):
        data_real_norm[i,j] = int(np.round(data_real_norm[i,j]))
sig = np.fft.ifft2(signal_ft(patch_size, wake_size_angle,  angle_per_pixel, shift_wake_angle, False)).real
############
sig_norm = ((sig-sig.min())*255/(sig.max()-sig.min()))
############
both = data_real + sig
############
both_norm = ((both-both.min())*255/(both.max()-both.min()))
############

'''im = Image.fromarray(both_norm)
im = im.convert('L')
im = im.filter(ImageFilter.FIND_EDGES)
im.show()'''


both_GF = ndimage.gaussian_filter(both_norm, sigma=2.)

fig = plt.figure()
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ax1.imshow(both)
ax2.imshow(both_GF)
plt.show()


G, theta = sobel_filters(both_GF)
plt.imshow(G)
plt.show()