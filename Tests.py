
import numpy as np
import math
import matplotlib.pyplot as plt


#rotation

'''theta = math.pi/4
field = np.zeros((500,500))
for i in range(50,150):
    for j in range(50,150):
        field[i][j] = -(j-50)/(100.)
field2 = np.zeros((500,500))
for k in range(50,150):
    for l in range(50,150):
        field2[int(np.floor(math.cos(theta)*(k-100)-math.sin(theta)*(l-100)))+100][int(np.floor(math.sin(theta)*(k-100)+math.cos(theta)*(l-100)))+100]=field[k][l]
for p in range(1,199):
    for q in range(1,199):
        if np.abs(field2[p][q-1]+field2[p-1][q]+field2[p+1][q]+field2[p][q+1])>2*max(np.abs([field2[p][q-1],field2[p-1][q],field2[p+1][q],field2[p][q+1]])):
            a=np.array([field2[p][q - 1], field2[p - 1][q], field2[p + 1][q], field2[p][q + 1]])
            field2[p][q]=np.mean(a[np.nonzero(a)])
field3= np.zeros((20,20))
for a in range(0,20):
    for b in range(0,20):
        field3[a][b]=field2[a+120][b+120]
field3[20-1][20-1]=1
plt.imshow(field2)
plt.show()'''







#Interferometer PS

'''z = 30 #redshift
T_sky = 60 * 1e3 * (1420/((1.+z) * 300))**-2.5 #in mK
T_inst = 1
T_sys = T_sky + T_inst #temperature
N_d = 256 #numper of dishes for MWA II: https://core.ac.uk/download/pdf/195695824.pdf
N_p = 0 #number of pointings: N_p Omega_p = 4 Pi f_sky
A_e = 1 #effective total dish area
D_min =1 #smallest baseling in m
D_max = 5300 #longest baseline in m for MWA II: "The Phase II Murchison Widefield Array: Design overview"
c = 3*1e8
t_tot = 1 #total integration time
d_nu =1 #bandwidth in that channel

def n_u(k): #trivial approximation for unknown baseline distributions
    lamb = 2 * math.pi/k
    lamb[k < 1e-2] = 2 * math.pi/1e-2
    return N_d * (N_d-1)/(2 * math.pi * ((D_max/lamb)**2 - (D_min/lamb)**2))


#test intererometer noise power spectrum
def pspec_inter(k):
    return (1420 * 1e6/(1. + z) * T_sys**2 * N_p)/(n_u(k) * t_tot * d_nu * A_e**2)


#define rest
patch_size = 512
patch_angle = 5. #in degree
bins = 300
angle_per_pixel = patch_angle/patch_size
c = 2 * math.pi * angle_per_pixel
N = 512
kx, ky = np.meshgrid(np.fft.fftfreq(N, c), np.fft.fftfreq(N, c))
mag_k = np.sqrt(kx**2 + ky**2)
k_bins = np.linspace(0.1, 0.95*mag_k.max(), bins)
k_bin_cents = k_bins[:-1] + (k_bins[1:] - k_bins[:-1])/2'''





#Integration methods

def deexitation_crosssection(t_k):
    if 1<t_k and t_k<=2:
        return 1.38e-13+(t_k-1)*(1.43-1.38) * 1e-13
    if 2 < t_k and t_k <= 4:
        return 1.43e-13 + (t_k-2)/2*(2.71-1.43) * 1e-13
    if 4 < t_k and t_k <=6:
        return 2.71e-13 + (t_k-4)/2*(6.6-2.71) * 1e-13
    if 6 < t_k and t_k <= 8:
        return 6.60e-13 + (t_k-6)/2*(1.47e-12-6.6e-13)
    if 8 < t_k and t_k <= 10:
        return 1.47e-12+ (t_k-8)/2*(2.88-1.47 )*1e-12
    if 10 < t_k and t_k <= 15:
        return 2.88e-12+ (t_k-10)/5*(9.10-2.88)*1e-12
    if 15 < t_k and t_k <= 20:
        return 9.1e-12+ (t_k-15)/5*(1.78e-11 -9.10*1e-12)
    if 20 < t_k and t_k <= 25:
        return 1.78e-11+ (t_k-20)/5*(2.73-1.78)*1e-11
    if 25 < t_k and t_k <= 30:
        return 2.73e-11+ (t_k-25)/5*(3.67-2.73)*1e-11
    if 30 < t_k and t_k <= 40:
        return 3.67e-11+ (t_k-30)/10*(5.38-3.67)*1e-11
    if 40 < t_k and t_k <= 50:
        return 5.38e-11+ (t_k-40)/10*(6.86-5.38)*1e-11
    if 50 < t_k and t_k <= 60:
        return 6.86e-11+ (t_k-50)/10*(8.14-6.86)*1e-11
    if 60 < t_k and t_k <= 70:
        return 8.14e-11+ (t_k-60)/10*(9.25-8.14)*1e-11
    if 70 < t_k and t_k <= 80:
        return 9.25e-11+ (t_k-70)/10*(1.02e-10-9.25*1e-11)
    if 80 < t_k and t_k <= 90:
        return 1.02e-10+ (t_k-80)/10*(1.11-1.02)*1e-10
    if 90 < t_k and t_k <=100:
        return 1.11e-10+ (t_k-90)/10*(1.19-1.11)*1e-10
    if 100 < t_k and t_k <= 200:
        return 1.19e-10+ (t_k-100)/100*(1.75-1.19)*1e-10
    if 200 < t_k and t_k <= 300:
        return 1.75e-10+ (t_k-200)/100*(2.09-1.75)*1e-10
    else:
        print('T_K is out of scope for the deexcitation fraction')
        return 0
n = 30
T_b_plot = np.zeros(n-5)
T_b_plot_2 = np.zeros(n-5)
T_b_plot_3 = np.zeros(n-5)
for i in range(5, n):
    #redshift interval probing #TODO: average all redshift dependent quantities over the redshift bin
    z = i
    #redshift string formation
    z_i = 1000
    #thickness redshift bin
    theta =math.pi/4
    delta_f = 0.04
    delta_z = -delta_f/(1420)*(z+1)
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
    xcc = 4*1.9e-7*deexitation_crosssection(T_K)* 0.068/(2.85e-15 *2.725)
    #wake brightness temperature [K]
    dz_wake = 24 * math.pi / 15 * gmu_6 * 1e-6 * vsgammas_square ** 0.5 * (z_i + 1) ** 0.5 * (z_wake + 1.) ** 0.5
    df_wake = 24 * math.pi / 15 * gmu_6 * 1e-6 * vsgammas_square ** 0.5 * (z_i + 1) ** 0.5 * (
            z_wake + 1.) ** -0.5 * 1420 / np.cos(theta)  # MHz. THe 2sin^2 theta cancels when multiplied with T_b
    T_b = 1e3*0.07 *(2*np.sin(theta)**2)**-1* xcc*(1+z_wake)**2/(xcc*(1+z_wake)**2+1.)*2./3*((1 + z_wake + dz_wake/2. )**1.5-(1 + z_wake - dz_wake/2. )**1.5) - 1e3*(2*np.sin(theta)**2)**-1*0.07 * xc/(xc+1.)*2.725/(20 * gmu_6**2 * vsgammas_square * (z_i+1.)) * 2/7. * ((1 + z_wake + dz_wake/2. )**3.5-(1 + z_wake - dz_wake/2. )**3.5) #in mK
    #print(T_bb)
    #T_b = 1e3*(2*np.sin(theta)**2)**-1* df_wake/delta_f * 0.07 * xc/(xc+1.)*(1-2.725*(1+z_wake)/(20 * gmu_6**2 * vsgammas_square * (z_i+1.)/(z_wake+1)))*(1.+z_wake)**0.5
    print(T_b)
    #T_b_2 = np.sqrt(1100 * 1e-6 * (1100 / 0.100) ** 3.3 * ((130. ** 2 / 1420 ** 2) *(1+z_wake)**2)**2.8)
    T_b_2 =  1e-3* (1./(3.8)*((1.+z)**(2.99+1.) - (1.+ z+delta_z)**(2.99+1.)))
    #fraction of baryonc mass comprised of HI. Given that we consider redshifts of the dark ages, we can assume that all the
    #hydrogen of the universe is neutral and we assume the mass fraction of baryoni is:
    xHI = 0.75
    #background temperature [K] (assume Omega_b, h, Omega_Lambda, Omega_m as in arXiv: 1405.1452[they use planck collaboration 2013b best fit])
    T_back = (0.19055e-3) * (0.049*0.67*(1.+i)**2 * xHI)/np.sqrt(0.267*(1.+i)**3 + 0.684)
    T_b_plot[i-5] = np.abs(T_b)
    T_b_plot_2[i-5] = T_b_2
    #T_b_plot_3[i-5] = T_b_3

#print(T_b_plot[13-1])
#print(T_b_plot[30-1])
#print(T_b_plot_2[13-1])
#print(T_b_plot_2[30-1])

markers_on =[8]
#plt.plot(np.linspace(5, n, n-5), np.abs(T_b_plot),"-|", markevery= markers_on)
plt.plot(np.linspace(5, n, n-5), T_b_plot,"-|", markevery= markers_on)
plt.plot(np.linspace(5, n, n-5), np.abs(T_b_plot_2),"-|", markevery= markers_on)
plt.show()
a = T_b_plot/T_b_plot_2
dum=np.zeros(2)
for i in range(5, len(a)):
    #if a[i]> dum[0]:
    dum[0]=a[i]
    dum[1]=i
    print(str(dum[1]+5)+'  '+str(dum[0]))
plt.plot(np.linspace(5, n, n-5), T_b_plot/T_b_plot_2 )
plt.xlabel('z')
plt.ylabel('mK')
plt.show()