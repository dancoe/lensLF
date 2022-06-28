import numpy as np

import astropy
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Distance
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Load magnification map
z = 9  # source redshift defined for map
filename = 'macs0417-11_lenstool_v2b_z9magnif.fits.gz'
hdu = fits.open(filename)[0]
magnif = hdu.data
magnifs = magnif.ravel()
magnifs = np.compress(magnifs, magnifs)
dmags = 2.5 * np.log10(magnifs)

# magnification grid
maxmagnif = 10  # 10 mags = 1e4
dmagnif = 0.01
dmag_bins = np.arange(0, maxmagnif+dmagnif/10., dmagnif)
magnif_bins = 10 ** (0.4 * dmag_bins)

# Luminosity Function magnitude grid
maglo, maghi = 24, 32 + maxmagnif
mags = np.arange(maglo, maghi, dmagnif)

# Luminosity Function parameters
phistar = 4.27e-4
alpha = -2
Mstar = -20.26

# Distance Modulus
# M - m = 2.5 log10(1+z) - distance_modulus
M_m = 2.5 * np.log10(1+z) - Distance(z=z, cosmology=cosmo).distmod.value
MUV = mags + M_m
m = MUV - Mstar

# Luminosity Function calculation
l = 10 ** (-0.4 * m)  # L / L*
phi = 0.4 * np.log(10) * phistar * (l ** (alpha + 1)) * np.exp(-l) # / Mpc^3

# / Mpc^3 -> arcmin^2
dVC = cosmo.differential_comoving_volume(z).to(u.Mpc**3 / u.arcmin**2)
phi *= dVC.value  # / arcmin^2

# Lens the luminosity function using convolution (where the magic happens!)
mag_hist, bins2 = np.histogram(dmags, np.array(dmag_bins))
phi_lensed = np.convolve(phi, mag_hist[::-1] / magnif_bins[:-1][::-1])
phi_lensed = phi_lensed[-len(phi):-len(mag_hist)]
phi_lensed /= np.sum(mag_hist)
mags_lensed = mags[:-len(mag_hist)]


import matplotlib.pyplot as plt

f,ax = plt.subplots(1,1,figsize=(6,4))
#plt.plot(mags, phi)
plt.plot(mags_lensed, phi[:-len(mag_hist)])
plt.plot(mags_lensed, phi_lensed, '--')

plt.semilogy()
plt.ylabel('# / mag / $\Delta$z / arcmin$^2$')
plt.xlabel('magnitude')