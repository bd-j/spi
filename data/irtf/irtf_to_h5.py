import os, glob, sys
import numpy as np
import astropy.io.fits as pyfits
import h5py
import json
import numpy.lib.recfunctions as rfn

files = glob.glob('fits_spectra/*fits')
names = [os.path.basename(f).replace('.fits', '') for f in files]
nirtf = len(files)

miles = h5py.File('../miles/miles_prugniel.h5', "r")
dat, hdr = pyfits.getdata(files[0], header=True)
nwave = len(dat)
spectra = []#np.zeros([nirtf, nwave])
uncertainty = []#np.zeros([nirtf, nwave])
all_ids, all_lum, all_dist = [], [], []
for i, n in enumerate(names):
    dat = pyfits.getdata(files[i])
    hdr = pyfits.getheader(files[i])
    mid = hdr['miles_id']
    miles_id = int(mid.strip()[:-1])
    #print(mid, miles_id)
    if len(dat['flux']) == nwave:
        spectra.append(dat['flux'])
        uncertainty.append(dat['uncertainty'])
        all_ids.append(miles_id)
        all_lum.append(float(hdr['logl']))
        try:
            all_dist.append(1 / (float(hdr['parallax'] * 1e-3)))
        except:
            all_dist.append(hdr['distance'])
    else:
        print(files[i], mid)
        

spectra = np.array(spectra)
uncertainty = np.array(uncertainty)
parameters = miles['parameters'][:][np.array(all_ids) - 1]
ancillary = miles['ancillary'][:][np.array(all_ids) - 1]
ancillary = rfn.append_fields(ancillary, ['logl', 'distance'],
                              [np.array(all_lum), np.array(all_dist)],
                               usemask=False)

assert len(parameters) == len(all_lum)
assert len(parameters) == len(spectra)


# normalize to be in L_sun/Hz
from astropy import constants as const
lsun = const.L_sun.cgs.value
lightspeed = 2.998e14 # micron/s
d = np.array(all_dist) * const.pc.cgs.value
# not sure why 1e6 necessary here....
l_nu = spectra * (np.pi * 4 * d[:, None]**2) / lsun
wave = dat['wavelength']
lum = np.trapz(l_nu * lightspeed / wave**2, wave)
#plot(lum / 10**np.array(all_lum), 'o')


with h5py.File('irtf_prugniel_extended.h5', "w") as h5:
    h5.create_dataset('wavelengths', data=dat['wavelength'])
    h5.create_dataset('spectra', data=l_nu)
    h5.create_dataset('uncertainty', data=uncertainty)
    h5.create_dataset('parameters', data=parameters)
    h5.create_dataset('ancillary', data=ancillary)
    units = json.loads(miles.attrs['units'])
    units['wavelengths'] = 'micron'
    units['distance'] = 'parsec'
    units['flux'] = 'L_sun/Hz'
    h5.attrs['units'] = json.dumps(units)

