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
all_ids, all_lum = [], []
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
    else:
        print(files[i], mid)
        

spectra = np.array(spectra)
uncertainty = np.array(uncertainty)
parameters = miles['parameters'][:][np.array(all_ids) - 1]
ancillary = miles['ancillary'][:][np.array(all_ids) - 1]
ancillary = rfn.append_fields(ancillary, 'logl', np.array(all_lum), usemask=False)

assert len(parameters) == len(all_lum)
assert len(parameters) == len(spectra)

with h5py.File('irtf_prugniel_extended.h5', "w") as h5:
    h5.create_dataset('wavelengths', data=dat['wavelength'])
    h5.create_dataset('spectra', data=spectra)
    h5.create_dataset('uncertainty', data=uncertainty)
    h5.create_dataset('parameters', data=parameters)
    h5.create_dataset('ancillary', data=ancillary)
    h5.attrs['units'] = miles.attrs['units']
