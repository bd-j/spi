import os, glob, sys
import numpy as np
import astropy.io.fits as pyfits
import h5py
import json

files = glob.glob('fits_spectra/*fits')
names = [os.path.basename(f).replace('.fits', '') for f in files]
nirtf = len(files)

miles = h5py.File('../miles/miles_prugniel.h5', "r")
dat, hdr = pyfits.getdata(files[0], header=True)
nwave = len(dat)
spectra = np.zeros([nirtf, nwave])
uncertainty = np.zeros([nirtf, nwave])
for i, n in enumerate(names):
    dat = pyfits.getdata(files[i])
    hdr = pyfits.getheader(files[
    miles_id = hdr['miles_id']
    spectra[i, :] = dat['flux']
    uncertainty[i, :] = dat['uncertainty']
    parameters[i] = miles['parameters'][miles_id - 1]
    ancillary[i] = miles['ancillary'][miles_id - 1]

with h5py.File('irtf_prugniel.h5', "w") as h5:
    h5.create_dataset('wavelengths', data=dat['wavelength'])
    h5.create_dataset('spectra', data=spectra)
    h5.create_dataset('uncertainty', data=uncertainty)
    h5.create_dataset('parameters', data=parameters)
    h5.create_dataset('ancillary', data=ancillary)
    h5.attrs['units'] = miles.attrs['units']
