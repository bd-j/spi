import os, glob, sys
import numpy as np
import astropy.io.fits as pyfits
import h5py

files = glob.glob('fits_spectra/*fits')
names = [os.path.basename(f).replace('.fits', '')]

miles = h5py.File('../miles/miles_prugniel.h5', "r")

nirtf = len(files)
