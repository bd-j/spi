import numpy as np
import h5py
import json
import astropy.io.fits as pyfits

prugniel_file = 'prugniel11_params.fits'
prugniel_ebv = 'prugniel11_ebv.fits'

p11_labels = {'Teff': 'teff', '__Fe_H_': 'feh', 'logg': 'logg'}
p11_ancillary = {'Name': 'name', 'Miles': 'miles_id', 'SimbadName': 'simbad',
                 '_RA': 'ra', '_DE': 'dec',
                 'cz': 'cz', 'sig': 'sigma',
                 'e_Teff': 'teff_unc', 'e__Fe_H_': 'feh_unc', 'e_logg':'logg_unc'
                 }
                      
def with_miles_params(miles_file):
    pass

def with_prugniel_params(prugniel_file=prugniel_file, outfile='miles_prugniel.h5'):
    """Read Prugniel parameters and ancillary data into structured arrays with
    sensible column names, while stripping all the pyfits cruft.

    Then read the spectra and put everything in an HDF5 file.

    Much procedural!
    """
    data = pyfits.getdata(prugniel_file)
    nstar = len(data)
    pdt = np.dtype([(n, np.float) for n in p11_labels.values()])
    parameters = np.zeros(nstar, dtype=pdt)
    adt = np.dtype([(b, data[a].dtype) for a, b in list(p11_ancillary.items())])
    ancillary = np.zeros(nstar, adt)
    for a, b in list(p11_labels.items()):
        parameters[b] = data[a]
    for a, b in list(p11_ancillary.items()):
        ancillary[b]= data[a]

    allfield = p11_labels.copy()
    allfield.update(p11_ancillary)
    units = {}
    for a, b in list(allfield.items()):
        units[b] = data.columns[a].unit
    units['wavelengths'] = 'angstroms'
    
    spectra = np.zeros([nstar, 4300])
    for i in range(1, nstar+1):
        n = 'ascii_spectra/m{:04.0f}V'.format(i)
        w, s = np.genfromtxt(n, unpack=True)
        spectra[i-1, :] = s

    with h5py.File(outfile, "x") as f:
        f.create_dataset('spectra', data=spectra)
        f.create_dataset('wavelengths', data=w)
        f.create_dataset('parameters', data=parameters)
        f.create_dataset('ancillary', data=ancillary)
        f.attrs['units'] = json.dumps(units)


if __name__ == "__main__":
    with_prugniel_params()
