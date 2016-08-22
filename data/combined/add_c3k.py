import sys
import numpy as np
import matplotlib.pyplot as pl
import h5py

from numpy.lib import recfunctions as rfn
from scipy.spatial import Delaunay, ConvexHull

from psi.library_models import SimplePSIModel, MILESInterpolator, CKCInterpolator
from psi.utils import dict_struct, flatten_struct

class CombinedInterpolator(SimplePSIModel):

    def load_training_data(mlib='', clib='', cweight=1e-1):
        # do MILES
        self.has_errors = True
        with h5py.File(mlib, "r") as f:
            self.wavelengths = f['wavelengths'][:]
            spectra = f['spectra'][:]
            labels = f['parameters'][:]
            weights = 1.0 / (f['uncertainty'][:]**2)

        # add C3K
        with h5py.File(clib, "r") as f:
            assert np.allclose(self.wavelengths, f['wavelengths'][:])
            self.spectra = np.vstack([spectra, f['spectra']])
            norm = np.median(weights) * cweight
            self.weights = np.vstack([weights,
                                      norm/(f['uncertainties'][:]**2)])
            # build a useful structured array and fill it
            newlabels = np.zeros(len(f['parameters']),
                                 dtype=slabels.dtype)
            for l in labels.dtype.names:
                if l in f['parameters'].dtype.names:
                    newlabels[l] = f['parameters'][l]

            self.library_labels = np.hstack([labels,
                                             newlabels])


def outside_hull(primary, secondary, use_labels=['logt', 'logg', 'feh']):
    L = flatten_struct(primary, use_labels=use_labels)
    l = flatten_struct(secondary, use_labels=use_labels)
    hull = Delaunay(L.T)
    return hull.find_simplex(l.T) >= 0


def logify_teff(params):
    params = rfn.append_fields(params, ['logt'], [np.log10(params['teff'])],
                                usemask=False)
    return params

def write_h5(outname, wave, spec, unc, label):
    with h5py.File(outname, "w") as f:
        w = f.create_dataset('wavelengths', data=wave)
        s = f.create_dataset('spectra', data=spec)
        l = f.create_dataset('parameters', data=label)
        u = f.create_dataset('uncertainty', data=unc)

def broaden(wave, spec, outwave=None):
    pass


def rectify_c3k(c3k, selection, broaden_params):
    with h5py.File(c3k, "r") as f:
        spectra = f['spectra'][selection]
        labels = f['parameters'][selection]
        wave = f['wavelengths'][:]

    # Renormalize
    logl, log4pi = 0.0, np.log10(4 * np.pi)
    twologR = (logl+log_lsun_cgs) - 4 * labels['logt'] - log_SB_cgs - log4pi
    spectra *= 10**(twologR[:, None] + 2 * log4pi)
    # Add new label info
    newcols = ['miles_id', 'logl', 'luminosity']
    newdata = [np.array(nobj * ['c3k']), np.zeros(nobj), np.ones(nobj)]
    labels = rfn.append_fields(labels, newfield, newdata, usemask=False)
    # Broaden
    w, spec = broaden(wave, spectra, **broaden_params)

    write_h5(outname, w, spec, unc, labels)


def rectify_miles(mlib, outname):
    with h5py.File(mlib, "r") as f:
        spectra = f['spectra'][:]
        labels = f['parameters'][:]
        wave = f['wavelengths'][:]
        ancillary = f['ancillary'][:]
        try:
            unc = f['uncertainty'][:]
        except:
            unc = np.ones_like(spectra)

    newfield = ['logt', 'miles_id', 'logl', 'luminosity']
    newdata = [np.log10(labels['teff']), ancillary['miles_id'],
               ancillary['logl'], 10**ancillary['logl']]
    labels = rfn.append_fields(labels, newfield, newdata, usemask=False)

    #return wavelengths, spectra, unc, labels
    write_h5(outname, wave, spectra, unc, labels)


def combine_libraries(mlib='', clib='', cweight=1e-1):
    # MILES
        
    # add C3K
    with h5py.File(clib, "r") as f:
        assert np.allclose(wavelengths, f['wavelengths'][:])
        spectra = np.vstack([spectra, f['spectra']])
        # build a useful structured array and fill it
        newlabels = np.zeros(len(f['parameters']),
                             dtype=labels.dtype)
        for l in labels.dtype.names:
            if l in f['parameters'].dtype.names:
                newlabels[l] = f['parameters'][l]
        newlabels['miles_id'] = 'C3K'
        labels = np.hstack([labels, newlabels])


if __name__ == "__main__":
        
    clibname = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/ckc_R10k.h5'
    mlibname = '/Users/bjohnson/Projects/psi/data/combined/with_mdwarfs_culled_lib_snr_cut.h5'
    clib = h5py.File(clibname, 'r')
    mlib = h5py.File(mlibname, 'r')
    
    # Find C3K objects outside the MILES convex hull
    mparams = mlib['parameters'][:]
    inside = outside_hull(logify_teff(mparams), clib['parameters'][:])
    bad = np.max(clib['spectra'], axis=-1) <= 0

    good = (~inside) & (~bad)
    #clib.close()
    #mlib.close()
    # Normalize C3K by Lbol

    # Broaden C3K to MILES+IRTF resolution, and put on same wavelength scale
