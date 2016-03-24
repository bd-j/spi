import numpy as np
import astropy.io.fits as pyfits
import h5py
from bsfh.source_basis import StarBasis
from model import within

__all__ = ["PiecewiseMILES", "TGM"]

class PiecewiseMILES(StarBasis):


    def load_lib(self, libname='', **extras):
        """Read a CKC library which has been pre-convolved to be close to your
        resolution.  This library should be stored as an HDF5 file, with the
        datasets ``wavelengths``, ``parameters`` and ``spectra``.  These are
        ndarrays of shape (nwave,), (nmodels,), and (nmodels, nwave)
        respecitvely.  The ``parameters`` array is a structured array.  Spectra
        with no fluxes > 1e-32 are removed from the library
        """
        import h5py
        with h5py.File(libname, "r") as f:
            self._wave = np.array(f['wavelengths'])
            self._libparams = np.array(f['parameters'])
            self._spectra = np.array(f['spectra'])
            ancillary = f['ancillary'][:]
        # add and rename labels here.  Note that not all labels need or will be
        # used in the feature generation
        newfield = ['logt', 'miles_id']
        newdata = [np.log10(self._libparams['teff']), ancillary['miles_id']]
        self._libparams = rfn.append_fields(self._libparams, newfield, newdata,
                                            usemask=False)
        
    def select(self, bounds=None, badvalues=None):

        if bounds is not None:
            good = np.ones(len(self._libparams), dtype=bool)
            for name, bound in bounds.items():
                good = good & within(bound, self._libparams[name])
            self._spectra = self._spectra[good, :]
            self._libparams = self._libparams[good, ...]

        if badvalues is not None:
            inds = []
            for name, bad in badvalues.items():
                inds += [self._libparams[name].tolist().index(b) for b in bad
                        if b in self._libparams[name]]
            self.leave_out(np.array(inds).flat)

        self.triangulate()

    def leave_out(self, inds):
        self._spectra = np.delete(self._spectra, inds, axis=0)
        self._libparams = np.delete(self._libparams, inds)
        self.reset()

    def reset(self):
        self.triangulate()
        try:
            self.build_kdtree()
        except NameError:
            pass


class TGM(object):
    """Just try the coefficients from Prugniel
    """
    def __init__(self, interpolator='miles_tgm.fits', trange='warm'):
        extension = {'warm': 0, 'hot':1, 'cold': 2}
        self.trange = trange
        coeffs, hdr = pyfits.getdata(interpolator, ext=extension[trange],
                                     header=True)
        self.coeffs = coeffs.T
        self.version = int(hdr['intrp_v'])

        w = (np.arange(coeffs.shape[1]) - (hdr['CRPIX1']-1)) * hdr['CDELT1'] + hdr['CRVAL1']
        self.wavelengths = w
        self.n_wave = len(self.wavelengths)
        
    def labels_to_features(self, logt=3.7617, logg=4.44, feh=0, **extras):
        logt_n = logt - 3.7617
        grav = logg - 4.44
        tt = logt_n / 0.2  # why?
        tt2 = tt**2 - 1.  # why? Chebyshev.
        # first 20 terms
        features = [1., tt, feh, grav, tt**2, tt*tt2, tt2*tt2, tt*feh, tt*grav,
                    tt2*grav, tt2*feh, grav**2, feh**2, tt*tt2**2, tt*grav**2, grav**3, feh**3,
                    tt*feh**2, grav*feh, grav**2*feh, grav*feh**2]
        if self.version == 2:
            features += [np.exp(tt) - 1. - tt*(1. + tt/2. + tt**2/6. + tt**3/24. + tt**4/120.),
                         np.exp(tt*2) - 1. - 2.*tt*(1. + tt + 2./3.*tt**2 + tt**3/3. + tt**4*2./15.)
                         ]
        elif self.version == 3:
            features += [tt*tt2*grav, tt2*tt2*grav, tt2*tt*feh, tt2*grav**2, tt2*grav**3]

        X = np.array(features)
        return X.T

    def get_star_spectrum(self, **kwargs):
        """Get an interpolated spectrum at the parameter values (labels)
        specified as keywords.  These *must* include all elements of
        ``label_names``.
        """
        features = self.labels_to_features(**kwargs)
        spectrum = np.dot(self.coeffs, features.T)
        return np.squeeze(spectrum.T)
