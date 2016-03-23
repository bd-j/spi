import numpy as np
import matplotlib.pyplot as pl
from numpy.lib import recfunctions as rfn

from bsfh.source_basis import StarBasis

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
        newdata = [np.log10(self.training_labels['teff']), ancillary['miles_id']]
        self._libparams = rfn.append_fields(self._libparams, newfield, newdata,
                                            usemask=False)
        
    def select(self, bounds=None, badvalues=None):

        if bounds is not None:
            good = np.ones(self.n_train, dtype=bool)
            for name, bound in bounds.items():
                good = good & within(bound, self._libparams[name])
            self._spectra = self._spectra[good, :]
            self._libparams = self._libparams[good, ...]

        if badvalues is not None:
            inds = []
            for name, bad in badvalues.items():
                inds += [self._libparams[name].tolist().index(bad) for b in badvalues
                        if b in self._libparams[name]]
            self.leave_out(inds)

        self.triangulate()

    def leave_out(self, inds):
        self._spectra = np.delete(self._spectra, inds, axis=0)
        self._libparams = np.delete(self.libparams, inds)
        self.reset()

    def reset(self):
        self.triangulate()
        try:
            self.build_kdtree()
        except NameError:
            pass

            
pint = PeicewiseMILES(libname=mlib, use_params=['logt', 'logg', 'feh'])
