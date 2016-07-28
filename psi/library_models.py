import numpy as np
from numpy.lib import recfunctions as rfn
import h5py
from .model import SimplePSIModel


__all__ = ["MILESInterpolator", "CKCInterpolator"]


class MILESInterpolator(SimplePSIModel):

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).
        """
        self.has_errors = False
        with h5py.File(training_data, "r") as f:
            self.training_spectra = f['spectra'][:]
            self.training_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]
            try:
                self.training_weights = 1/(f['uncertainty'][:]**2)
                self.has_errors = True
            except:
                pass
            ancillary = f['ancillary'][:]
        # add and rename labels here.  Note that not all labels need to be or
        # will be used in the feature generation
        newfield = ['logt', 'miles_id']
        newdata = [np.log10(self.training_labels['teff']), ancillary['miles_id']]
        self.training_labels = rfn.append_fields(self.training_labels,
                                                 newfield, newdata, usemask=False)
        try:
            # assuming f_nu
            fbol = np.trapz(self.training_spectra/self.wavelengths**2, self.wavelengths)
            newfield = ['logl', 'luminosity', 'fbol']
            newdata = [ancillary['logl'], 10**ancillary['logl'], fbol]
            self.training_labels = rfn.append_fields(self.training_labels,
                                                     newfield, newdata, usemask=False)            
        except:
            pass


class CKCInterpolator(SimplePSIModel):

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).
        """
        self.has_errors = False
        with h5py.File(training_data, "r") as f:
            self.training_spectra = f['spectra'][:]
            self.training_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]
            try:
                self.training_weights = 1/(f['uncertainty'][:]**2)
                self.has_errors = True
            except:
                pass
        try:
            # assuming f_nu
            fbol = np.trapz(self.training_spectra/self.wavelengths**2, self.wavelengths)
            newfield = ['fbol']
            newdata = [fbol]
            self.training_labels = rfn.append_fields(self.training_labels,
                                                     newfield, newdata, usemask=False)            
        except:
            pass
