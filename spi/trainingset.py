import numpy as np
import h5py
from .utils import *

__all__ = ["TrainingSet"]

class TrainingSet(object):

    """This object loads and stores the full spectral library from which the
    training data are drawn, and handles masking and unmasking specific spectra
    based on various criteria.  It also has a method for renormalizing the
    spectra.  It is meant to be inherited by the SPIModel classes, which will
    operate on the selected training data.

    The unmasked spectra and labels are accessed through the `training_spectra`
    and `training_labels` attributes.
    """

    def __init__(self):
        """This object is only meant to be inherited.
        """
        pass

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Store these in the `library_labels` and
        `library_spectra` attributes.  We will use a lightweight adjustable
        mask to then get the `training_labels` and `training_spectra`
        attributes.
        """
        with h5py.File(training_data, "r") as f:
            self.library_spectra = f['spectra'][:]
            self.library_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]
        # add and rename labels here
        self.has_errors = False
        self.reset_mask()

    def reset_mask(self):
        self.library_mask = np.ones(self.n_library, dtype=bool)

    def select(self, training_data=None, bounds=None, badvalues=None,
               delete=False, **extras):
        """Load and select a subsample of a particular library of training data.
        """
        if training_data is not None:
            self.load_training_data(training_data=training_data, **extras)

        if bounds is not None:
            self.restrict_sample(bounds=bounds, **extras)

        if badvalues is not None:
            inds = []
            for name, bad in badvalues.items():
                inds += [self.library_labels[name].tolist().index(b) for b in bad
                        if b in self.library_labels[name]]
            self.leave_out(np.array(inds).flat)

        if delete:
            self.delete_masked()

        self.build_training_info()
        self.reset()

    def reset(self):
        """Implemented by subclasses"""
        pass

    def build_training_info(self):
        """Implemented by subclasses"""
        pass
        
    def restrict_sample(self, bounds=None, **extras):
        """Remove training objects that are not within some sample.
        """
        if bounds is None:
            return
        for name, bound in bounds.items():
            self.library_mask = self.library_mask & within(bound, self.library_labels[name])

    def leave_out(self, inds):
        """Remove training objects specified by `inds`.  Useful for
        leave-one-out validation.

        :param inds:
            Int, array of ints, or slice specifying which training objects to
            remove.  Passed to numpy.delete
        """
        self.library_mask[inds] = False
        # self.training_spectra = np.delete(self.training_spectra, inds, axis=0)
        # self.training_labels = np.delete(self.training_labels, inds)

    def delete_masked(self):
        """Remove masked spectra from the training set entirely.  This can
        speed up later operations, but is irreversible, and will change indices
        of individual spectra that remain.
        """
        self.library_spectra = self.library_spectra[self.library_mask, :]
        self.library_labels = self.library_labels[self.library_mask]
        if self.has_errors:
            self.library_snr = self.library_snr[self.library_mask, :]
        self.reset_mask()

    def renormalize_library_spectra(self, normwave=None, bylabel=None):
        """Renormalize the spectra by some quantity.  This can either be the
        flux at a specfic wavelength, or one of the library_labels
        """
        if normwave is not None:
            ind_wave = np.argmin(np.abs(self.wavelengths - normwave))
            self.library_spectra /= self.library_spectra[:, ind_wave][:, None]
        elif bylabel is not None:
            self.library_spectra /= self.library_labels[bylabel][:, None]

    @property
    def n_library(self):
        return len(self.library_labels)

    @property
    def training_indices(self):
        return np.where(self.library_mask)[0]

    @property
    def training_spectra(self):
        return self.library_spectra[self.library_mask, :]

    @property
    def training_labels(self):
        return self.library_labels[self.library_mask]

    @property
    def training_snr(self):
        return self.library_snr[self.library_mask, :]
