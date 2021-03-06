# Subclasses of the generic SimpleSPIModel used for specific spectral data.
# This mostly involves over-riding the load_training_data method of the
# TrainingSet object inherited by SimpleSPIModel.

import numpy as np
from numpy.lib import recfunctions as rfn
import h5py
from .models import SimpleSPIModel


__all__ = ["MILESInterpolator", "CKCInterpolator"]

lightspeed = 2.998e18  # AA/s
log_rsun_cgs = np.log10(6.955) + 10
log_lsun_cgs = np.log10(3.839) + 33
log_SB_cgs = np.log10(5.670367e-5)
log_SB_solar = log_SB_cgs + 2 * log_rsun_cgs - log_lsun_cgs


class MILESInterpolator(SimpleSPIModel):

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).
        """
        self.has_errors = False
        with h5py.File(training_data, "r") as f:
            self.library_spectra = f['spectra'][:]
            self.library_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]
            try:
                self.library_snr = self.library_spectra / f['uncertainty'][:]
                self.has_errors = True
            except:
                pass
            ancillary = f['ancillary'][:]
        # add and rename labels here.  Note that not all labels need to be or
        # will be used in the feature generation
        newfield = ['logt', 'miles_id']
        newdata = [np.log10(self.library_labels['teff']), ancillary['miles_id']]
        self.library_labels = rfn.append_fields(self.library_labels,
                                                 newfield, newdata, usemask=False)
        try:
            # assuming f_nu
            fbol = np.trapz(self.library_spectra / self.wavelengths**2, self.wavelengths)
            newfield = ['logl', 'luminosity', 'fbol']
            newdata = [ancillary['logl'], 10**ancillary['logl'], fbol]
            self.library_labels = rfn.append_fields(self.library_labels,
                                                     newfield, newdata, usemask=False)            
        except:
            pass
        self.reset_mask()


class CKCInterpolator(SimpleSPIModel):

    def load_training_data(self, training_data='', renormalize_spec=True,
                           continuum_normalize=False, wlo=0, whi=np.inf, **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).
        """

        self.file_for_training = training_data
        # Read the HDF5 file
        self.has_errors = False
        with h5py.File(training_data, "r") as f:
            try:
                sel = f['parameters']["afe"] == 0
            except:
                sel = slice(None)
                print("getting all spectra")
            self.library_spectra = f['spectra'][:][sel]
            self.library_labels = f['parameters'][:][sel]
            self.wavelengths = f['wavelengths'][:]
            if continuum_normalize:
                self.library_spectra /= f['continuua'][:][sel]

        # Restrict wavelengh range
        gw = (self.wavelengths >= wlo) & (self.wavelengths <= whi)
        if gw.sum() < len(self.wavelengths):
            self.wavelengths = self.wavelengths[gw]
            self.library_spectra = self.library_spectra[:, gw]

        # Deal with oldstyle files that only have Z, not feh
        if 'Z' in self.library_labels.dtype.names:
            newcols = ['feh']
            newdata = [np.log10(self.library_labels['Z']/0.0134)]
            labels = rfn.append_fields(self.library_labels, newcols, newdata, usemask=False)
            self.library_labels = labels

        self.reset_mask()
        # remove nan spectra
        hasnan = np.isnan(self.training_spectra).sum(axis=-1) > 1 # keep spectra where only one pixel is a nan
        if hasnan.sum() > 0:
            self.leave_out(np.where(hasnan)[0])
            self.library_spectra[np.isnan(self.library_spectra)] = -1
        # remove zero spectra
        bad = np.where(np.max(self.library_spectra, axis=-1) <= 1e-33)
        if len(bad[0]) > 0:
            self.leave_out(bad[0])
        # replace negatives with tiny number
        tiny_number = 1e-30
        self.library_spectra[self.library_spectra <= 0.0] = tiny_number
        self.delete_masked()
            
        if renormalize_spec and (not continuum_normalize):
            # renormalize spectra to Lbol = 1 L_sun
            try:
                # Renormalize so that all stars have logl=0
                # The native unit of the C3K library is erg/s/cm^2/Hz/sr.  We
                # need to multply by 4pi (for the sr) and then by a radius
                # (squared) that gets us to logL=0 We can work out this radius
                # from logL=log(4pi\sigma_SB) + 2logR + 4logT
                logl, log4pi = 0.0, np.log10(4 * np.pi)
                # This is in cm
                twologR = (logl+log_lsun_cgs) - 4 * self.library_labels['logt'] - log_SB_cgs - log4pi

                # Now multiply by 4piR^2, with another 4pi for the solid angle
                self.library_spectra *= 10**(twologR[:, None] + 2 * log4pi)
                #self.spectral_units = 'erg/s/Hz/solar luminosity'
            except:
                print('Did not renormalize spectra by luminosity.')
        
    def build_training_info(self):
        self.reference_index = None
        self.reference_spectrum = np.median(self.training_spectra, axis=0)

        self.label_range = {}
        for l in self.label_names:
            self.label_range[l] = np.array([self.training_labels[l].min(),
                                            self.training_labels[l].max()])

    def show_coeffs(self):
        fscale = [np.diff(label_range[j]) for j in self.features[i]]
        for i, f in enumerate(self.features):
            fscale = [np.diff(self.label_range[l]) for l in f]
            c = np.median(self.coeffs[:, i+1])
            print(f, c, np.prod(fscale) * c)
