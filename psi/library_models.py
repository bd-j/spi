import numpy as np
from numpy.lib import recfunctions as rfn
import h5py
from .model import SimplePSIModel


__all__ = ["MILESInterpolator", "CKCInterpolator"]

lightspeed = 2.998e18  # AA/s
log_rsun_cgs = np.log10(6.955) + 10
log_lsun_cgs = np.log10(3.839) + 33
log_SB_cgs = np.log10(5.670367e-5)
log_SB_solar = log_SB_cgs + 2 * log_rsun_cgs - log_lsun_cgs


class MILESInterpolator(SimplePSIModel):

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


class CKCInterpolator(SimplePSIModel):

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).
        """
        # Read the HDF5 file
        self.has_errors = False
        with h5py.File(training_data, "r") as f:
            self.library_spectra = f['spectra'][:]
            self.library_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]

        # renormalize spectra to Lbol = 1 L_sun
        try:
            # Renormalize so that all stars have logl=0
            # The native unit of the C3K library is erg/s/cm^2/Hz/sr.
            # We need to multply by 4pi (for the sr) and then by a radius
            # (squared) that gets us to logL=0
            # We can work out this radius from logL=log(4pi\sigma_SB) + 2logR + 4logT
            logl, log4pi = 0.0, np.log10(4 * np.pi)
            # This is in cm
            twologR = (logl+log_lsun_cgs) - 4 * self.library_labels['logt'] - log_SB_cgs - log4pi

            # Now multiply by 4piR^2, with another 4pi for the solid angle
            self.library_spectra *= 10**(twologR[:, None] + 2 * log4pi)
            #self.spectral_units = 'erg/s/Hz/solar luminosity'

        except:
            print('Did not renormalize spectra by luminosity.')

        self.reset_mask()
        # remove zero spectra
        bad = np.where(np.max(self.library_spectra, axis=-1) <= 0)
        if len(bad[0]) > 0:
            self.leave_out(bad[0])
        
    def build_training_info(self):
        self.reference_index = None
        self.reference_spectrum = self.training_spectra.mean(axis=0)

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
