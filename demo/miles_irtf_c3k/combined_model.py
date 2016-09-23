import numpy as np
import h5py
from scipy.spatial import Delaunay

from psi.model import SimplePSIModel
from psi.utils import flatten_struct


class CombinedInterpolator(SimplePSIModel):

    def load_training_data(self, training_data='', c3k_weight=1e-1,
                           snr_threshold=1e-10, snr_max=np.inf, **extras):
        # --- read the data ---
        with h5py.File(training_data, "r") as f:
            self.wavelengths = f['wavelengths'][:]
            self.library_spectra = f['spectra'][:]
            self.library_labels = f['parameters'][:]
            unc = f['uncertainties'][:]

        # Weighting stuff
        self.c3k_weight = c3k_weight
        self.library_snr = self.library_spectra / unc #* 0.0 + 1.0
        self.has_errors = True

        # enforce a max S/N
        if snr_max < np.inf:
            self.library_snr = 1 / np.hypot(1.0 / self.library_snr,
                                            1.0 / snr_max) # np.clip(self.library_snr, 0, snr_max)

        # --- set negative (or very low) S/N fluxes to zero weight ---
        bad = ((self.library_snr < snr_threshold) |
               (~np.isfinite(self.library_snr)) |
               (self.library_spectra < 0)
               )
        self.bad_flux_value = np.nanmedian(self.library_spectra[~bad])
        self.library_spectra[bad] = self.bad_flux_value
        self.library_snr[bad] = 0.0
        self.reset_mask()

    def get_weights(self, ind_wave, spec):
        """
        :param spec:
            Flux in linear units of the training spectra
        """

        if (not self.has_errors) or (self.unweighted):
            return None
        else:
            if self.logify_flux:
                # if training log(flux), use relative (S/N)**2 for weights
                relative_weights = self.training_snr[:, ind_wave]**2
            else:
                # else just use the inverse flux variance (S/N)**2 /S**2 
                relative_weights = (self.training_snr[:, ind_wave] / spec)**2

            # --- do relative weighting of c3k ---
            c3k = (self.training_labels['miles_id'] == 'c3k')
            # median of MILES weights.  If zero, just give c3k full weight 
            wmiles = np.nanmedian(relative_weights[~c3k, :], axis=0)
            wmiles[wmiles == 0.] = 1.0
            relative_weights[c3k, :] = (wmiles * self.c3k_weight)[None, :]
                      
            return relative_weights
        
    def build_training_info(self):
        self.reference_index = None
        self.reference_spectrum = self.training_spectra.std(axis=0)

    def remove_c3k_inside(self):
        """Returns a boolean array of same length as secondary describing
        whether that element of secondary is outside the hull formed by primary
        """
        c3k = self.training_labels['miles_id'] == 'c3k'
        miles = ~c3k
        L = flatten_struct(self.training_labels[miles], use_labels=self.used_labels)
        l = flatten_struct(self.training_labels[c3k], use_labels=self.used_labels)
        hull = Delaunay(L.T)
        inside = hull.find_simplex(l.T) >= 0

        #inside = self.inside_hull(self.training_labels[c3k])
        bad = self.training_indices[c3k][inside]
        self.leave_out(bad)
        return bad
