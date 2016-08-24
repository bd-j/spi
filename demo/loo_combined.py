import sys
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import h5py
from psi.library_models import SimplePSIModel
from psi.utils import dict_struct

lightspeed = 2.998e18
from combined_params import bounds, features

class CombinedInterpolator(SimplePSIModel):

    def load_training_data(self, training_data='', ckc_weight=1e-1, **extras):
        # --- read the data ---
        with h5py.File(training_data, "r") as f:
            self.wavelengths = f['wavelengths'][:]
            self.library_spectra = f['spectra'][:]
            self.library_labels = f['parameters'][:]
            unc = f['uncertainties'][:]

        # --- do relative weighting --
        c3k = self.library_labels['miles_id'] == 'c3k'
        # median miles uncertainty at each wavelength
        umed = np.nanmedian(unc[~c3k, :], axis=0)
        unc[c3k,:] = (umed / np.sqrt(ckc_weight))[None,:]

        self.library_weights = 1 / unc**2
        self.has_errors = True
        self.reset_mask()

    def build_training_info(self):
        self.reference_index = None
        self.reference_spectrum = self.training_spectra.std(axis=0)


showlines = {'CO': (2.26, 2.35),
             'CaT': (0.845, 0.870),
             'Feh': (0.980, 1.0),
             'NaD': (0.580, 0.596)
             }
props = dict(boxstyle='round', facecolor='w', alpha=0.5)

def zoom_lines(wave, predicted, observed, showlines={}):
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    fig, axes = pl.subplots(2,2, figsize=(10.0, 8.5))
    for i, (line, (lo, hi)) in enumerate(showlines.items()):
        ax = axes.flat[i]
        g = (wave > lo) & (wave < hi)
        ax.plot(wave[g], observed[g], label='Observed')
        ax.plot(wave[g], predicted[g], color='k', label='PSI miles+c3k')
        delta = predicted[g]/observed[g]
        residual, rms = np.median(delta), (delta - 1).std()
        ax.plot(wave[g], predicted[g] / residual, color='k',
                linestyle='--', label='PSI shifted')
        values = [100*(residual-1), 100*(delta-1).std(), line]
        label='Offset: {:4.2f}%\nRMS: {:4.2f}%\n{}'.format(*values)
        ax.text(0.75, 0.20, label,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        if i == 0:
            ax.legend(loc=0, prop={'size':8})
        pl.setp(ax.xaxis.get_majorticklabels(), rotation=35,
                        horizontalalignment='center')
    return fig, axes

        
if __name__ == "__main__":

    ckc_weight = 1e-1 # the relative weight of the CKC models compared to the MILES models.
    regime = 'Warm Giants'

    # The PSI Model
    mlib = '/Users/bjohnson/Projects/psi/data/combined/with_c3k_with_mdwarfs_culled_lib_snr_cut.h5'
    spi = CombinedInterpolator(training_data=mlib, ckc_weight=1e-1,
                               unweighted=False, logify_flux=True)
    spi.renormalize_library_spectra(bylabel='luminosity')
    spi.select(bounds=bounds[regime], delete=False)
    spi.features = features[regime]

    # --- Leave-one-out ----
    # These are the indices in the full library of the training spectra
    loo_indices = spi.training_indices.copy()
    # Only leave out MILES
    miles = spi.training_labels['miles_id'] != 'c3k'
    loo_indices = loo_indices[miles]
    observed = spi.library_spectra[loo_indices, :]
    predicted = np.zeros([len(loo_indices), spi.n_wave])
    inhull = np.zeros(len(loo_indices), dtype=bool)
    # Loop over spectra to leave out and predict
    for i, j in enumerate(loo_indices):
        if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
        # Get full sample and the parameters of the star to leave out
        spec = spi.library_spectra[j, :]
        tlabels = spi.library_labels[j]
        labels = dict([(n, tlabels[n]) for n in spi.label_names])
        # Leave one out and re-train
        spi.library_mask[j] = False
        spi.train()
        predicted[i, :] = spi.get_star_spectrum(**labels)
        inhull[i] = spi.inside_hull(labels)
        # now put it back
        spi.library_mask[j] = True

    # --- Calculate statistics ---
    # get fractional residuals
    wmin, wmax = 3800, 7200
    imin = np.argmin(np.abs(spi.wavelengths - wmin))
    imax = np.argmin(np.abs(spi.wavelengths - wmax))
    imin, imax = 0, len(spi.wavelengths) -1
    delta = predicted / observed - 1.0

    var_spectrum = np.nanvar(delta, axis=0)
    var_total = np.nanvar(delta[:, imin:imax], axis=1)
    # Get chi^2
    snr = 100
    chisq =np.nansum( ((snr * delta)**2)[:,imin:imax], axis=1)

    # --- Make Plots ---

    # Plot the variance spectrum
    fig, ax = pl.subplots()
    ax.plot(spi.wavelengths, var_spectrum*100)
    ax.set_ylim(0.01, 100)
    ax.set_yscale('log')
    ax.set_ylabel('RMS (%)')
    # Plot a map of total variance as a function of label
    l1, l2, l3 = 'logt', 'feh', 'logg'
    lab = spi.library_labels[loo_indices]
    mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))
    #varc = np.clip(np.sqrt(var_total[loo_indices])*100, 0, 1e3)
    varc = np.log(chisq)
    sc = mapaxes[0].scatter(lab[l1], lab[l2], marker='o', c=varc)
    mapaxes[0].set_xlabel(l1)
    mapaxes[0].set_ylabel(l2)
    sc = mapaxes[1].scatter(lab[l1], lab[l3], marker='o', c=varc)
    mapaxes[1].set_xlabel(l1)
    mapaxes[1].set_ylabel(l3)
    mapaxes[1].invert_yaxis()
    cbar = pl.colorbar(sc)
    #cbar.ax.set_ylabel('Fractional RMS (%)')
    cbar.ax.set_ylabel(r'$log \, \chi^2$ (S/N={})'.format(snr))
    [ax.invert_xaxis() for ax in mapaxes]
    mapfig.show()                   
    #mapfig.savefig('figures/residual_map.pdf')

    # plot zoom ins around individual lines
    with PdfPages('{}_lines.pdf'.format(regime.replace(' ','_'))) as pdf:
        for i, j in enumerate(loo_indices):
            fig, ax = zoom_lines(spi.wavelengths, predicted[i,:], observed[i,:],
                                 showlines=showlines)
            fig.suptitle("{}: In hull={}".format(spi.library_labels[j]['name'], inhull[i]))
            pdf.savefig(fig)
            
