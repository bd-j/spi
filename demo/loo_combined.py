import sys, time
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import h5py
from psi.model import SimplePSIModel
from psi.utils import dict_struct
from psi.plotting import *

lightspeed = 2.998e18
from combined_params import bounds, features

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


def run_matrix():
    from itertools import product
    nmiles = [78, 15, 68, 6, 35]
    regimes = ['Warm Giants', 'Cool Giants', 'Warm Dwarfs', 'Cool Dwarfs', 'Hot Stars']
    fake_weights = [ True, False ]
    c3k_weight = [1e-3, 1e-1, 10.0]

    for regime, wght, fake_unc in product(regimes, c3k_weight, fake_weights):
        outroot = '{}_unc={}_cwght={:03.2f}'.format(regime.replace(' ','_'),
                                                not fake_unc, wght)
        _ = loo(regime=regime, c3k_weight=wght, fake_weights=fake_unc, outroot=outroot)


def loo(regime='Warm Giants',c3k_weight=1e-1, # the relative weight of the CKC models compared to the MILES models.
        fake_weights=False, snr_max=1e3, outroot='test'):

#if __name__ == "__main__":
#    c3k_weight = 1e-1 # the relative weight of the CKC models compared to the MILES models.
#    regime = 'Warm Giants'
#    fake_weights = False
#    snr_max = 1e3
    ts = time.time()
#    outroot = '{}_unc={}_cwght={:03.2f}'.format(regime.replace(' ','_'),
#                                                not fake_weights, c3k_weight)

    # --- The PSI Model ---
    mlib = '/Users/bjohnson/Projects/psi/data/combined/culled_lib_w_mdwarfs_w_unc_w_c3k.h5'
    spi = CombinedInterpolator(training_data=mlib, c3k_weight=c3k_weight,
                               unweighted=False, snr_max=snr_max, logify_flux=True)
    # renormalize by bolometric luminosity
    spi.renormalize_library_spectra(bylabel='luminosity')
    # Use fake, constant SNR for all the MILES spectra
    if fake_weights:
        g = spi.library_snr > 0
        spi.library_snr[g] = 100
    # mask the Mann mdwarf stars for now
    mann = np.where(spi.library_labels['miles_id'] == 'mdwarf')[0]
    c3k = np.where(spi.library_labels['miles_id'] == 'c3k')[0]
    spi.leave_out(mann)
    # Choose parameter regime and features
    spi.restrict_sample(bounds=bounds[regime])
    spi.features = features[regime]

    # --- Leave-one-out ----
    # These are the indices in the full library of the training spectra
    loo_indices = spi.training_indices.copy()
    # Only leave out MILES
    miles = spi.training_labels['miles_id'] != 'c3k'
    loo_indices = loo_indices[miles]
    # build output and other useful arrays
    observed = spi.library_spectra[loo_indices, :]
    obs_unc = observed / spi.library_snr[loo_indices, :]
    predicted = np.zeros([len(loo_indices), spi.n_wave])
    inhull = np.zeros(len(loo_indices), dtype=bool)
    # Loop over spectra to leave out and predict
    for i, j in enumerate(loo_indices):
        if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
        # Get full sample and the parameters of the star to leave out
        spec = spi.library_spectra[j, :]
        labels = dict_struct(spi.library_labels[j])
        #labels = dict([(n, tlabels[n]) for n in spi.label_names])
        # Leave one out and re-train
        spi.library_mask[j] = False
        spi.train()
        predicted[i, :] = spi.get_star_spectrum(**labels)
        inhull[i] = spi.inside_hull(labels)
        # now put it back
        spi.library_mask[j] = True

        
    print('time to retrain {} models: {:.1f}s'.format(len(loo_indices), time.time()-ts))
        
    # --- Calculate statistics ---
    # get fractional residuals
    wmin, wmax = 0.38, 2.4
    imin = np.argmin(np.abs(spi.wavelengths - wmin))
    imax = np.argmin(np.abs(spi.wavelengths - wmax))
    #imin, imax = 0, len(spi.wavelengths) - 1
    delta = predicted / observed - 1.0
    snr = observed / obs_unc
    # snr = np.clip(snr, 0, 100)
    chi = delta * snr

    var_spectrum = np.nanvar(delta, axis=0)
    bias_spectrum = np.nanmean(delta, axis=0)
    chi_mean_spectrum = np.nanmean(chi, axis=0)
    chi_var_spectrum = np.nanvar(chi, axis=0)
    
    var_total = np.nanvar(delta[:, imin:imax], axis=1)
    # Get chi^2
    #snr = 100
    chisq = np.nansum((chi**2)[:,imin:imax], axis=1)

    # --- Make Plots ---

    # Plot the bias and variance spectrum
    colors = pl.rcParams['axes.color_cycle']
    sfig, sax = pl.subplots()
    sax.plot(spi.wavelengths, np.sqrt(chi_var_spectrum), label='$\sigma_\chi$',
             color=colors[0])
    sax.plot(spi.wavelengths, chi_mean_spectrum, label=r'$\bar{\chi}$',
             color=colors[1])
    sax.axhline(1, linestyle=':', color=colors[0])
    sax.axhline(0, linestyle=':', color=colors[1])
    sax.set_ylim(-5, 30)
    sax.set_ylabel('$\chi$ (predicted - observed)')
    #sax.plot(spi.wavelengths, np.sqrt(var_spectrum)*100, label='Dispersion')
    #sax.plot(spi.wavelengths, np.abs(bias_spectrum)*100, label='Mean absolute offset')
    #sax.set_ylim(0.001, 100)
    #sax.set_yscale('log')
    #sax.set_ylabel('%')
    sax.legend(loc=0)
    sfig.savefig('{}_biasvar.pdf'.format(outroot))

    # Plot a map of total variance as a function of label
    labels = spi.library_labels[loo_indices]
    quality, quality_label = np.log10(chisq), r'$log \, \chi^2$'
    mapfig, mapaxes = quality_map(labels, quality, quality_label=quality_label)
    mapfig.savefig('{}_qmap.pdf'.format(outroot))

    # plot zoom ins around individual lines
    with PdfPages('{}_lines.pdf'.format(outroot)) as pdf:
        for i, j in enumerate(loo_indices):
            fig, ax = zoom_lines(spi.wavelengths, predicted[i,:], observed[i,:],
                                 uncertainties=obs_unc[i,:], showlines=showlines)
            values = dict_struct(spi.library_labels[j])
            values['inhull'] = inhull[i]
            ti = "{name:s}: teff={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.2f}, In hull={inhull}".format(**values)
            fig.suptitle(ti)
            pdf.savefig(fig)
            pl.close(fig)
            
    print('finished training and plotting in {:.1f}'.format(time.time()-ts))
    
    # --- Write output ---
    import json
    with h5py.File('{}_results.h5'.format(outroot), 'w') as f:
        w = f.create_dataset('wavelengths', data=spi.wavelengths)
        obs = f.create_dataset('observed', data=observed)
        mod = f.create_dataset('predicted', data=predicted)
        unc = f.create_dataset('uncertainty', data=obs_unc)
        p = f.create_dataset('parameters', data=spi.library_labels[loo_indices])
        f.attrs['terms'] = json.dumps(spi.features)

    return spi

if __name__ == "__main__":
    #loo(regime='Warm Giants', c3k_weight=1e-1, fake_weights=False)
    run_matrix()
