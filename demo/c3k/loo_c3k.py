import sys

import numpy as np
import matplotlib.pyplot as pl

from psi.library_models import CKCInterpolator
from psi.utils import dict_struct, within_bounds
from psi.plotting import get_stats, quality_map, bias_variance, specpages

from c3k_regimes import bounds, features, pad_bounds

lightspeed = 2.998e18
showlines = {'CaT': (8450, 8700),
             'NaD': (5800, 5960),
             r'H$\beta$': (4820, 4920),
             'U':(3500, 4050),
             'B': (4000, 4500),
             'Mgb': (4900, 5250),
             }


# The PSI Model
regime = 'Warm Giants'
mlib = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/ckc_R10k.h5'
#mlib = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/irtf/ckc14_irtf.flat.h5'
spi = CKCInterpolator(training_data=mlib, logify_flux=True)
spi.delete_masked()
spi.features = features[regime]
spi.select(bounds=pad_bounds(bounds[regime]))


def leave_one_out(spi, loo_indices, retrain=True, **extras):
    """ --- Leave-one-out ----
    """
    # build output  arrays
    predicted = np.zeros([len(loo_indices), spi.n_wave])
    inhull = np.zeros(len(loo_indices), dtype=bool)
    if not retrain:
        spi.train()
        inhull = spi.inside_hull(spi.library_labels[loo_indices])
    # Loop over spectra to leave out and predict
    for i, j in enumerate(loo_indices):
        if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
        # Get full sample and the parameters of the star to leave out
        spec = spi.library_spectra[j, :]
        labels = dict_struct(spi.library_labels[j])
        # Leave one out and re-train
        if retrain:
            spi.library_mask[j] = False
            inhull[i] = spi.inside_hull(labels)
            spi.train()
        predicted[i, :] = spi.get_star_spectrum(**labels)
        # Now put it back
        if retrain:
            spi.library_mask[j] = True
    return spi, predicted, inhull


if __name__ == "__main__":

    outroot = 'figures/test'
    plotspec = True

    # Do a solar spectrum
    if regime == 'Warm dwarfs':
        spi.train()
        params = spi.training_labels
        ind_solar = np.searchsorted(np.unique(params['logt']), np.log10(5877.0))
        logt_solar = np.unique(params['logt'])[ind_solar]
        solar = (params['logt'] == logt_solar) & (params['feh'] == 0) & (params['logg'] == 4.5)
        specsun = spi.get_star_spectrum(logt=logt_solar, feh=0, logg=4.5)

        import matplotlib.pyplot as pl
        fig, axes = pl.subplots(2, 1)
        axes[0].plot(spi.wavelengths, spi.training_spectra[solar,:][0])
        axes[0].plot(spi.wavelengths, specsun)
        axes[1].plot(spi.wavelengths, specsun / spi.training_spectra[solar,:][0])
        pl.show()
        
    #sys.exit()

    # --- Run leave one out on (almost) everything ---
    loo_indices = spi.training_indices.copy()[::3]
    spi, predicted, inhull = leave_one_out(spi, loo_indices, retrain=False)

    # --- Useful arrays and Stats ---
    labels = spi.library_labels[loo_indices]
    # Keep track of whether MILES stars in padded region
    #inbounds = within_bounds(bounds[regime], labels)
    inbounds = inhull #slice(None)
    wave = spi.wavelengths.copy()
    observed = spi.library_spectra[loo_indices, :]
    obs_unc = np.zeros_like(observed)
    snr = 100.0
    wmin, wmax = 0, 4e4
    bias, variance, chisq = get_stats(wave, observed[inbounds, :],
                                      predicted[inbounds, :], snr,
                                      wmin=wmin, wmax=wmax)
    sigma = np.sqrt(variance)

    # --- Write output ---
    spi.dump_coeffs_ascii('{}_coeffs.dat'.format(outroot))
    #write_results(outroot, spi, fgk_bounds,
    #              wave, predicted, observed, obs_unc, labels, **kwargs)

    # --- Make Plots ---

    # Plot the bias and variance spectrum
    sfig, sax = bias_variance(wave, bias, sigma, qlabel='\chi')
    sax.set_ylim(max(-100, min(-1, np.nanmin(sigma[100:-100]), np.nanmin(bias[100:-100]))),
                 min(1000, max(30, np.nanmax(bias[100:-100]), np.nanmax(sigma[100:-100]))))
    sfig.savefig('{}_biasvar.pdf'.format(outroot))
    
    # Plot a map of "quality" as a function of label
    quality, quality_label = np.log10(chisq), r'$log \, \chi^2$'
    mapfig, mapaxes = quality_map(labels[inbounds], quality,
                                  quality_label=quality_label, add_offsets=True)
    mapfig.savefig('{}_qmap.pdf'.format(outroot))
    if plotspec:
        tistring = "logt={logt:4.0f}, logg={logg:3.2f}, feh={feh:3.2f}, In hull={inhull}"
        # plot full SED
        filename = '{}_sed.pdf'.format(outroot)
        fstat = specpages(filename, wave, predicted, observed, obs_unc, labels,
                          c3k_model=None, inbounds=inbounds, inhull=inhull,
                          showlines={'Full SED': (0.37, 2.5)}, show_native=False,
                          tistring=tistring)
        # plot zoom-ins around individual lines
        filename = '{}_lines.pdf'.format(outroot)
        lstat = specpages(filename, wave, predicted, observed, obs_unc, labels,
                          c3k_model=None, inbounds=inbounds, inhull=inhull,
                          showlines=showlines, show_native=True,
                          tistring=tistring)

    print('finished training and plotting in {:.1f}'.format(time.time()-ts))

    
    # Plot the n worst and n best
    nshow = 5
    sorted_inds = loo_indices[np.argsort(chisq[loo_indices])]
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)

    bestfig, bestax = pl.subplots(nshow, 2, sharex=True)
    for i, j in enumerate(sorted_inds[:nshow]):
        tlab = dict_struct(spi.training_labels[j])
        tlab = 'logT={logt:4.3f}, [Fe/H]={feh:3.1f}, logg={logg:3.2f}'.format(**tlab)
        ax = bestax[i, 0]
        ax.plot(spi.wavelengths, predicted[j, :])
        ax.plot(spi.wavelengths, spi.training_spectra[j, :])
        ax.text(0.1, 0.05, tlab, transform=ax.transAxes, verticalalignment='top', bbox=props)
        bestax[i,1].plot(spi.wavelengths, (predicted[j,:]/spi.training_spectra[j, :] - 1)*100)

    worstfig, worstax = pl.subplots(nshow, 2, sharex=True)
    for i, j in enumerate(sorted_inds[-nshow:]):
        tlab = dict_struct(spi.training_labels[j])
        tlab = 'logT={logt:4.3f}, [Fe/H]={feh:3.1f}, logg={logg:3.2f}'.format(**tlab)
        ax = worstax[i, 0]
        ax.plot(spi.wavelengths, predicted[j, :])
        ax.plot(spi.wavelengths, spi.training_spectra[j, :])
        ax.text(0.1, 0.05, tlab, transform=ax.transAxes, verticalalignment='bottom', bbox=props)
        worstax[i,1].plot(spi.wavelengths, (predicted[j,:]/spi.training_spectra[j, :] - 1)*100)

    bestfig.show()
    worstfig.show()

    sys.exit()
