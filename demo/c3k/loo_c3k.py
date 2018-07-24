import sys

import numpy as np
import matplotlib.pyplot as pl

from spi.library_models import CKCInterpolator
from spi.utils import dict_struct, within_bounds
from spi.plotting import get_stats, quality_map, bias_variance, specpages, write_results

from c3k_regimes import bounds, features, pad_bounds

lightspeed = 2.998e18
showlines = {'CaT': (8450, 8700),
             'NaD': (5800, 5960),
             r'H$\beta$': (4820, 4920),
             'U':(3500, 4050),
             'B': (4000, 4500),
             'Mgb': (4900, 5250),
             }


# The SPI Model

def leave_one_out(psi, loo_indices, retrain=True, **extras):
    """ --- Leave-one-out ----
    """
    # build output  arrays
    predicted = np.zeros([len(loo_indices), psi.n_wave])
    inhull = np.zeros(len(loo_indices), dtype=bool)
    if not retrain:
        psi.train()
        inhull = psi.inside_hull(psi.library_labels[loo_indices])
    # Loop over spectra to leave out and predict
    for i, j in enumerate(loo_indices):
        if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
        # Get full sample and the parameters of the star to leave out
        spec = psi.library_spectra[j, :]
        labels = dict_struct(psi.library_labels[j])
        # Leave one out and re-train
        if retrain:
            psi.library_mask[j] = False
            inhull[i] = psi.inside_hull(labels)
            psi.train()
        predicted[i, :] = psi.get_star_spectrum(**labels)
        # Now put it back
        if retrain:
            psi.library_mask[j] = True
    return psi, predicted, inhull



def get_interpolator(mlib='', regime='', snr=100,
                     padding=True, **kwargs):
    """
    """
    # --- The PSI Model ---
    # spectra are normalized by bolometric luminosity
    psi = CKCInterpolator(training_data=mlib, logify_flux=True)
    #psi.renormalize_library_spectra(bylabel='luminosity')
    # Add library_snr?
    if snr is not None:
        psi.library_snr = np.ones_like(psi.library_spectra) * snr
    # Choose parameter regime and features
    if padding:
        b = pad_bounds(bounds[regime], **kwargs)
    else:
        b = bounds[regime]
    psi.restrict_sample(bounds=b)
    psi.features = features[regime]
    return psi


def loo(regime='Warm Giants', mlib='/Users/bjohnson/Codes/SPS/ckc/ckc/spectra/lores/ckc_R10k.h5',
        snr=100, outroot=None, **kwargs):


    if outroot is None:
        pdict= {'regime': regime.replace(' ','_'),
                'snr': snr,
                'mlib': os.path.basename(mlib).replace('.h5', '')}
        outroot = '{mlib}_{regime}_snr{snr}'.format(**pdict)

    psi = get_interpolator(mlib, regime, snr=snr, **kwargs)

    # --- Run leave one out on (almost) everything ---
    loo_indices = psi.training_indices.copy()[::3]
    psi, predicted, inhull = leave_one_out(psi, loo_indices, **kwargs)


    outroot = 'figures/test'
    plotspec = True

    # Do a solar spectrum
    if regime == 'Warm dwarfs':
        psi.train()
        params = psi.training_labels
        ind_solar = np.searchsorted(np.unique(params['logt']), np.log10(5877.0))
        logt_solar = np.unique(params['logt'])[ind_solar]
        solar = (params['logt'] == logt_solar) & (params['feh'] == 0) & (params['logg'] == 4.5)
        specsun = psi.get_star_spectrum(logt=logt_solar, feh=0, logg=4.5)

        import matplotlib.pyplot as pl
        fig, axes = pl.subplots(2, 1)
        axes[0].plot(psi.wavelengths, psi.training_spectra[solar,:][0])
        axes[0].plot(psi.wavelengths, specsun)
        axes[1].plot(psi.wavelengths, specsun / psi.training_spectra[solar,:][0])
        pl.show()


    # --- Useful arrays and Stats ---
    labels = psi.library_labels[loo_indices]
    # Keep track of whether star is in padded region
    inbounds = inhull # | slice(None)
    wave = psi.wavelengths.copy()
    observed = psi.library_spectra[loo_indices, :]
    uncertainty = observed / snr
    wmin, wmax = 0, 4e4
    bias, variance, chisq = get_stats(wave, observed[inbounds, :], predicted[inbounds, :],
                                      snr, wmin=wmin, wmax=wmax)
    sigma = np.sqrt(variance)

    # --- Write output ---
    psi.dump_coeffs_ascii('{}_coeffs.dat'.format(outroot))

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
        fstat = specpages(filename, wave, predicted, observed, uncertainty, labels,
                          c3k_model=None, inbounds=inbounds, inhull=inhull,
                          showlines={'Full SED': (0.37, 2.5)}, show_native=False,
                          tistring=tistring)
        # plot zoom-ins around individual lines
        filename = '{}_lines.pdf'.format(outroot)
        lstat = specpages(filename, wave, predicted, observed, uncertainty, labels,
                          c3k_model=None, inbounds=inbounds, inhull=inhull,
                          showlines=showlines, show_native=True,
                          tistring=tistring)

    print('finished training and plotting in {:.1f}'.format(time.time()-ts))

    best, worst = plot_best_worst(psi, loo_indices[inbounds],
                                  predicted[inbounds, :], chisq, nshow=5)

    best.show()
    worst.show()

    sys.exit()


if __name__ == "__main__":

    try:
        test = sys.argv[1] == 'test'
    except(IndexError):
        test = False

    run_params = {'retrain': True,
                  'padding': True,
                  'tpad': 500.0, 'gpad': 0.25, 'zpad': 0.1,
                  'snr': 100,
                  'mlib': '/Users/bjohnson/Codes/SPS/ckc/ckc/spectra/lores/irtf/ckc14_irtf.flat.h5',
                  'nbox': -1,
                  }

    if test:
        print('Test mode')
        psi, inds, pred = loo(regime='Warm Dwarfs', outroot='test', **run_params)
    else:
        run_matrix(**run_params)
