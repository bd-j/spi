import sys, time
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import h5py

from combined_model import CombinedInterpolator
from spi.comparison_models import PiecewiseC3K
from spi.utils import dict_struct, within_bounds
from spi.plotting import get_stats, quality_map, bias_variance, specpages

from combined_params import bounds, features, pad_bounds


showlines = {'CO': (2.26, 2.35),
             'CaT': (0.845, 0.870),
             'Feh': (0.980, 1.0),
             'NaD': (0.580, 0.596),
             r'H$\beta$': (0.482, 0.492),
             'NaI': (0.816, 0.824)}


def get_interpolator(mlib='', regime='', c3k_weight=1e-1, snr_max=1e3,
                     fake_weights=False, padding=True, mask_mann=True, **kwargs):
    """
    """
    # --- The PSI Model ---
    psi = CombinedInterpolator(training_data=mlib, c3k_weight=c3k_weight,
                               unweighted=False, snr_max=snr_max, logify_flux=True)
    # renormalize by bolometric luminosity
    psi.renormalize_library_spectra(bylabel='luminosity')
    # Use fake, constant SNR for all the MILES spectra
    if fake_weights:
        g = psi.library_snr > 0
        psi.library_snr[g] = 100
    # mask the Mann mdwarf stars for now
    if mask_mann:
        mann = np.where(psi.library_labels['miles_id'] == 'mdwarf')[0]
        psi.leave_out(mann)
    #c3k = np.where(psi.library_labels['miles_id'] == 'c3k')[0]
    # Choose parameter regime and features
    if padding:
        b = pad_bounds(bounds[regime], **kwargs)
    else:
        b = bounds[regime]
    psi.restrict_sample(bounds=b)
    psi.features = features[regime]
    return psi


def leave_one_out(psi, loo_indices, retrain=True, **extras):
    """ --- Leave-one-out ----
    """
    # build output  arrays
    predicted = np.zeros([len(loo_indices), psi.n_wave])
    inhull = np.zeros(len(loo_indices), dtype=bool)
    if not retrain:
        cinside = psi.remove_c3k_inside()
        psi.train()
        inhull = psi.inside_hull(psi.library_labels[loo_indices])
        psi.library_mask[cinside] = True
    # Loop over spectra to leave out and predict
    for i, j in enumerate(loo_indices):
        if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
        # Get full sample and the parameters of the star to leave out
        spec = psi.library_spectra[j, :]
        labels = dict_struct(psi.library_labels[j])
        #labels = dict([(n, tlabels[n]) for n in psi.label_names])
        # Leave one out and re-train
        if retrain:
            psi.library_mask[j] = False
            c3k_inside = psi.remove_c3k_inside()
            inhull[i] = psi.inside_hull(labels)
            psi.train()
        predicted[i, :] = psi.get_star_spectrum(**labels)
        # now put it back
        if retrain:
            psi.library_mask[j] = True
            psi.library_mask[c3k_inside] = True
    return psi, predicted, inhull


def loo(regime='Warm Giants', outroot=None, nbox=-1, plotspec=True, **kwargs):
    """
    """
    if outroot is None:
        pdict= {'regime': regime.replace(' ','_'),
                'unc': not kwargs['fake_weights']}
        pdict.update(**kwargs)
        outroot = '{regime}_unc={unc}_cwght={c3k_weight:04.3f}'.format(**pdict)

    # --- Build models ----

    psi = get_interpolator(regime=regime, **kwargs)
    clibname = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/irtf/ckc14_irtf.flat.h5'
    c3k_model = PiecewiseC3K(libname=clibname, use_params=['logt', 'logg', 'feh'],
                             verbose=False, n_neighbors=1, log_interp=True,
                             rescale_libparams=False, in_memory=True)

    # --- Leave-one-out retraining ---

    ts = time.time()
    # These are the indices in the full library of the training spectra
    loo_indices = psi.training_indices.copy()
    # Only leave out MILES
    miles = psi.training_labels['miles_id'] != 'c3k'
    loo_indices = loo_indices[miles]
    # Now do the leave out, with or without retraining
    psi, predicted, inhull = leave_one_out(psi, loo_indices, **kwargs)

    print('time to retrain {} models: {:.1f}s'.format(len(loo_indices), time.time()-ts))

    # --- Useful arrays and Stats ---

    labels = psi.library_labels[loo_indices]
    # Keep track of whether MILES stars in padded region
    inbounds = within_bounds(bounds[regime], labels)
    wave = psi.wavelengths.copy()
    observed = psi.library_spectra[loo_indices, :]
    obs_unc = observed / psi.library_snr[loo_indices, :]
    snr = observed / obs_unc
    bias, variance, chisq = get_stats(wave, observed[inbounds,:],
                                      predicted[inbounds,:], snr[inbounds,:], **kwargs)
    sigma = np.sqrt(variance)

    # --- Write output ---

    psi.dump_coeffs_ascii('{}_coeffs.dat'.format(outroot))
    write_results(outroot, psi, bounds[regime],
                  wave, predicted, observed, obs_unc, labels, **kwargs)

    # --- Make Plots ---

    # Plot the bias and variance spectrum
    sfig, sax = bias_variance(wave, bias, sigma, qlabel='\chi')
    sax.set_ylim(max(-100, min(-1, np.nanmin(sigma[100:-100]), np.nanmin(bias[100:-100]))),
                 min(1000, max(30, np.nanmax(bias[100:-100]), np.nanmax(sigma[100:-100]))))
    sfig.savefig('{}_biasvar.pdf'.format(outroot))
    # Plot a map of total variance as a function of label
    quality, quality_label = np.log10(chisq), r'$log \, \chi^2$'
    mapfig, mapaxes = quality_map(labels[inbounds], quality, quality_label=quality_label)
    mapfig.savefig('{}_qmap.pdf'.format(outroot))
    if plotspec:
        # plot full SED
        filename = '{}_sed.pdf'.format(outroot)
        fstat = specpages(filename, wave, predicted, observed, obs_unc, labels,
                          c3k_model=c3k_model, inbounds=inbounds, inhull=inhull,
                          showlines={'Full SED': (0.37, 2.5)}, show_native=False)
        # plot zoom-ins around individual lines
        filename = '{}_lines.pdf'.format(outroot)
        lstat = specpages(filename, wave, predicted, observed, obs_unc, labels,
                          c3k_model=c3k_model, inbounds=inbounds, inhull=inhull,
                          showlines=showlines, show_native=True)

    print('finished training and plotting in {:.1f}'.format(time.time()-ts))

    return psi, loo_indices, predicted


def write_results(outroot, psi, bounds, wave, pred, obs, unc, labels, **extras):
    import json
    with h5py.File('{}_results.h5'.format(outroot), 'w') as f:
        w = f.create_dataset('wavelengths', data=wave)
        o = f.create_dataset('observed', data=obs)
        p = f.create_dataset('predicted', data=pred)
        u = f.create_dataset('uncertainty', data=unc)
        l = f.create_dataset('parameters', data=labels)
        f.attrs['terms'] = json.dumps(psi.features)
        f.attrs['bounds'] = json.dumps(bounds)
        f.attrs['options'] = json.dumps(extras)
        c = f.create_dataset('coefficients', data=psi.coeffs)
        r = f.create_dataset('reference_spectrum', data=psi.reference_spectrum)


def run_matrix(**run_params):
    from itertools import product
    nmiles = [78, 15, 68, 6, 35]
    regimes = ['Hot Stars', 'Warm Giants', 'Warm Dwarfs', 'Cool Giants', 'Cool Dwarfs']
    fake_weights = [ False]
    c3k_weight = [1e-9, 1e-3, 1e-2]

    for regime, wght, fake_unc in product(regimes, c3k_weight, fake_weights):
        outroot = 'results/figures_v5b/{}_unc={}_cwght={:04.3f}'.format(regime.replace(' ','_'),
                                                    not fake_unc, wght)
        _ = loo(regime=regime, c3k_weight=wght, fake_weights=fake_unc, outroot=outroot, **run_params)


if __name__ == "__main__":

    try:
        test = sys.argv[1] == 'test'
    except(IndexError):
        test = False

    run_params = {'retrain': False,
                  'padding': True,
                  'tpad': 500.0, 'gpad': 0.25, 'zpad': 0.1,
                  'snr_max': 300,
                  'mask_mann': False,
                  'mlib': '/Users/bjohnson/Projects/spi/data/combined/culled_libv5_w_mdwarfs_w_unc_w_allc3k.h5',
                  'snr_threshold': 1e-10,
                  'nbox': -1,
                  }

    if test:
        print('Test mode')
        psi, inds, pred = loo(regime='Warm Dwarfs', c3k_weight=1e-3, fake_weights=False,
                              outroot='test', **run_params)
    else:
        run_matrix(**run_params)
