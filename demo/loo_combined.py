import sys, time
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import h5py

from combined_model import CombinedInterpolator
from psi.comparison_models import PiecewiseC3K
from psi.utils import dict_struct, within, flatten_struct
from psi.plotting import *

from combined_params import bounds, features, pad_bounds


showlines = {'CO': (2.26, 2.35),
             'CaT': (0.845, 0.870),
             'Feh': (0.980, 1.0),
             'NaD': (0.580, 0.596),
             r'H$\beta$': (0.482, 0.492),
             'NaI': (0.816, 0.824)}


def get_stats(wave, observed, predicted, snr,
              wmin=0.38, wmax=2.4, **extras):
    """Calculate useful statistics
    """
    imin = np.argmin(np.abs(wave - wmin))
    imax = np.argmin(np.abs(wave - wmax))
    delta = predicted / observed - 1.0
    chi = delta * snr

    var_spectrum = np.nanvar(delta, axis=0)
    bias_spectrum = np.nanmean(delta, axis=0)
    var_total = np.nanvar(delta[:, imin:imax], axis=1)
    
    chi_bias_spectrum = np.nanmean(chi, axis=0)
    chi_var_spectrum = np.nanvar(chi, axis=0)
    chisq = np.nansum((chi**2)[:,imin:imax], axis=1)

    return chi_bias_spectrum, chi_var_spectrum, chisq


def get_c3k_spectrum(c3k_model, outwave=None, **params):
    """Get a C3k spectrum interpolated to the correct wavelength
    """
    cwave, cspec, _ = c3k_model.get_star_spectrum(**params)
    spec = np.interp(outwave, cwave / 1e4, cspec)
    return spec


def get_interpolator(mlib='', regime='', c3k_weight=1e-1, snr_max=1e3,
                     fake_weights=False, padding=True, mask_mann=True, **kwargs):
    """
    """
    # --- The PSI Model ---
    spi = CombinedInterpolator(training_data=mlib, c3k_weight=c3k_weight,
                               unweighted=False, snr_max=snr_max, logify_flux=True)
    # renormalize by bolometric luminosity
    spi.renormalize_library_spectra(bylabel='luminosity')
    # Use fake, constant SNR for all the MILES spectra
    if fake_weights:
        g = spi.library_snr > 0
        spi.library_snr[g] = 100
    # mask the Mann mdwarf stars for now
    if mask_mann:
        mann = np.where(spi.library_labels['miles_id'] == 'mdwarf')[0]
        spi.leave_out(mann)
    #c3k = np.where(spi.library_labels['miles_id'] == 'c3k')[0]
    # Choose parameter regime and features
    if padding:
        b = pad_bounds(bounds[regime], **kwargs)
    else:
        b = bounds[regime]
    spi.restrict_sample(bounds=b)
    spi.features = features[regime]
    return spi


def leave_one_out(spi, loo_indices, retrain=True, **extras):
    """ --- Leave-one-out ----
    """
    # build output  arrays
    predicted = np.zeros([len(loo_indices), spi.n_wave])
    inhull = np.zeros(len(loo_indices), dtype=bool)
    if not retrain:
        cinside = spi.remove_c3k_inside()
        spi.train()
        inhull = spi.inside_hull(spi.library_labels[loo_indices])
        spi.library_mask[cinside] = True
    # Loop over spectra to leave out and predict
    for i, j in enumerate(loo_indices):
        if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
        # Get full sample and the parameters of the star to leave out
        spec = spi.library_spectra[j, :]
        labels = dict_struct(spi.library_labels[j])
        #labels = dict([(n, tlabels[n]) for n in spi.label_names])
        # Leave one out and re-train
        if retrain:
            spi.library_mask[j] = False
            c3k_inside = spi.remove_c3k_inside()
            inhull[i] = spi.inside_hull(labels)
            spi.train()
        predicted[i, :] = spi.get_star_spectrum(**labels)
        # now put it back
        if retrain:
            spi.library_mask[j] = True
            spi.library_mask[c3k_inside] = True
    return spi, predicted, inhull


def loo(mlib='', regime='Warm Giants', debug=False, outroot=None, nbox=-1, **kwargs):
    """
    """
    if outroot is None:
        pdict= {'regime': regime.replace(' ','_'),
                'unc': not kwargs['fake_weights']}
        pdict.update(**kwargs)
        outroot = '{regime}_unc={unc}_cwght={c3k_weight:04.3f}'.format(**pdict)

    # --- Build models ----
    spi = get_interpolator(mlib, regime=regime, **kwargs)
    clibname = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/irtf/ckc14_irtf.flat.h5'
    c3k_model = PiecewiseC3K(libname=clibname, use_params=['logt', 'logg', 'feh'],
                             verbose=False, n_neighbors=0, log_interp=True,
                             rescale_libparams=False, in_memory=True)


    # --- Leave-one-out retraining ---
    ts = time.time()
    # These are the indices in the full library of the training spectra
    loo_indices = spi.training_indices.copy()
    # Only leave out MILES
    miles = spi.training_labels['miles_id'] != 'c3k'
    loo_indices = loo_indices[miles]
    # But keep track of whether in padded region
    inbounds = np.ones(len(loo_indices), dtype=bool)
    for n, b in bounds[regime].items():
        inbounds = inbounds & within(b, spi.library_labels[loo_indices][n])
    # Now do the leave out, with or without retraining
    spi, predicted, inhull = leave_one_out(spi, loo_indices, **kwargs)
    print('time to retrain {} models: {:.1f}s'.format(len(loo_indices), time.time()-ts))

    # --- Useful arrays and Stats ---
    wave = spi.wavelengths
    observed = spi.library_spectra[loo_indices, :]
    obs_unc = observed / spi.library_snr[loo_indices, :]
    snr = observed / obs_unc
    bias, variance, chisq = get_stats(spi.wavelengths, observed[inbounds,:],
                                      predicted[inbounds,:], snr[inbounds,:], **kwargs)
    sigma = np.sqrt(variance)

    # --- Make Plots ---

    # Plot the bias and variance spectrum
    sfig, sax = bias_variance(wave, bias, sigma, qlabel='\chi')
    sax.set_ylim(max(-100, min(-1, np.nanmin(sigma[100:-100]), np.nanmin(bias[100:-100]))),
                 min(1000, max(30, np.nanmax(bias[100:-100]), np.nanmax(sigma[100:-100]))))
    sfig.savefig('{}_biasvar.pdf'.format(outroot))

    # Plot a map of total variance as a function of label
    labels = spi.library_labels[loo_indices]
    quality, quality_label = np.log10(chisq), r'$log \, \chi^2$'
    mapfig, mapaxes = quality_map(labels[inbounds], quality, quality_label=quality_label)
    mapfig.savefig('{}_qmap.pdf'.format(outroot))

    # plot full SED
    # filename = '{}_sed.pdf'.format(outroot)
    # fstat = specpages(filename, wave, predicted, observed, obs_unc, labels,
    #                   c3k_model=c3k_model, inbounds=inbounds, inhull=inhull
    #                   showlines={'Full SED': (0.37, 2.5)}, show_native=False)
    # plot zoom-ins around individual lines
    # filename = '{}_lines.pdf'.format(outroot)
    # lstat = specpages(filename, wave, predicted, observed, obs_unc, labels,
    #                  c3k_model=c3k_model, inbounds=inbounds, inhull=inhull
    #                  showlines=showlines, show_native=True)
    with PdfPages('{}_sed.pdf'.format(outroot)) as pdf:
        sed = {'Full SED': (0.37, 2.5)}
        for i, j in enumerate(loo_indices):
            if not inbounds[i]:
                continue
            values = dict_struct(spi.library_labels[j])
            ref = get_c3k_spectrum(c3k_model, outwave=spi.wavelengths, **values)
            p, o, u, r = predicted[i,:], observed[i,:], obs_unc[i,:], ref
            if nbox > 0:
                p, o, u, r = [boxsmooth(x, nbox) for x in [p, o, u, r]]
            fig, ax = zoom_lines(spi.wavelengths, p, o,uncertainties=u, c3k=r,
                                 figsize=(8.5, 7), show_native=False, showlines=sed)
            ax.set_xlim(0.35, 2.5)
            values['inhull'] = inhull[i]
            values['inbounds'] = inbounds[i]
            ti = ("{name:s}: teff={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.2f}, In hull={inhull}, In bounds={inbounds}").format(**values)
            fig.suptitle(ti)
            pdf.savefig(fig)
            pl.close(fig)

    # plot zoom ins around individual lines
    with PdfPages('{}_lines.pdf'.format(outroot)) as pdf:
        for i, j in enumerate(loo_indices):
            if not inbounds[i]:
                continue
            values = dict_struct(spi.library_labels[j])
            ref = get_c3k_spectrum(c3k_model, outwave=spi.wavelengths, **values)
            fig, ax = zoom_lines(spi.wavelengths, predicted[i,:], observed[i,:],
                                 uncertainties=obs_unc[i,:], c3k=ref,
                                 showlines=showlines)
            values['inhull'] = inhull[i]
            values['inbounds'] = inbounds[i]
            ti = ("{name:s}: teff={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.2f}, In hull={inhull}, In bounds={inbounds}").format(**values)
            fig.suptitle(ti)
            pdf.savefig(fig)
            pl.close(fig)
            
    print('finished training and plotting in {:.1f}'.format(time.time()-ts))
    
    # --- Write output ---

    spi.dump_coeffs_ascii('{}_coeffs.dat'.format(outroot))
    
    import json
    with h5py.File('{}_results.h5'.format(outroot), 'w') as f:
        w = f.create_dataset('wavelengths', data=wave)
        obs = f.create_dataset('observed', data=observed)
        mod = f.create_dataset('predicted', data=predicted)
        unc = f.create_dataset('uncertainty', data=obs_unc)
        p = f.create_dataset('parameters', data=spi.library_labels[loo_indices])
        f.attrs['terms'] = json.dumps(spi.features)
        f.attrs['bounds'] = json.dumps(bounds[regime])
        c = f.create_dataset('coefficients', data=spi.coeffs)
        r = f.create_dataset('reference_spectrum', data=spi.reference_spectrum)

    return spi, loo_indices, predicted


def run_matrix(**run_params):
    from itertools import product
    nmiles = [78, 15, 68, 6, 35]
    regimes = ['Hot Stars', 'Warm Giants', 'Warm Dwarfs', 'Cool Giants', 'Cool Dwarfs']
    fake_weights = [ False]
    c3k_weight = [1e-9, 1e-3, 1e-2]

    for regime, wght, fake_unc in product(regimes, c3k_weight, fake_weights):
        outroot = 'figures_v2/{}_unc={}_cwght={:04.3f}'.format(regime.replace(' ','_'),
                                                    not fake_unc, wght)
        _ = loo(regime=regime, c3k_weight=wght, fake_weights=fake_unc, outroot=outroot, **run_params)


if __name__ == "__main__":

    run_params = {'retrain': False,
                  'padding': True,
                  'tpad': 500.0, 'gpad': 0.25, 'zpad': 0.1,
                  'snr_max': 300,
                  'mask_mann': True,
                  'mlib': '/Users/bjohnson/Projects/psi/data/combined/culled_lib_w_mdwarfs_w_unc_w_allc3k.h5',
                  'snr_threshold': 1e-10,
                  'nbox': -1,
                  }

    #spi, inds, pred = loo(regime='Cool Dwarfs', c3k_weight=1e-3, fake_weights=False, **run_params)

    run_matrix(**run_params)
