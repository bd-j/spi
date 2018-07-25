import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

from .utils import dict_struct

__all__ = ["get_stats", "bias_variance", "quality_map",
           "specpages", "zoom_lines",
           "boxsmooth", "get_c3k_spectrum",
           "flux_teff"]

props = dict(boxstyle='round', facecolor='w', alpha=0.5)


def boxsmooth(x, width=10):
    if x is None:
        return None
    return np.convolve(x, np.ones(width*1.0) / width, mode='same')


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


_titlestring = "{name:s}: teff={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.2f}, In hull={inhull}, In bounds={inbounds}"
def specpages(filename, wave, pred, obs, unc, labels,
              c3k_model=None, inbounds=None, inhull=None,
              show_outbounds=True, tistring=_titlestring, nbox=-1, **kwargs):
    """Given arrays of spectra and labels for a set of stars, and regions to
    plot, make a multipage PDF with those spectra plotted, one page for each
    star, one panel for each spectral region.

    Returns the statistics for each star in each region (mean offset, rms of offset)
    """
    stats = []
    if inbounds is None:
        inbounds = np.ones(len(labels), dtype=bool)
    if inhull is None:
        inhull = np.ones(len(labels), dtype=bool)
        
    with PdfPages(filename) as pdf:
        for i, (l, p, o, u) in enumerate(zip(labels, pred, obs, unc)):
            if (not inbounds[i]) and (show_outbounds is False):
                continue
            values = dict_struct(l)
            if c3k_model is not None:
                #print(l)
                r = get_c3k_spectrum(c3k_model, outwave=wave, **values)
            else:
                r = None
            if nbox > 0:
                p, o, u, r = [boxsmooth(x, nbox) for x in [p, o, u, r]]
            fig, ax, s = zoom_lines(wave, p, o, uncertainties=u, c3k=r, **kwargs)
            stats.append(s)
            values['inhull'] = inhull[i]
            values['inbounds'] = inbounds[i]
            if 'name' not in values:
                values['name'] = ''
            ti = (tistring).format(**values)
            fig.suptitle(ti)
            pdf.savefig(fig)
            pl.close(fig)

    return stats


def zoom_lines(wave, predicted, observed, uncertainties=None,
               c3k=None, show_native=True, showlines={}, figsize=None):
    """Plot several spectra on a 
    """
    # Set up figure
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    nline = len(showlines)
    nx = np.floor(np.sqrt(nline * 1.0))
    ny = np.ceil(nline / nx)
    if figsize is None:
        figsize = (4.2*nx + 1.5, 3.8*ny + 1.0)
    fig, axes = pl.subplots(int(ny), int(nx), figsize=figsize)

    # set up to hold stats
    meanrms = np.zeros([nline, 2])
    
    for i, (line, (lo, hi)) in enumerate(showlines.items()):
        if nline > 1:
            ax = axes.flat[i]
            alpha = 1.0
        else:
            ax = axes
            alpha=0.5
        g = (wave > lo) & (wave < hi)
        if uncertainties is not None:
            ax.fill_between(wave[g], observed[g] - uncertainties[g], observed[g] + uncertainties[g],
                            color='grey', alpha=0.5)
        ax.plot(wave[g], observed[g], alpha=alpha, label='Observed')
        if show_native:
            ax.plot(wave[g], predicted[g], color='k',
                    alpha=alpha, label='SPI miles+c3k')
        delta = predicted[g] / observed[g]
        residual = np.median(delta)
        rms = (delta / residual - 1).std()
        meanrms[i,:] = np.array([residual, rms])
        ax.plot(wave[g], predicted[g] / residual, color='k', alpha=alpha, linestyle='--',
                label='SPI shifted')
        values = [100*(residual-1), 100*rms, line]
        label='SPI\nOffset: {:4.2f}%\nRMS: {:4.2f}%\n{}'.format(*values)
        ax.text(0.75, 0.20, label,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        if c3k is not None:
            delta = c3k[g] / observed[g]
            residual = np.median(delta)
            rms = (delta / residual - 1).std()
            ax.plot(wave[g], c3k[g] / residual, color='crimson',
                    linestyle='--', label='C3K shifted', alpha=alpha)
            values = [100*(residual-1), 100*rms]
            label='C3K\nOffset: {:4.2f}%\nRMS: {:4.2f}%'.format(*values)
            ax.text(0.05, 0.20, label,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
            
        
        if i == 0:
            ax.legend(loc=0, prop={'size':8})
        pl.setp(ax.xaxis.get_majorticklabels(), rotation=35,
                        horizontalalignment='center')
    return fig, axes, meanrms

        
def bias_variance(wave, bias, sigma, qlabel='\chi'):
    colors = [p['color'] for p in pl.rcParams['axes.prop_cycle']]
    sfig, sax = pl.subplots()
    sax.plot(wave, sigma, label='$\sigma_{{{}}}$'.format(qlabel),
             color=colors[0])
    sax.plot(wave, bias, label=r'$\bar{{{}}}$'.format(qlabel),
             color=colors[1])
    sax.axhline(1, linestyle=':', color=colors[0])
    sax.axhline(0, linestyle=':', color=colors[1])
    sax.set_ylim(-5, 30)
    sax.set_ylabel('${}$ (predicted - observed)'.format(qlabel))
    sax.legend(loc=0)
    return sfig, sax


def quality_map(labels, quality, quality_label, add_offsets=0.0, **extras):
    """
    """
    lab, varc = labels, quality
    l1, l2, l3 = 'logt', 'feh', 'logg'
    ranges = dict([(l, labels[l].max() - labels[l].min()) for l in [l1, l2, l3]])
    mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))

    
    rr = add_offsets * np.random.uniform(0, 1, size=len(lab))
    unit_coord = dict([(l, labels[l] / ranges[l]) for l in ["logt", "logg", "feh"]])

    rr = add_offsets * unit_coord[l3]
    sc = mapaxes[0].scatter(lab[l1] + rr*ranges[l1], lab[l2] + rr*ranges[l2], marker='o', c=varc)
    mapaxes[0].set_xlabel(l1)
    mapaxes[0].set_ylabel(l2)
    rr = add_offsets * unit_coord[l2]
    sc = mapaxes[1].scatter(lab[l1] + rr*ranges[l1], lab[l3] + rr*ranges[l3], marker='o', c=varc)
    mapaxes[1].set_xlabel(l1)
    mapaxes[1].set_ylabel(l3)
    mapaxes[1].invert_yaxis()
    cbar = pl.colorbar(sc)
    #cbar.ax.set_ylabel('Fractional RMS (%)')
    cbar.ax.set_ylabel(quality_label)
    [ax.invert_xaxis() for ax in mapaxes]
    return mapfig, mapaxes              


def plot_best_worst(psi, libindices, predicted, metric, nshow=5):
    # Plot the n worst and n best
    nshow = 5
    oo = np.argsort(metric)
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)

    bestfig, bestax = pl.subplots(nshow, 2, sharex=True)
    for i, j in enumerate(oo[:nshow]):
        k = libindices[j]
        tlab = dict_struct(psi.library_labels[k])
        tlab = 'logT={logt:4.3f}, [Fe/H]={feh:3.1f}, logg={logg:3.2f}'.format(**tlab)
        ax = bestax[i, 0]
        ax.plot(psi.wavelengths, predicted[j, :])
        ax.plot(psi.wavelengths, psi.library_spectra[k, :])
        ax.text(0.1, 0.05, tlab, transform=ax.transAxes, verticalalignment='top', bbox=props)
        bestax[i,1].plot(psi.wavelengths, (predicted[j,:] / psi.library_spectra[k, :] - 1)*100)

    worstfig, worstax = pl.subplots(nshow, 2, sharex=True)
    for i, j in enumerate(oo[-nshow:]):
        k = libindices[j]
        tlab = dict_struct(psi.library_labels[k])
        tlab = 'logT={logt:4.3f}, [Fe/H]={feh:3.1f}, logg={logg:3.2f}'.format(**tlab)
        ax = worstax[i, 0]
        ax.plot(psi.wavelengths, predicted[j, :])
        ax.plot(psi.wavelengths, psi.library_spectra[k, :])
        ax.text(0.1, 0.05, tlab, transform=ax.transAxes, verticalalignment='bottom', bbox=props)
        worstax[i,1].plot(psi.wavelengths, (predicted[j,:] / psi.library_spectra[k, :] - 1) * 100)

    return bestfig, worstfig


def flux_teff(psi, nt=500, nw=5, showgrid=False):
    fig, ax = pl.subplots()
    #clrs = [p['color'] for p in pl.rcParams['axes.prop_cycle']]
    logg = np.zeros(nt) + np.median(psi.training_labels['logg'])
    feh = np.zeros(nt) + np.median(psi.training_labels['feh'])
    logt = np.linspace(psi.training_labels['logt'].min(), psi.training_labels['logt'].max(), nt)
    spec, covered = psi.get_star_spectrum(logt=logt, logg=logg, feh=feh, check_coverage=True)
    
    inds = (np.linspace(0, len(psi.wavelengths), nw+2)[1:-1]).astype(int)
    for j, i in enumerate(inds):
        ax.plot(logt, spec[covered, i], 'o', label='$\lambda={}$'.format(psi.wavelengths[i]))
        if covered.sum() < len(covered):
            ax.plot(logt, spec[~covered, i], 'o', mfc=None)

    if showgrid:
        feh_grid = np.unique(psi.training_labels["feh"])
        logg_grid = np.unique(psi.training_labels["logg"])
        fg = feh_grid[np.argmin(np.abs(feh[0] - feh_grid))]
        gg = logg_grid[np.argmin(np.abs(logg[0] - logg_grid))]
        good = (psi.training_labels["feh"] == fg) & (psi.training_labels["logg"] == gg)
        for j, i in enumerate(inds):
            ax.plot(psi.training_labels[good]["logt"], psi.training_spectra[good, i], 's')

    return fig, ax


def show_fit_slice(psi, param, n=500, nw=5, waves=None, showgrid=True, **kwargs):
    """
    :param psi: A trained interpolator

    :param param:
        The name of the parameter forming the slice.  
        One of "logt" | "logg" | "feh"

    :param n: number of points along the slice

    :param nw: number of wavelength points to show

    :param showgrid:
        If True, show the grid on top of the predictions. 
        Only works if the training set is actually a grid.

    :param kwargs: (optional)
        Can specify desired values for the non-'param' parameters, as
        e.g. logg=4.5.  If not given, the median of the training set will be
        used.
    """

    parnames = ["logt", "logg", "feh"]

    pars = {}
    for p in parnames:
        if p == param:
            value = np.linspace(psi.training_labels[p].min(), psi.training_labels[p].max(), n)
        elif p in kwargs:
            print(kwargs[p])
            value = np.zeros(n) + kwargs[p]
        else:
            value = np.zeros(n) + np.median(psi.training_labels[p])
        pars[p] = value

    # get predicted spectra along the parameter slice
    spec, covered = psi.get_star_spectrum(check_coverage=True, **pars)

    # plot flux_i versus param
    fig, ax = pl.subplots()
    cmap = pl.cm.get_cmap("viridis")

    if waves is None:
        inds = (np.linspace(0, len(psi.wavelengths), nw+2)[1:-1]).astype(int)
    else:
        inds = np.array([np.argmin(np.abs(w - psi.wavelengths)) for w in waves])

    cinds = (inds - inds.min()) * 1.0 / (inds.max() - inds.min())
    for j, i in enumerate(inds):
        ax.plot(pars[param], spec[covered, i], 'o',
                color=cmap(cinds[j]), label='$\lambda={}$'.format(psi.wavelengths[i]))
        if covered.sum() < len(covered):
            ax.plot(pars[param], spec[~covered, i], 'o', mfc=None, mec=cmap(cinds[j]))

    t = ["{}={:4.3f}".format(p, pars[p][0]) for p in parnames if p != param]
    t = ", ".join(t)
    ax.set_title(t)
    ax.set_xlabel(param)

    if showgrid:
        grid = {}
        sel = np.ones(len(psi.training_labels), dtype=bool)
        for p in parnames:
            if p != param:
                gridpoints = np.unique(psi.training_labels[p])
                grid_closest = gridpoints[np.argmin(np.abs(pars[p][0] - gridpoints))]
                grid[p] = grid_closest
                sel = sel & (psi.training_labels[p] == grid_closest)

        tg = ["{}={:4.3f}".format(p, grid[p]) for p in grid.keys()]
        tg = ", ".join(tg)
        for j, i in enumerate(inds):
            ax.plot(psi.training_labels[sel][param], psi.training_spectra[sel, i], 's',
                    markerfacecolor=cmap(cinds[j]), markeredgecolor="k", markeredgewidth=1)
        ax.set_title(t + "; " + tg)

    return fig, ax, psi.wavelengths[inds]


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


