import numpy as np
import matplotlib.pyplot as pl


props = dict(boxstyle='round', facecolor='w', alpha=0.5)

def boxsmooth(x, width=10):
    return np.convolve(x, np.ones(width*1.0) / width, mode='same')


def zoom_lines(wave, predicted, observed, uncertainties=None,
               c3k=None, show_native=True, showlines={}, figsize=None):
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    nline = len(showlines)
    nx = np.floor(np.sqrt(nline * 1.0))
    ny = np.ceil(nline / nx)
    if figsize is None:
        figsize = (4.2*nx + 1.5, 3.8*ny + 1.0)
    fig, axes = pl.subplots(int(ny), int(nx), figsize=figsize)
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
                    alpha=alpha, label='PSI miles+c3k')
        delta = predicted[g] / observed[g]
        residual = np.median(delta)
        rms = (delta / residual - 1).std()
        ax.plot(wave[g], predicted[g] / residual, color='k', alpha=alpha, linestyle='--',
                label='PSI shifted')
        values = [100*(residual-1), 100*rms, line]
        label='PSI\nOffset: {:4.2f}%\nRMS: {:4.2f}%\n{}'.format(*values)
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
    return fig, axes

        
def bias_variance(wave, bias, sigma, qlabel='\chi'):
    colors = pl.rcParams['axes.color_cycle']
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


def quality_map(labels, quality, quality_label):
    """
    """
    lab, varc = labels, quality
    l1, l2, l3 = 'logt', 'feh', 'logg'
    mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))

    sc = mapaxes[0].scatter(lab[l1], lab[l2], marker='o', c=varc)
    mapaxes[0].set_xlabel(l1)
    mapaxes[0].set_ylabel(l2)
    sc = mapaxes[1].scatter(lab[l1], lab[l3], marker='o', c=varc)
    mapaxes[1].set_xlabel(l1)
    mapaxes[1].set_ylabel(l3)
    mapaxes[1].invert_yaxis()
    cbar = pl.colorbar(sc)
    #cbar.ax.set_ylabel('Fractional RMS (%)')
    cbar.ax.set_ylabel(quality_label)
    [ax.invert_xaxis() for ax in mapaxes]
    return mapfig, mapaxes              
