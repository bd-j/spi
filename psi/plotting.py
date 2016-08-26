import numpy as np
import matplotlib.pyplot as pl

showlines = {'CO': (2.26, 2.35),
             'CaT': (0.845, 0.870),
             'Feh': (0.980, 1.0),
             'NaD': (0.580, 0.596)
             }
props = dict(boxstyle='round', facecolor='w', alpha=0.5)


def zoom_lines(wave, predicted, observed, uncertainties=None,
               showlines={}):
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    nline = len(showlines)
    nx = np.floor(np.sqrt(nline * 1.0))
    
    fig, axes = pl.subplots(2,2, figsize=(10.0, 8.5))
    for i, (line, (lo, hi)) in enumerate(showlines.items()):
        ax = axes.flat[i]
        g = (wave > lo) & (wave < hi)
        if uncertainties is not None:
            ax.fill_between(wave[g], observed[g] - uncertainties[g], observed[g] + uncertainties[g],
                            color='grey', alpha=0.5)
        ax.plot(wave[g], observed[g], label='Observed')
        ax.plot(wave[g], predicted[g], color='k', label='PSI miles+c3k')
        delta = predicted[g]/observed[g]
        residual = np.median(delta)
        rms = (delta / residual - 1).std()
        ax.plot(wave[g], predicted[g] / residual, color='k',
                linestyle='--', label='PSI shifted')
        values = [100*(residual-1), 100*rms, line]
        label='Offset: {:4.2f}%\nRMS: {:4.2f}%\n{}'.format(*values)
        ax.text(0.75, 0.20, label,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        if i == 0:
            ax.legend(loc=0, prop={'size':8})
        pl.setp(ax.xaxis.get_majorticklabels(), rotation=35,
                        horizontalalignment='center')
    return fig, axes


def bias_variance():
    sfig, sax = pl.subplots()
    sax.plot(spi.wavelengths, np.sqrt(var_spectrum)*100, label='Dispersion')
    sax.plot(spi.wavelengths, np.abs(bias_spectrum)*100, label='Mean absolute offset')
    sax.set_ylim(0.001, 100)
    sax.set_yscale('log')
    sax.set_ylabel('%')
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
