import sys
import numpy as np
import matplotlib.pyplot as pl
from psi.library_models import CKCInterpolator
from psi.utils import dict_struct

lightspeed = 2.998e18

# The PSI Model
mlib = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/ckc_R10k.h5'
fgk_bounds = {'logt': (np.log10(4000.0), np.log10(9000.0)),
              'logg': (2.0, 5.0),
              'feh': (-2, 0.5)}

spi = CKCInterpolator(training_data=mlib, logify_flux=True)
spi.features = (['logt'], ['feh'], ['logg'],
                # Square terms
                ['logt', 'logt'], ['feh', 'feh'], ['logg', 'logg'],
                # Cubic terms
                ['feh', 'feh', 'feh'], ['logt', 'logt', 'logt'], ['logg', 'logg', 'logg'],
                # Cross terms
                ['logt', 'feh'], ['logg', 'logt'], ['logg', 'feh'],
                # logt high order
                ['logt', 'logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt', 'logt'],
                #['logt', 'logt', 'logt', 'logt', 'logt', 'logt'],
                # feh high order
                ['feh', 'feh', 'feh', 'feh'], #['feh', 'feh', 'feh', 'feh', 'feh'],
                # logg-logt high order cross terms
                ['logt', 'logt', 'logg'], ['logt', 'logg', 'logg'],
                # feh-logt high order cross terms
                ['logt', 'logt', 'feh'], ['logt', 'feh', 'feh'],
                # feh-logg high order cross terms
                ['logg', 'logg', 'feh'], ['logg', 'feh', 'feh'],
                # feh-logt higher order cross terms
                ['logt', 'logt', 'feh', 'feh'], ['logt', 'logt', 'logt', 'feh'], ['logt', 'feh', 'feh', 'feh'],
                ['logt', 'logt', 'logt', 'logt', 'feh'],
                #['logt', 'logt', 'logt', 'logt', 'logt', 'feh'],
                #['logt', 'logt', 'logt', 'logt', 'feh', 'feh']
                )

spi.select(bounds=fgk_bounds, delete=True)
tot = np.trapz(spi.training_spectra * lightspeed/spi.wavelengths**2, spi.wavelengths, axis=-1)
#bad = tot <= 0
#spi.leave_out(np.where(bad)[0])
ntrain = spi.n_train

spi.train()
params = spi.training_labels
ind_solar = np.searchsorted(np.unique(params['logt']), np.log10(5870.0))
logt_solar = np.unique(params['logt'])[ind_solar]
solar = (params['logt'] == logt_solar) & (params['feh'] == 0) & (params['logg'] == 4.5)
specsun = spi.get_star_spectrum(logt=logt_solar, feh=0, logg=4.5)

import matplotlib.pyplot as pl
fig, axes = pl.subplots(2, 1)
axes[0].plot(spi.wavelengths, spi.training_spectra[solar,:][0])
axes[0].plot(spi.wavelengths, specsun)
axes[1].plot(spi.wavelengths, specsun / spi.training_spectra[solar,:][0])
pl.show()


predicted = np.zeros([ntrain, spi.n_wave])
# Retrain and predict after leaving one out
loo_indices = spi.training_indices.copy()[::10]
for i, j in enumerate(loo_indices):
    if (i % 10) == 0: print('{} of {}'.format(i, len(loo_indices)))
    # get full sample and the parameters of the star to leave out
    spec = spi.training_spectra[i,:]
    tlabels = spi.training_labels[i]
    labels = dict([(n, tlabels[n]) for n in spi.label_names])
    # leave one out and train
    spi.library_mask[j] = False
    spi.train()
    predicted[j, :] = spi.get_star_spectrum(**labels)
    spi.library_mask[j] = True

# reload the full training set
# spi.select(training_data=mlib, bounds=fgk_bounds)

# get fractional residuals
wmin, wmax = 3800, 7200
imin = np.argmin(np.abs(spi.wavelengths - wmin))
imax = np.argmin(np.abs(spi.wavelengths - wmax))
imin, imax = 0, len(spi.wavelengths) -1
delta = predicted / spi.training_spectra - 1.0


var_spectrum = delta[loo_indices,:].var(axis=0)
var_total = delta[:, imin:imax].var(axis=1)
snr = 100
chisq = ((snr * delta)**2).sum(axis=1)
#lines, indlines = {'Ha':6563., 'NaD': 5897.0, 'CaK': 3933.0, 'CaH': 3968, 'Mg5163':5163.1}, {}
#for l, w in lines.items():
#    indlines[l] = np.argmin(np.abs(spi.wavelengths - w))


# Plot the variance spectrum
sfig, sax = pl.subplots()
sax.plot(spi.wavelengths, np.sqrt(var_spectrum)*100, label='$\sigma(m/o-1)$')
sax.set_xlabel('$\lambda (\AA)$')
sax.set_ylabel('Fractional RMS (%)')
sax.set_ylim(0, 100)
sfig.show()
#sfig.savefig('figures/residual_spectrum.pdf')

# Plot a map of total variance as a function of label
l1, l2, l3 = 'logt', 'feh', 'logg'
lab = spi.training_labels[loo_indices]
mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))
varc = np.clip(np.sqrt(var_total[loo_indices])*100, 0, 1e3)
varc = np.log(chisq[loo_indices])
offset = np.random.uniform(0, 1, size=len(loo_indices))
sc = mapaxes[0].scatter(lab[l1] + 0.01 * offset, lab[l2] + 0.1*offset, marker='o', c=varc)
mapaxes[0].set_xlabel(l1)
mapaxes[0].set_ylabel(l2)
sc = mapaxes[1].scatter(lab[l1] + 0.01 * offset, lab[l3]+0.1*offset, marker='o', c=varc)
mapaxes[1].set_xlabel(l1)
mapaxes[1].set_ylabel(l3)
mapaxes[1].invert_yaxis()
cbar = pl.colorbar(sc)
#cbar.ax.set_ylabel('Fractional RMS (%)')
cbar.ax.set_ylabel(r'$log \, \chi^2$ (S/N={})'.format(snr))
[ax.invert_xaxis() for ax in mapaxes]
mapfig.show()                   
#mapfig.savefig('figures/residual_map.pdf')


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

    
# Plot a map of line residual as a function of label
showlines = lines.keys()
showlines = ['CaK', 'NaD']
for line in showlines:
    vlim = -50, 50
    if lines[line] < 4000:
        vlim = -100, 100
    mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))
    sc = mapaxes[0].scatter(lab[l1], lab[l2], marker='o', c=delta[:, indlines[line]]*100,
                            cmap=pl.cm.coolwarm, vmin=vlim[0], vmax=vlim[1])
    mapaxes[0].set_xlabel(l1)
    mapaxes[0].set_ylabel(l2)
    sc = mapaxes[1].scatter(lab[l1], lab[l3], marker='o', c=delta[:,indlines[line]]*100,
                            cmap=pl.cm.coolwarm, vmin=vlim[0], vmax=vlim[1])
    mapaxes[1].set_xlabel(l1)
    mapaxes[1].set_ylabel(l3)
    mapaxes[1].invert_yaxis()
    cbar = pl.colorbar(sc)
    cbar.ax.set_ylabel('Residual @ {}'.format(line))
    [ax.invert_xaxis() for ax in mapaxes]
    mapfig.show()
    mapfig.savefig('figures/residual_{}_map.pdf'.format(line))

# Plot cumulative number as a function of RMS
rms = np.sqrt(var_total)*100
rms[~np.isfinite(rms)]=1000
oo = np.argsort(rms)
cfig, cax = pl.subplots()
cax.plot(rms[oo], np.arange(len(oo)))
cax.set_ylabel('N(<RMS)')
cax.set_xlabel('Fractional RMS (%)')
cax.set_xlim(0,100)
cfig.show()
cfig.savefig('figures/cumlative_rms.pdf')

badfig, badaxes = pl.subplots(10, 1, sharex=True, figsize=(6, 12))
for i, bad in enumerate(oo[-10:][::-1]):
    ax = badaxes[i]
    labels = dict([(n, spi.training_labels[bad][n]) for n in spi.training_labels.dtype.names])
    title = "T={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.1f}".format(**labels)
    ax.plot(spi.wavelengths, predicted[bad,:], label='predicted')
    ax.plot(spi.wavelengths, spi.training_spectra[bad,:], label='actual')
    ax.text(0.05, 0.9, "#{}, RMS={:4.1f}%".format(spi.training_labels[bad]['miles_id'], rms[bad]),
            transform=ax.transAxes, fontsize=10)
    ax.text(0.7, 0.05, title, transform=ax.transAxes, fontsize=10)
    if i == 0:
        ax.legend(loc=0, prop={'size':8})
    if i < 9:
        ax.set_xticklabels([])
badfig.savefig('figures/worst10.pdf')



sys.exit()
# compute covariance matrix
dd = delta[:, imin:imax]
dd[~np.isfinite(dd)] = 0.0
cvmat = np.cov(dd.T)

l = 'Ha'
plot(spi.wavelengths[imin:imax], cvmat[lineinds[l]-imin,:], label=lines.keys()[i])
     
