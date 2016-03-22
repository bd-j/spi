import sys
import numpy as np
import matplotlib.pyplot as pl
from model import MILESInterpolator
from badstar import allbadstars

def select(psi, mlib, bad_ids, bounds):
    """Select a training set using bounds and removing bad stars listed by miles_id
    """
    psi.load_training_data(training_data=mlib)
    ind = [psi.training_labels['miles_id'].tolist().index(b) for b in bad_ids
           if b in psi.training_labels['miles_id']]
    psi.leave_out(ind)
    psi.restrict_sample(bounds=bounds)
    return psi

# The PSI Model
mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (3000.0, 10000.0)}
psi = MILESInterpolator(training_data=mlib, normalize_labels=False)
badstar_ids = np.array(allbadstars.tolist() + [929, 815])
psi.features = (['logt'], ['feh'], ['logg'],
                ['logt', 'logt'], ['feh', 'feh'], ['logg', 'logg'],
                ['logt', 'feh'], ['logg', 'logt'], ['logg', 'feh'],
                ['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'],
                ['logt', 'logt', 'logg']
                )
psi = select(psi, mlib, badstar_ids, fgk_bounds)

ntrain = psi.n_train
predicted = np.zeros([ntrain, psi.n_wave])

#sys.exit()
# Retrain and predict after leaving one out
for i in range(ntrain):
    if (i % 10) == 0: print(i)
    psi = select(psi, mlib, badstar_ids, fgk_bounds)
    psi.features = (['logt'], ['feh'], ['logg'],
                    ['logt', 'logt'], ['feh', 'feh'], ['logg', 'logg'],
                    ['logt', 'feh'], ['logg', 'logt'], ['logg', 'feh'],
                    ['logt', 'logt', 'logt'],
                    ['logt', 'logt', 'logt', 'logt'],
                    ['logt', 'logt', 'logg']
                    )

    spec = psi.training_spectra[i,:]
    tlabels = psi.training_labels[i]
    labels = dict([(n, tlabels[n]) for n in psi.label_names])
    psi.leave_out(i)
    psi.train()
    #print(psi.coeffs)
    predicted[i, :] = psi.get_star_spectrum(**labels)

# reload the full training set
psi = select(psi, mlib, badstar_ids, fgk_bounds)

# get fractional residuals
imin = np.argmin(np.abs(psi.wavelengths - 3800))
imax = np.argmin(np.abs(psi.wavelengths - 7400))
delta = predicted/psi.training_spectra - 1.0
var_spectrum = delta.var(axis=0)
var_total = delta[:, imin:imax].var(axis=1)
lines, indlines = {'Ha':6563., 'Na D': 5897.0, 'Ca K': 3933.0, 'Ca H': 3968}, {}
for l, w in lines.items():
    indlines[l] = np.argmin(np.abs(psi.wavelengths - w))


# Plot the variance spectrum
sfig, sax = pl.subplots()
sax.plot(psi.wavelengths, np.sqrt(var_spectrum)*100, label='$\sigma(m/o-1)$')
sax.set_xlabel('$\lambda (\AA)$')
sax.set_ylabel('Fractional RMS (%)')
sax.set_ylim(0, 100)
sfig.show()
sfig.savefig('figures/residual_spectrum.pdf')

# Plot a map of total variance as a function of label
l1, l2, l3 = 'logt', 'feh', 'logg'
lab = psi.training_labels 
mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))
cm = pl.cm.get_cmap('gnuplot2_r')
sc = mapaxes[0].scatter(lab[l1], lab[l2], marker='o', c=np.sqrt(var_total)*100)
mapaxes[0].set_xlabel(l1)
mapaxes[0].set_ylabel(l2)
sc = mapaxes[1].scatter(lab[l1], lab[l3], marker='o', c=np.sqrt(var_total)*100)
mapaxes[1].set_xlabel(l1)
mapaxes[1].set_ylabel(l3)
cbar = pl.colorbar(sc)
cbar.ax.set_ylabel('Fractional RMS (%)')
mapfig.show()                   
mapfig.savefig('figures/residual_map.pdf')

# Plot a map of line residual as a function of label
showlines = lines.keys()
for line in showlines:
    mapfig, mapaxes = pl.subplots(1, 2, figsize=(12.5,7))
    cm = pl.cm.get_cmap('gnuplot2_r')
    sc = mapaxes[0].scatter(lab[l1], lab[l2], marker='o', c=delta[:, indlines[line]]*100)
    mapaxes[0].set_xlabel(l1)
    mapaxes[0].set_ylabel(l2)
    sc = mapaxes[1].scatter(lab[l1], lab[l3], marker='o', c=delta[:,indlines[line]]*100)
    mapaxes[1].set_xlabel(l1)
    mapaxes[1].set_ylabel(l3)
    cbar = pl.colorbar(sc)
    cbar.ax.set_ylabel('Residual @ {}'.format(line))
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

badfig, badaxes = pl.subplots(10, 1, sharex=True, figsize=(5, 12))
for i, bad in enumerate(oo[-10:][::-1]):
    ax = badaxes[i]
    labels = dict([(n, psi.trining_labels[n]) for n in psi.label_names])
    title = "T={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.1f}".format(**labels)
    ax.plot(psi.wavelengths, predicted[bad,:], label='predicted')
    ax.plot(psi.wavelengths, psi.training_spectra[bad,:], label='actual')
    ax.text(0.05, 0.95, "#{}, RMS={}%".format(psi.training_labels[bad]['miles_id'], rms[bad]),
            transform=ax.transAxes, fontsize=10)
    ax.text(0.5, 0.05, title, transform=ax.transAxes, fontsize=10)
    
    if i == 0:
        ax.legend(loc=0, props={'size':8})
    if i < 9:
        ax.set_xticklabels([])
badfig.savefig('worst10.pdf')
        
sys.exit()
# compute covariance matrix
dd = delta[:, imin:imax]
dd[~np.isfinite(dd)] = 0.0
cvmat = np.cov(dd.T)

l = 'Ha'
plot(psi.wavelengths[imin:imax], cvmat[ind[l]-imin,:], label=lines.keys()[i])
     
