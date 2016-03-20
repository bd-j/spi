import sys
import numpy as np
import matplotlib.pyplot as pl
from model import MILESInterpolator, TGM

mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (4000.0, 9000.0)}
interpolator = '/Users/bjohnson/Projects/psi/data/miles/ulys/miles_tgm2.fits'
trange = 'warm'

fgk = MILESInterpolator(training_data=mlib, normalize_labels=False)
fgk.restrict_sample(bounds=fgk_bounds)
fgk.train()

tgm = TGM(interpolator=interpolator, trange=trange)

predicted_tgm = np.zeros([fgk.n_train, tgm.n_wave])
predicted_psi = np.zeros([fgk.n_train, fgk.n_wave])

for i in range(fgk.n_train):
    train = fgk.training_labels[i]
    labels = dict([(n, train[n]) for n in fgk.label_names])
    predicted_tgm[i, :] = tgm.get_star_spectrum(**labels)
    predicted_psi[i, :] = fgk.get_star_spectrum(**labels)
    

ratio = predicted_tgm[:,5:-1] / fgk.training_spectra

for ind in [10, 100]:
    train = fgk.training_labels[ind]
    labels = dict([(n, train[n]) for n in fgk.label_names])
    fig, axes = pl.subplots()
    axes.plot(fgk.wavelengths, fgk.training_spectra[ind,:], label='MILES spectrum')
    axes.plot(tgm.wavelengths, predicted_tgm[ind,:], label='TGM v{}'.format(tgm.version))
    axes.plot(fgk.wavelengths, predicted_psi[ind,:], label='PSI')
    axes.set_title("T={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.1f}".format(**labels))
    axes.legend(loc=0)
    fig.show()


fig, axes = pl.subplots()
axes.plot(fgk.training_labels['logt'], np.squeeze(fgk.training_spectra[:,1000]), 'o')
axes.plot(fgk.training_labels['logt'], np.squeeze(predicted_psi[:,1000]), 'o')
fig.show()
