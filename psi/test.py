import sys
import numpy as np
import matplotlib.pyplot as pl
from model import MILESInterpolator, TGM


# The PSI Model
mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (4000.0, 9000.0)}
psi = MILESInterpolator(training_data=mlib, normalize_labels=False)
psi.restrict_sample(bounds=fgk_bounds)
psi.train()

# The ULYSS/TGM model from Pugniel11/Sharma15
interpolator = '/Users/bjohnson/Projects/psi/data/miles/ulys/miles_tgm2.fits'
trange = 'warm'
tgm = TGM(interpolator=interpolator, trange=trange)

# Set up output
predicted_tgm = np.zeros([psi.n_train, tgm.n_wave])
predicted_psi = np.zeros([psi.n_train, psi.n_wave])

# Make predictions for every star in the training set
for i in range(psi.n_train):
    train = psi.training_labels[i]
    labels = dict([(n, train[n]) for n in psi.label_names])
    predicted_tgm[i, :] = tgm.get_star_spectrum(**labels)
    predicted_psi[i, :] = psi.get_star_spectrum(**labels)

# The tgm wavelength scale extends slightly blueward and redward of the
# released MILES data
ratio = predicted_tgm[:,5:-1] / fgk.training_spectra

# Plot for a couple stars
for ind in [10, 100]:
    train = fgk.training_labels[ind]
    labels = dict([(n, train[n]) for n in fgk.label_names])
    fig, axes = pl.subplots()
    axes.plot(psi.wavelengths, fgk.training_spectra[ind,:], label='MILES spectrum')
    axes.plot(tgm.wavelengths, predicted_tgm[ind,:], label='TGM v{}'.format(tgm.version))
    axes.plot(psi.wavelengths, predicted_psi[ind,:], label='GPSI')
    axes.set_title("T={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.1f}".format(**labels))
    axes.legend(loc=0)
    fig.show()

# plot for one wavelength
fig, axes = pl.subplots()
axes.plot(psi.training_labels['logt'], np.squeeze(psi.training_spectra[:,1000]), 'o')
axes.plot(psi.training_labels['logt'], np.squeeze(predicted_psi[:,1000]), 'o')
fig.show()
