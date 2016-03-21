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
ratio = predicted_tgm[:,5:-1] / psi.training_spectra



# Plot for a couple stars
props = dict(boxstyle='round', facecolor='w', alpha=0.5)
for ind in [10, 100]:
    train = psi.training_labels[ind]
    labels = dict([(n, train[n]) for n in psi.label_names])
    title = "T={teff:4.0f}, logg={logg:3.2f}, feh={feh:3.1f}".format(**labels)
    fig, axes = pl.subplots()
    axes.plot(psi.wavelengths, psi.training_spectra[ind,:], label='MILES spectrum')
    axes.plot(tgm.wavelengths, predicted_tgm[ind,:], label='TGM v{}'.format(tgm.version))
    axes.plot(psi.wavelengths, predicted_psi[ind,:], label='GPSI')
    axes.text(0.05, 0.95, title, transform=axes.transAxes, fontsize=14,
              verticalalignment='top')
    axes.set_title('MILES #{}'.format(ind+1))
    axes.legend(loc=0)
    fig.show()
    fig.savefig('figures/prediction_miles{:04.0f}'.format(ind+1))

# plot for one wavelength
fig, axes = pl.subplots()
axes.plot(psi.training_labels['logt'], np.squeeze(psi.training_spectra[:,1000]), 'o')
axes.plot(psi.training_labels['logt'], np.squeeze(predicted_psi[:,1000]), 'o')
fig.show()
