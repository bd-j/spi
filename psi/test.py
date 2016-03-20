import numpy as np
import matplotlib.pyplot as pl
from model import MILESInterpolator

mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (4000.0, 9000.0)}

fgk = MILESInterpolator(training_data=mlib, normalize_labels=True)
fgk.restrict_sample(bounds=fgk_bounds)
fgk.train()

predicted = np.zeros([fgk.n_train, fgk.n_wave])
for i in range(fgk.n_train):
    train = fgk.training_labels[i]
    labels = dict([(n, train[n]) for n in fgk.label_names])
    predicted[i, :] = fgk.get_star_spectrum(**labels)
    

delta = predicted - fgk.training_spectra

for ind in [0, 300]:
    fig, axes = pl.subplots()
    axes.plot(fgk.wavelengths, fgk.training_spectra[ind,:])
    axes.plot(fgk.wavelengths, predicted[ind,:])
    fig.show()
