import sys
import numpy as np
import matplotlib.pyplot as pl
from model import MILESInterpolator

# The PSI Model
mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (4000.0, 9000.0)}
psi = MILESInterpolator(training_data=mlib, normalize_labels=False)
psi.restrict_sample(bounds=fgk_bounds)

ntrain = psi.n_train
predicted = np.zeros([ntrain, psi.n_wave])

for i in range(ntrain):
    psi.load_training_data(training_data=mlib)
    psi.restrict_sample(bounds=fgk_bounds)
    spec= psi.training_spectrum[i,:]
    tlabels = psi.training_label[i,:]
    labels = dict([(n, tlabels[n]) for n in psi.label_names])
    psi.leave_out(i)
    psi.train()
    predicted[i, :] = psi.get_spectrum(**labels)
