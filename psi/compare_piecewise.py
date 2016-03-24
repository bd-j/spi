import numpy as np
import matplotlib.pyplot as pl
from numpy.lib import recfunctions as rfn

from model import MILESInterpolator
from comparison_model import PiecewiseMILES

def psi_select(psi, mlib, bad_ids, bounds):
    """Select a training set using bounds and removing bad stars listed by miles_id
    """
    psi.select(training_data=mlib, bounds=bounds, badvalues={'miles_id': bad_ids})
    return psi

def plin_select(plin, mlib, bad_ids, bounds):
    """Select a training set using bounds and removing bad stars listed by miles_id
    """
    plin.load_lib(libname=mlib)
    plin.select(bounds=bounds, badvalues={'miles_id': bad_ids})
    return plin

# Library and selections
mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (3000.0, 10000.0)}
from badstar import allbadstars
badstar_ids = np.array(allbadstars.tolist())

# The PSI Model
psi = MILESInterpolator(training_data=mlib, normalize_labels=False)
psi.features = (['logt'], ['feh'], ['logg'],
                ['logt', 'logt'], ['feh', 'feh'], ['logg', 'logg'],
                ['logt', 'feh'], ['logg', 'logt'], ['logg', 'feh'],
                ['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'],
                ['logt', 'logt', 'logg']
                )

# The Piecewise linear Model
plin = PiecewiseMILES(libname=mlib, log_interp=True,
                      use_params=['logt', 'logg', 'feh'])

subpsi = psi_select(psi, mlib, badstar_ids, fgk_bounds)
ntrain = subpsi.n_train
predicted_psi = np.zeros([ntrain, psi.n_wave])
predicted_plin = np.zeros([ntrain, len(plin.wavelengths)])

on_edge = np.zeros(ntrain, dtype=bool)

for i in range(ntrain):
    if (i % 10) == 0: print(i)
    # get full sample and the parameters of the star to leave out
    psi = psi_select(psi, mlib, badstar_ids, fgk_bounds)
    plin = plin_select(plin, mlib, badstar_ids, fgk_bounds)
    spec = psi.training_spectra[i,:]
    tlabels = psi.training_labels[i]
    labels = dict([(n, tlabels[n]) for n in psi.label_names])
    # leave one out and train
    psi.leave_out(i)
    psi.train()
    plin.leave_out(i)
    plin.reset()
    
    predicted_psi[i, :] = psi.get_star_spectrum(**labels)
    try:
        predicted_plin[i, :] = plin.get_star_spectrum(**labels)[1]
    except(ValueError):
        on_edge[i] = True

    psi = psi_select(psi, mlib, badstar_ids, fgk_bounds)
    delta_psi = predicted_psi/psi.training_spectra - 1.0
    delta_plin = predicted_plin/psi.training_spectra - 1.0
    
    var_psi = np.nanvar(delta_psi[~on_edge, :], axis=0)
    var_plin = np.nanvar(delta_plin[~on_edge, :], axis=0)

    sfig, sax = pl.subplots()
    sax.plot(psi.wavelengths, np.sqrt(var_psi)*100, label='PSI')
    sax.plot(psi.wavelengths, np.sqrt(var_plin)*100, label='Piecewise linear')
    sax.set_ylim(0,100)
    sax.set_xlabel('$\lambda (\AA)$')
    sax.set_ylabel('Fractional RMS (%)')
    sax.legend(loc=0)
    
    sfig.savefig('figures/piecewise_compare.pdf')
    
