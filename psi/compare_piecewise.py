import numpy as np
import matplotlib.pyplot as pl
from numpy.lib import recfunctions as rfn

from model import MILESInterpolator, within
from bsfh.source_basis import StarBasis

class PiecewiseMILES(StarBasis):


    def load_lib(self, libname='', **extras):
        """Read a CKC library which has been pre-convolved to be close to your
        resolution.  This library should be stored as an HDF5 file, with the
        datasets ``wavelengths``, ``parameters`` and ``spectra``.  These are
        ndarrays of shape (nwave,), (nmodels,), and (nmodels, nwave)
        respecitvely.  The ``parameters`` array is a structured array.  Spectra
        with no fluxes > 1e-32 are removed from the library
        """
        import h5py
        with h5py.File(libname, "r") as f:
            self._wave = np.array(f['wavelengths'])
            self._libparams = np.array(f['parameters'])
            self._spectra = np.array(f['spectra'])
            ancillary = f['ancillary'][:]
        # add and rename labels here.  Note that not all labels need or will be
        # used in the feature generation
        newfield = ['logt', 'miles_id']
        newdata = [np.log10(self._libparams['teff']), ancillary['miles_id']]
        self._libparams = rfn.append_fields(self._libparams, newfield, newdata,
                                            usemask=False)
        
    def select(self, bounds=None, badvalues=None):

        if bounds is not None:
            good = np.ones(len(self._libparams), dtype=bool)
            for name, bound in bounds.items():
                good = good & within(bound, self._libparams[name])
            self._spectra = self._spectra[good, :]
            self._libparams = self._libparams[good, ...]

        if badvalues is not None:
            inds = []
            for name, bad in badvalues.items():
                inds += [self._libparams[name].tolist().index(b) for b in bad
                        if b in self._libparams[name]]
            self.leave_out(np.array(inds).flat)

        self.triangulate()

    def leave_out(self, inds):
        self._spectra = np.delete(self._spectra, inds, axis=0)
        self._libparams = np.delete(self._libparams, inds)
        self.reset()

    def reset(self):
        self.triangulate()
        try:
            self.build_kdtree()
        except NameError:
            pass

def psi_select(psi, mlib, bad_ids, bounds):
    """Select a training set using bounds and removing bad stars listed by miles_id
    """
    psi.load_training_data(training_data=mlib)
    ind = [psi.training_labels['miles_id'].tolist().index(b) for b in bad_ids
           if b in psi.training_labels['miles_id']]
    psi.leave_out(ind)
    psi.restrict_sample(bounds=bounds)
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
    
