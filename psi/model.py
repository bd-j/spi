import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.linalg import inv
import h5py


class PSIModel(object):

    def __init__(self, **kwargs):
        self.normalize_training_data = False
        self.load_training_data(**kwargs)
        self.restrict_sample(**kwargs)
        self.configure_features(**kwargs)
        self.reset()
        
    def load_training_data(self, **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to an ndarray of labels of shape
        (ntrain, nlabel) and an ndarray of training spectra of shape (nwave,
        ntrain).  Optionally subtract off the median label and spectrum.
        """
        # Need to write this
        self.training_spectra = np.zeros([nobj, nwave])
        self.training_labels = np.zeros([nobj, nlabel])
        self.label_names = []
        dlabel = (self.training_labels.T - self.reference_label)
        
    def labels_from_dict(self, **label_dict):
        return np.atleast_2d([label_dict[n] for n in self.label_names])
        
    def get_star_spectrum(**kwargs):
        """Get an interpolated spectrum at the parameter values (labels)
        specified as keywords.  These *must* include all elements of
        ``label_names``.
        """
        assert True in self.trained
        labels = self.labels_from_dict(**kwargs)
        feature = self.labels_to_features(labels)
        spectrum = np.dot(self.coeffs, feature.T)
        return np.squeeze(spectrum)

    def configure_features(self, **extras):
        """Here you set up which terms to use.  This is set up to include all
        linear and quadratic (cross) terms.  """
        inds = combinations_with_replacement(range(self.n_labels), r=2)
        self.inds = np.array(list(inds))
        self.features = (self.label_names +
                         list(combinations_with_replacement(self.label_names, r=2)))

    def labels_to_features(labels):
        """Construct a feature vector from a label vector.  This is a simple
        quadratic model, and a placeholder.  It should be reimplemented by
        subclasses.

        :param labels:
            Label vector(s).  ndarray of shape (nobj, nlabels)

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """        
        quad = np.einsum('...i,...j->...ij', labels, labels)[:, self.inds[:, 0], self.inds[:,1]]
        return np.hstack([labels, quad])

    def construct_design_matrix(self, bounds=None, order=2):
        """Construct and store the [Nobj x Nfeatures] design matrix and the
        precision matrix.
        """        
        self.X = self.labels_to_features(self.training_labels)
        self.Ainv = inv(np.dot(self.X.T, self.X))

    def train(self, inds=None, pool=None):
        """Do the regression for the indicated wavelengths.  This can take a
        pool object with a ``map`` method to enable parallelization.
        """
        if pool is None:
            M = map
        else:
            M = pool.map
        if inds is None:
            inds = range(self.n_wave)
        self.coeffs[inds, :] = np.array(M(self.train_one_wave,  inds))
        self.trained[inds] = True

    def train_one_wave(self, ind_wave):
        """Do the regression for one wavelength.
        """
        return np.dot(self.Ainv, np.dot(self.X.T, self.training_spectra[:, ind_wave]))

    def restrict_sample(self, bounds=None, **extras):
        if bounds is None:
            return
        good = np.ones(self.n_train, dtype=bool)
        for i, b in bounds.items():
            good = good & within(bound, self.training_labels[i])
        self.training_spectra = self.training_spectra[good, :]
        self.training_labels = self.training_labels[good, ...]
        self.reset()

    def reset(self):
        """Zero out everything in case the training data or features changed.
        """
        # We reloaded the training data, so we should zero out the trained flag
        self.trained = np.zeros(self.n_wave, dtype=bool)
        self.coeffs = np.empty([self.n_wave, self.n_features])
        self.X = None
        self.Ainv = None
        
    @property
    def n_labels(self):
        return len(self.label_names)
        
    @property
    def n_train(self):
        return self.training_spectra.shape[0]

    @property
    def n_wave(self):
        return self.training_spectra.shape[1]

    @property
    def n_features(self):
        return len(self.features) + int(not self.normalize_training_data)


class MILESInterpolator(PSIModel):

    def labels_from_dict(self, **label_dict):
        """Convert from a dictionary of labels to a numpy structured array
        """
        dtype=np.dtype([(n, np.float) for n in self.label_names])
        nl = len(label_dict[self.label_names[0]])
        labels = np.zeros(nl, dtype=dtype)
        for n in self.label_names:
            labels[n] = label_dict[n]
        return labels

    def configure_features(self, **extras):
        """Features based on Eq. 3 of Prugniel 2011, where they are called
        "developments".
        """
        features = (['logt'], ['feh'],
                    ['logg'], ['logt', 'logt'],
                    ['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'],
                    ['logt', 'feh'], ['logt', 'logg'],
                    ['logt', 'logt', 'logg'],
                    ['logt', 'logt', 'feh'],
                    ['logg', 'logg'], ['feh', 'feh'],
                    ['logt', 'logt', 'logt', 'logt', 'logt'],
                    ['logt', 'logg', 'logg'],
                    ['logg', 'logg', 'logg'], ['feh', 'feh'],
                    ['logt', 'feh', 'feh'],
                    ['logg', 'feh'],
                    ['logg', 'logg', 'feh'],
                    ['logg', 'feh', 'feh'],
                    ['teff'], ['teff', 'teff']
                    )
        self.features = features

    def labels_to_features(self, labels):
        """Construct a feature vector from a label structure. This uses features
        that are named by hand, and specified in the ``features`` attribute as
        a tuple of lists.  This method is slower than the ``einsum`` based method,
        but allows for more flexibility and interpretability.

        :param labels:
            Label vector(s).  Structured array of length nobj, with nlabels
            fields.

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        # add bias term if you didn't normalize the training data by
        # subtracting a reference spectrum.
        if self.normalize_training_data:
            X = []
        else:
            X = [np.ones(len(labels))]
        for feature in self.features:
            X.append(np.product(np.array([labels[lname] for lname in feature]), axis=0))
        return np.array(X).T

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).  Optionally subtract off the median
        label and spectrum.
        """
        with h5py.File(training_data, "r") as f:
            self.training_spectra = f['spectra'][:]
            self.training_labels = f['parameters'][:]
        # add and rename labels here
        newfields = ['logt']
        newdata = [np.log10(self.training_labels['teff'])]
        self.training_labels = rfn.append_fields(self.training_labels,
                                                 newfields, newdata)
        self.reset()
                    
    @property
    def label_names(self):
        return self.training_labels.dtype.names


class function_wrapper(object):
    """A hack to make a function pickleable for MPI.
    """
    def __init__(self, function, function_kwargs):
        self.function = function
        self.kwargs = function_kwargs

    def __call__(self, args):
        return self.function(*args, **self.kwargs)


        
def within(bound, value):
    return (value < bound[1]) & (value > bound[0])


if __name__ == "__main__":
    pass
