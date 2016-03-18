import numpy as np
from np.linalg import inv


class function_wrapper(object):
    """A hack to make a function pickleable for MPI.
    """
    def __init__(self, function, function_kwargs):
        self.function = function
        self.kwargs = function_kwargs

    def __call__(self, args):
        return self.function(*args, **self.kwargs)


class PSIModel(object):

    def __init__(self, **kwargs):
        self.load_training_data(**kwargs)
        self.configure(**kwargs)
        self.coeffs = np.empty([self.n_wave, self.n_features])
        self.trained = np.zeros(self.n_wave, dtype=bool)
        
    def load_training_data(self, **extras):
        # Need to write this
        self.training_spectra = np.zeros([nwave, nobj])
        self.training_labels = np.zeros([nobj, nlabel])
        self.label_names = []
        dlabel = (self.training_labels.T - self.reference_label)
        
    def label_from_dict(self, **label_dict):
        return np.atleast_2d([label_dict[n] for n in self.label_names])
        
    def get_star_spectrum(**kwargs):
        assert True in self.trained
        label = self.label_from_dict(**kwargs)
        feature = self.labels_to_features(label)
        spectrum = np.dot(self.coeffs, feature.T)
        return np.squeeze(spectrum)

    def configure(self, **extras):
        """Here you set up which terms to use.  This is set up to include all
        linear and quadratic (cross) terms.  """
        inds = combinations_with_replacement(range(self.n_labels), r=2)
        self.inds = np.array(list(inds))
        self.terms = (self.label_names +
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
        self.Ainv = inv(np.dot(self.X, self.X.T))

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
        spectra = self.training_spectra[ind_wave, :] - self.reference_spectrum[ind_wave]
        return np.dot(self.Ainv, np.dot(self.X, spectra))

    def restrict_sample(self, **bounds):
        good = np.ones(self.n_train, dtype=bool)
        for i, b in bounds.items():
            good = good & within(bound, self.training_labels[i])

        return good
            
    @property
    def n_labels(self):
        return len(self.label_names)
        
    @property
    def n_train(self):
        return 

    @property
    def n_wave(self):
        pass



class MILESInterpolater(PSIModel):

    def configure(self):
        """Features based on Eq. 3 of Prugniel 2011, where they are called
        "developments".
        """
        self.terms = (['logt'], ['feh'],
                      ['logg'], ['logt', 'logt'],
                      ['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'],
                      ['logt', 'feh'], ['logt', 'logg'],
                      ['logt', 'logt', 'logg'],
                      ['logt', 'logt', 'feh'],
                      ['logg', 'logg', 'feh', 'feh'],
                      ['logt', 'logt', 'logt', 'logt', 'logt'],
                      ['logt', 'logg', 'logg'],
                      ['logg', 'logg', 'logg'], ['feh', 'feh'],
                      ['logt', 'feh', 'feh'],
                      ['logg', 'feh'],
                      ['logg', 'logg', 'feh'],
                      ['logg', 'feh', 'feh'],
                      ['teff'], ['teff', 'teff']
                      )
    
    def labels_to_features(self, labels):
        """Construct a feature vector from a label vector. This uses terms that
        are named by hand, and specified in the ``terms`` attribute as a tuple of lists.

        :param labels:
            Label vector(s).  Structured array of length nobj, with nlabels fields.

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        X = [np.ones(len(labels))]
        for term in self.terms:
            X.append(np.product(np.array([labels[lname] for lname in term]), axis=0))
        return np.array(X).T

    @property
    def n_features(self):
        return len(self.terms) + 1

def within(bound, value):
    return = (value < bound[1]) & (value > bound[0])


if __name__ == "__main__":
    pass
