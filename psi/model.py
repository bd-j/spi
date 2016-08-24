import numpy as np
from numpy.linalg import inv
try:
    from scipy.spatial import Delaunay, ConvexHull
except:
    pass
from .utils import *
from .library import TrainingSet


__all__ = ["PSIModel", "SimplePSIModel", "FastPSIModel"]


class PSIModel(TrainingSet):

    def __init__(self, unweighted=True, logify_flux=False, **kwargs):
        self.unweighted = unweighted
        self.logify_flux = logify_flux
        self.configure_features(**kwargs)
        self.select(**kwargs)

    def build_training_info(self):
        """Calculate and store quantities about the training set that will be
        used to normalize labels and spectra.
        """
        self.reference_spectrum = np.ones(self.n_wave)
        
    def reset(self):
        """Zero out the coeffs, design_matrix, and Ainv.  Useful in case the
        training data or features changed.
        """
        # We reloaded the training data, so we should zero out the trained flag
        self.trained = np.zeros(self.n_wave, dtype=bool)
        self.coeffs = np.empty([self.n_wave, self.n_features])
        self.X = None
        self.Ainv = None

    def rescale_labels(self, labels, **extras):
        """Rescale the labels.  This should be overridden by subclasses.  It
        will be applied to the training labels before training and to test
        labels when making a prediction.
        """
        raise(NotImplementedError)

    def configure_features(self, **extras):
        """Here you set up which terms to use.  This should be overridden by
        subclasses.
        """
        raise(NotImplementedError)

    def labels_to_features(self, labels):
        """Construct a feature vector from a label vector.  It should be
        reimplemented by subclasses.

        :param labels:
            Label vector(s).  structured array 

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        raise(NotImplementedError)

    def construct_design_matrix(self, **extras):
        """Construct and store the [Nobj x Nfeatures] design matrix and its
        [Nfeature x Nfeature] inverse square.
        """
        self.X = self.labels_to_features(self.training_labels)
        self.Ainv = inv(np.dot(self.X.T, self.X))

    def train(self, inds=None, pool=None, reset=True):
        """Do the regression for the indicated wavelengths.  This can take a
        pool object with a ``map`` method to enable parallelization.
        """
        if reset:
            self.reset()
            self.build_training_info()
            self.construct_design_matrix()
        if pool is None:
            if inds is None:
                inds = slice(None)
            self.coeffs[inds,:] = self.train_one_wave(inds).T
        else:
            M = pool.map
            if inds is None:
                inds = range(self.n_wave)
            self.coeffs[inds, :] = np.array(M(self.train_one_wave,  inds))

        self.trained[inds] = True

    def train_one_wave(self, ind_wave):
        """Do the regression for one wavelength.
        """
        spec = self.training_spectra[:, ind_wave] / self.reference_spectrum[ind_wave]
        if self.logify_flux:
            # logify before training
            spec = np.log(spec)
        if (not self.has_errors) or (self.unweighted):
            return self.ordinary_least_squares(spec)
        else:
            weights = self.training_weights[:, ind_wave] #* self.reference_spectrum[ind_wave]**2
            if self.logify_flux:
                 weights *= spec**2
            return self.weighted_least_squares(spec, weights)

    def ordinary_least_squares(self, spec):
        """OLS
        """
        return np.dot(self.Ainv, np.dot(self.X.T, spec))

    def weighted_least_squares(self, spec, weights):
        """Weighted least-squares.  Vectorized to work simultaneously on large
        numbers of wavelengths
        """
        ww = weights.T
        wx = ww[:, :, None] * self.X

        B = np.dot(self.X.T, weights * spec).T #nwave x nfeature
        # This is the time suck
        A = np.dot(self.X.T, wx).transpose(1,0,2) #nwave x nfeature x nfeature
        return np.linalg.solve(A, B).T

    def get_star_spectrum(self, check_coverage=False, **kwargs):
        """Get an interpolated spectrum at the parameter values (labels)
        specified as keywords.
        """
        assert True in self.trained
        labels = make_struct(**kwargs)
        features = self.labels_to_features(labels)
        spectrum = np.dot(self.coeffs, features.T)
        if self.logify_flux:
            # Delogify
            spectrum = np.exp(spectrum)
        if check_coverage:
            is_inside = self.inside_hull(labels)
            return np.squeeze(spectrum.T * self.reference_spectrum), is_inside
        return np.squeeze(spectrum.T * self.reference_spectrum)

    @property
    def label_names(self):
        return self.library_labels.dtype.names
    
    @property
    def n_labels(self):
        return len(self.label_names)

    @property
    def n_train(self):
        return self.training_labels.shape[0]

    @property
    def n_wave(self):
        return self.library_spectra.shape[1]

    @property
    def n_features(self):
        return len(self.features) + 1
        

class SimplePSIModel(PSIModel):
    """A simpler version of PSIModel that overrides the ``labels_to_features``,
    ``rescale``, ``configure_features``, and ``build_training_info`` methods to
    be simpler and more expressive.  In this model labels and features can be
    accessed and created by name, instead of needing to have properly ordered
    vectors.  Furthermore not all labels need to be used.
    """

    def configure_features(self, **extras):
        """Features based on Eq. 3 of Prugniel 2011, where they are called
        "developments".
        """
        features = (['logt'], ['feh'],
                    ['logg'], ['logt', 'logt'],
                    #['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'],
                    #['logt', 'feh'], ['logt', 'logg'],
                    )
        self.features = features

    def labels_to_features(self, labels):
        """Construct a feature vector from a label structure. This uses
        features that are named by hand, and specified in the ``features``
        attribute as a tuple of lists.  This method is slower than the
        ``einsum`` based method of FastPSIModel, but allows for more flexibility
        and interpretability.

        :param labels:
            Label vector(s).  Structured array of length nobj, with nlabels
            fields.

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        slabels = self.rescale_labels(labels)
        # add bias term
        X = [np.ones(len(slabels))]
        for feature in self.features:
            X.append(np.product(np.array([slabels[lname]
                                          for lname in feature]), axis=0))
        return np.array(X).T

    def rescale_labels(self, label):
        nlabel = label.copy()
        nlabel['logt'] -= 3.7617
        nlabel['logg'] -= 4.44
        return nlabel

    def build_training_info(self):
        self.reference_index = None
        self.reference_spectrum = self.training_spectra.std(axis=0)

    def inside_hull(self, labels):
        """This method checks whether a requested set of labels is inside the
        convex hull of the training data.  Returns a bool.  This method relies
        on Delauynay Triangulation and is thus exceedlingly slow for large
        dimensionality.
        """
        L = flatten_struct(self.training_labels, use_labels=self.used_labels)
        l = flatten_struct(labels, use_labels=self.used_labels)
        hull = Delaunay(L.T)
        return hull.find_simplex(l.T) >= 0
        
    @property
    def used_labels(self):
        try:
            used_labels = np.unique([n for f in self.features for n in f])
            return tuple(used_labels)
        except(AttributeError):
            return []


class FastPSIModel(PSIModel):

    def configure_features(self, **extras):
        """Here you set up which terms to use.  This is set up to include all
        linear and quadratic (cross) terms for all label_names fields.  This
        should be overridden by subclasses.
        """
        try:
            qinds = combinations_with_replacement(range(self.n_labels), r=2)
            qnames = combinations_with_replacement(self.label_names, r=2)
            self.qinds = np.array(list(qinds))
            self.features = (self.label_names + list(qnames))
        except(AttributeError):
            # Labels not set yet.
            pass

    def labels_to_features(self, labels):
        """Construct a feature vector from a label vector.  This is a simple
        quadratic model, and a placeholder.  It should be reimplemented by
        subclasses.

        :param labels:
            Label vector(s).  structured array 

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        linear = self.rescale_labels(labels)
        quad = np.einsum('...i,...j->...ij', linear, linear)[:, self.qinds[:, 0], self.qinds[:, 1]]
        return np.hstack([linear, quad])

    def rescale_labels(self, labels, **extras):
        """Rescale the labels. It will be applied to the training labels before
        training and to test labels when making a prediction.

        For the particular rescaling given here, to reconstruct absolute labels
        from a normalized label, use:
            label = label_range * normed_label + reference_label
            spectrum = normed_spectrum + reference_spectrum
        """
        normlabels = flatten_struct(labels) - flatten_struct(self.reference_label)
        return normlabels / flatten_struct(self.training_label_range)

    def build_training_info(self):
        """Calculate and store quantities about the training set that will be
        used to normalize labels and spectra.  Here we try to normalize all
        labels to the range [-1, 1], but this is probably broken.
        """

        normlabels = flatten_struct(self.training_labels)
        lo = normlabels.min(axis=0)
        hi = normlabels.max(axis=0)
        normlabels = (normlabels - lo) / (hi - lo)
        self.reference_index = np.argmin(np.sum((normlabels - 0.5)**2, axis=1))
        self.reference_spectrum = self.training_spectra[self.reference_index, :]
        self.reference_label = self.training_labels[self.reference_index]
        self.training_label_range = hi - lo


if __name__ == "__main__":
    pass
