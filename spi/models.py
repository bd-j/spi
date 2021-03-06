import numpy as np
from numpy.linalg import inv
try:
    from scipy.spatial import Delaunay, ConvexHull
except:
    pass
from .utils import *
from .trainingset import TrainingSet


__all__ = ["SPIModel", "SimpleSPIModel", "FastSPIModel"]



class SPIModel(TrainingSet):

    def __init__(self, unweighted=True, logify_flux=False, **kwargs):
        self.unweighted = unweighted
        self.logify_flux = logify_flux
        self.configure_features(**kwargs)
        self.select(**kwargs)

    def build_training_info(self):
        """Calculate and store quantities about the training set that can be
        used to normalize labels and spectra. This will typically be
        subclassed.
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

    def construct_design_matrix(self, regularization=None, **extras):
        """Construct and store the [Nobj x Nfeatures] design matrix and its
        [Nfeature x Nfeature] inverse square.

        :param ridge:
            Add a regularization term along the diagonal.  not implemented
        """
        self.X = self.labels_to_features(self.training_labels)
        if regularization is not None:
            raise(NotImplementedError)
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
        # Get weights if neccesary.  Takes into account logification
        relative_weights = self.get_weights(ind_wave, spec)
        if self.logify_flux:
            # logify before training
            spec = np.log(spec)
        if relative_weights is None:
            return self.ordinary_least_squares(spec)
        else:
            return self.weighted_least_squares(spec, relative_weights)

    def get_weights(self, ind_wave, spec):
        """Get weights for each star, for each wavelength.  

        :param ind_wave:
            The indices of the wavelengths at which to calculate the weights.

        :param spec:
            Linear flux units.

        :returns weights:
            The weights to be applied to each data pooint in weighted regression.
        """
        if (not self.has_errors) or (self.unweighted):
            return None
        else:
            if self.logify_flux:
                # if training log(flux), use relative (S/N)**2 for weights
                relative_weights = self.training_snr[:, ind_wave]**2
            else:
                # else just use the inverse flux variance (S/N)**2 /S**2 
                relative_weights = (self.training_snr[:, ind_wave] / spec)**2
                
            return relative_weights

    def ordinary_least_squares(self, spec):
        """OLS
        """
        return np.dot(self.Ainv, np.dot(self.X.T, spec))

    def weighted_least_squares(self, spec, weights):
        """Weighted least-squares.  Vectorized to work simultaneously on large
        numbers of wavelengths
        """
        ww = weights.T # nwave x ntrain
        wx = ww[:, :, None] * self.X # nwave x ntrain x nfeature

        b = np.dot(self.X.T, weights * spec).T # nwave x nfeature
        # This is the time suck
        a = np.matmul(self.X.T, wx) # nwave x nfeature x nfeature
        #a = np.dot(self.X.T, wx).transpose(1,0,2)
        return np.linalg.solve(a, b).T

    def get_star_spectrum(self, check_coverage=False, **kwargs):
        """Get an interpolated spectrum at the parameter values (labels)
        specified as keywords.
        """
        assert True in self.trained, "Not trained yet!"
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

    def dump_coeffs_ascii(self, filename='test.dat'):
        """Write wavelengths, reference spectrum, and coefficients to an ascii file"
        """
        beta = np.vstack([self.wavelengths, self.reference_spectrum, self.coeffs.T])
        if self.logify_flux:
            hdr = 'F = beta[0] * exp(beta[1] + beta[2:]*X) \n'
        else:
            hdr = 'F = beta[0] * (beta[1] + beta[2:]*X) \n'
        hdr += ('X = ' + (self.n_features-1) * '{}\n').format(*self.features)
        hdr += 'lambda(micron), beta\n'
        np.savetxt(filename, beta.T, header=hdr)

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
        


class SKModel(SPIModel):

    """
    LinearRegression:
    Ridge(CV): alpha, (alphas)
    [MultiTask]Lasso(CV): alpha (alphas)
    [MultiTask]ElasticNet(CV): alpha, l1_ratio (alphas, l1_ratio)
    BayesianRidge:
    """
    
    def __init__(self, logify_flux=False, fit_intercept=True, model_kwargs={}, **kwargs):
        self.logify_flux = logify_flux
        self.fit_intercept = fit_intercept
        self.configure_features(**kwargs)
        self.select(**kwargs)

        self.model = model(fit_intercept=self.fit_intercept, **model_kwargs)

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
            spec = np.log(spec)
        self.model.fit(self.X, spec)
        coeffs = model.coef_
        if self.fit_intercept:
            # do something
            pass
        return coeffs

    def construct_design_matrix():
        self.scaler = None
        X_train = self.labels_to_features(self.training_labels)
        self.scaler = preprocessing.StandardScaler().fit(X_train)
        self.X = self.scaler.transform(X_train)

    def configure_features(self, orders=range(1, 5), **extras):
        from itertools import combinations_with_replacement as cwr
        all_features = [t for order in orders for t in cwr(['logt', 'logg', 'feh'], order)]

    def labels_to_features(self, labels):
        if self.fit_intercept:
            X = []
        else:
            X = [np.ones(len(labels))]
        for feature in self.features:
            x = np.array([labels[n] for n in feature])
            X.append(x.prod(axis=0))
        if self.scaler is None:
            return np.array(X).T
        else:
            return self.scaler.transform(np.array(X).T)

    def get_star_spectrum(self, check_coverage=False, **kwargs):
        assert True in self.trained, "Not trained yet!"
        labels = make_struct(**kwargs)
        features = self.labels_to_features(labels)
        spectrum = self.intercepts + np.dot(self.coeffs, features.T)
        if self.logify_flux:
            # Delogify
            spectrum = np.exp(spectrum)
        if check_coverage:
            is_inside = self.inside_hull(labels)
            return np.squeeze(spectrum.T * self.reference_spectrum), is_inside
        return np.squeeze(spectrum.T * self.reference_spectrum)

    def dump_coeffs(self):
        pass


class SimpleSPIModel(SPIModel):
    """A simpler version of SPIModel that overrides the ``labels_to_features``,
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
        ``einsum`` based method of FastSPIModel, but allows for more flexibility
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

        

class FastSPIModel(SPIModel):

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
