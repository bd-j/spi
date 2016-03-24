import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.linalg import inv
from scipy.spatial import Delauynay, ConvexHull
import h5py
import astropy.io.fits as pyfits

__all__ = ["PSIModel", "MILESInterpolator", "TGM"]


class PSIModel(object):

    def __init__(self, normalize_labels=False, unweighted=True, **kwargs):
        self.force_ordinary = unweighted
        self.normalize_labels = normalize_labels
        self.load_training_data(**kwargs)
        self.restrict_sample(**kwargs)
        self.configure_features(**kwargs)
        self.reset()

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Store these in the `training_labels` and
        `training_spectra` attributes
        """
        with h5py.File(training_data, "r") as f:
            self.training_spectra = f['spectra'][:]
            self.training_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]
        # add and rename labels here
        self.label_names = self.training_labels.dtype.names
        self.build_training_info()
        self.has_errors = False
        #self.reset()
        
    def build_training_info(self):
        """Calculate and store quantities about the training set tht will be
        used to normalize labels and spectra
        """
        if self.normalize_labels:
            normlabels = flatten_struct(self.training_labels)
            lo = normlabels.min(axis=0)
            hi = normlabels.max(axis=0)
            normlabels = (normlabels - lo) / (hi - lo)
            self.reference_index = np.argmin(np.sum((normlabels - 0.5)**2, axis=1))
            self.reference_spectrum = self.training_spectra[self.reference_index, :]
            self.reference_label = self.training_labels[self.reference_index]
            self.training_label_range = hi - lo
        else:
            self.reference_index = None
            self.reference_spectrum = np.zeros(self.n_wave)
            self.reference_label = np.zeros(1, dtype=self.training_labels.dtype)
            self.training_label_range = 1.0
        
    def reset(self):
        """Zero out everything in case the training data or features changed.
        """
        # We reloaded the training data, so we should zero out the trained flag
        self.trained = np.zeros(self.n_wave, dtype=bool)
        self.coeffs = np.empty([self.n_wave, self.n_features])
        self.X = None
        self.Ainv = None

    def restrict_sample(self, bounds=None, **extras):
        if bounds is None:
            return
        good = np.ones(self.n_train, dtype=bool)
        for name, bound in bounds.items():
            good = good & within(bound, self.training_labels[name])
        self.training_spectra = self.training_spectra[good, :]
        self.training_labels = self.training_labels[good, ...]
        self.build_training_info()
        self.reset()

    def leave_out(self, inds):
        """Remove training objects specified by `inds`.  Useful for
        leave-one-out validation.

        :param inds:
            Int, array of ints, or slice specifying which training objects to
            remove.  Passed to numpy.delete
        """
        self.training_spectra = np.delete(self.training_spectra, inds, axis=0)
        self.training_labels = np.delete(self.training_labels, inds)
        self.build_training_info()
        self.reset()

    def rescale(self, labels, **extras):
        """Normalize labels by by their range in the training set (stored as
        `label_range`), and subtract the "median" label and spectrum of the
        training set, which are stored as `reference_label` and
        `reference_spectrum`.

        To reconstruct absolute labels from a normalized label, use:
            label = label_range * normed_label + reference_label
            spectrum = normed_spectrum + reference_spectrum
        """
        normlabels = flatten_struct(labels) - flatten_struct(self.reference_label)
        return normlabels / self.training_label_range

    def configure_features(self, **extras):
        """Here you set up which terms to use.  This is set up to include all
        linear and quadratic (cross) terms.
        """
        qinds = combinations_with_replacement(range(self.n_labels), r=2)
        qnames = combinations_with_replacement(self.label_names, r=2)
        self.qinds = np.array(list(qinds))
        self.features = (self.label_names + list(qnames))

    def labels_to_features(self, labels):
        """Construct a feature vector from a label vector.  This is a simple
        quadratic model, and a placeholder.  It should be reimplemented by
        subclasses.

        :param labels:
            Label vector(s).  structured array 

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        linear = self.rescale(labels)
        quad = np.einsum('...i,...j->...ij', linear, linear)[:, self.qinds[:, 0], self.qinds[:, 1]]
        return np.hstack([linear, quad])

    def inside_hull(self, labels):
        L = flatten_struct(self.training_labels, use_labels=self.label_names)
        l = flatten_struct(labels, use_labels=self.label_names)
        hull = Delaunay(ConvexHull(L))
        return hull.find_simplex(l)>=0
    
    def construct_design_matrix(self, **extras):
        """Construct and store the [Nobj x Nfeatures] design matrix and its
        [Nfeature x Nfeature] inverse square.
        """
        self.X = self.labels_to_features(self.training_labels)
        self.Ainv = inv(np.dot(self.X.T, self.X))

    def train(self, inds=None, pool=None):
        """Do the regression for the indicated wavelengths.  This can take a
        pool object with a ``map`` method to enable parallelization.
        """
        if (self.X is None) or (self.Ainv is None):
            self.construct_design_matrix()
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
        spec = self.training_spectra[:, ind_wave] / self.reference_spectrum[ind_wave]
        if (not self.has_errors) or (self.force_ordinary):
            return self.ordinary(spec)
        else:
            weights = self.training_weights[:, ind_wave] * self.reference_spectrum[ind_wave]**2
            return self.weighted(spec, weights)
        
    def ordinary(self, spec):
        """OLS
        """
        return np.dot(self.Ainv, np.dot(self.X.T, spec))

    def weighted(self, spec, weights):
        """Weighted least-squares. Should use woodbury matrix lemma here to
        update self.Ainv instead of reinverting each time.
        """
        Xp = np.dot(weights, self.X)
        Ainv = inv(np.dot(Xp.T, Xp))
        return np.dot(Ainv, np.dot(Xp.T, spec * weights))

    def get_star_spectrum(self, check_coverage=False, **kwargs):
        """Get an interpolated spectrum at the parameter values (labels)
        specified as keywords.
        """
        assert True in self.trained
        labels = make_struct(**kwargs)
        if check_coverage:
            inside = self.inside_hull(labels)
        features = self.labels_to_features(labels)
        spectrum = np.dot(self.coeffs, features.T)
        return np.squeeze(spectrum.T * self.reference_spectrum)

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
        return len(self.features) + int(not self.normalize_labels)
        

class MILESInterpolator(PSIModel):


    def rescale(self, label):
        nlabel = label.copy()
        nlabel['logt'] -= 3.7617
        nlabel['logg'] -= 4.44
        return nlabel
    
    def configure_features(self, **extras):
        """Features based on Eq. 3 of Prugniel 2011, where they are called
        "developments".
        """
        features = (['logt'], ['feh'],
                    ['logg'], ['logt', 'logt'],
                    #['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'],
                    #['logt', 'feh'], ['logt', 'logg'],
                    #['logt', 'logt', 'logg'],
                    #['logt', 'logt', 'feh'],
                    #['logg', 'logg'], ['feh', 'feh'],
                    #['logt', 'logt', 'logt', 'logt', 'logt'],
                    #['logt', 'logg', 'logg'],
                    #['logg', 'logg', 'logg'],
                    #['feh', 'feh'],
                    #['logt', 'feh', 'feh'],
                    #['logg', 'feh'],
                    #['logg', 'logg', 'feh'],
                    #['logg', 'feh', 'feh'],
                    #['teff'], ['teff', 'teff']
                    # The following features are directly from the ulyss code for v3 miles:
                    #['logt', 'logt', 'logt', 'logg'],
                    #['logt', 'logt', 'logt', 'logt', 'logg'],
                    #['logt', 'logt', 'logt', 'feh'],
                    #['logt', 'logt', 'logg', 'logg'],
                    #['logt', 'logt', 'logg', 'logg', 'logg']
                    )
        self.features = features

    def labels_to_features(self, labels):
        """Construct a feature vector from a label structure. This uses
        features that are named by hand, and specified in the ``features``
        attribute as a tuple of lists.  This method is slower than the
        ``einsum`` based method, but allows for more flexibility and
        interpretability.

        :param labels:
            Label vector(s).  Structured array of length nobj, with nlabels
            fields.

        :returns X:
            Design matrix, ndarray of shape (nobj, nfeatures)
        """
        slabels = self.rescale(labels)
        # add bias term if you didn't normalize the training data by
        X = [np.ones(len(slabels))]
        for feature in self.features:
            X.append(np.product(np.array([slabels[lname]
                                          for lname in feature]), axis=0))
        return np.array(X).T

    def load_training_data(self, training_data='', **extras):
        """Read an HDF5 file with `parameters` a structured ndarray and
        `spectra` an ndarray.  Convert to a structured array of labels of
        length `ntrain` with `nlabel` fields. and an ndarray of training
        spectra of shape (nwave, ntrain).  Optionally subtract off the median
        label and spectrum.
        """
        self.has_errors = False
        with h5py.File(training_data, "r") as f:
            self.training_spectra = f['spectra'][:]
            self.training_labels = f['parameters'][:]
            self.wavelengths = f['wavelengths'][:]
            try:
                self.training_weights = 1/(f['uncertainty'][:]**2)
                self.has_errors = True
            except:
                pass
            ancillary = f['ancillary'][:]
        # add and rename labels here.  Note that not all labels need or will be
        # used in the feature generation
        newfield = ['logt', 'miles_id']
        newdata = [np.log10(self.training_labels['teff']), ancillary['miles_id']]
        self.training_labels = rfn.append_fields(self.training_labels,
                                                 newfield, newdata, usemask=False)
        self.build_training_info()
        # change where spectra are normalized
        #self.training_spectra /= self.training_spectra[:, 1000][:, None]
        #self.reset()

    def renormalize_training_spectra(self, normwave):
        ind_wave = np.argmin(np.abs(self.wavelengths - normwave))
        self.training_spectra /= self.training_spectra[:, ind_wave][:, None]
        
    def build_training_info(self):
        self.reference_index = None
        self.reference_spectrum = self.training_spectra.std(axis=0)
        self.reference_label = np.zeros(self.n_labels)
        self.training_label_range = 1.0
        self._dtri = Delaunay(
        
    @property
    def label_names(self):
        try:
            used_labels = np.unique([n for f in self.features for n in f])
            return tuple(used_labels)
        except(AttributeError):
            return []



def flatten_struct(struct, use_labels=None):
    """This is slow, should be replaced with a view-based method.
    """
    if use_labels is None:
        return np.array(struct.tolist())
    else:
        return np.array([struct[n] for n in use_labels])

def dict_struct(struct):
    """Convert from a structured array to a dictionary.  This shouldn't really
    be necessary.
    """
    return dict([(n, struct[n]) for n in struct.dtype.names])
    
def make_struct(**label_dict):
        """Convert from a dictionary of labels to a numpy structured array
        """
        dtype = np.dtype([(n, np.float) for n in label_dict.keys()])
        try:
            nl = len(label_dict[label_dict.keys()[0]])
        except:
            nl = 1
        labels = np.zeros(nl, dtype=dtype)
        for n in label_dict.keys():
            labels[n] = label_dict[n]
        return labels
 
    
def within(bound, value):
    return (value < bound[1]) & (value > bound[0])


class function_wrapper(object):
    """A hack to make a function pickleable for MPI.
    """
    def __init__(self, function, function_kwargs):
        self.function = function
        self.kwargs = function_kwargs

    def __call__(self, args):
        return self.function(*args, **self.kwargs)


if __name__ == "__main__":
    pass
