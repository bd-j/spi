import numpy as np

class function_wrapper(object):
    """A hack to make a function pickleable for MPI.
    """
    def __init__(self, function, function_kwargs):
        self.function = function
        self.kwargs = function_kwargs

    def __call__(self, args):
        return self.function(*args, **self.kwargs)


def prediction(params, interpolator):
    
    return interpolator.get_spectrum(**pardict)
    

def fit(spectrum, interpolator):
    assert len(spectrum) == interpolator.n_wave
    
    
