PSI: Polynomial Spectral Interpolator
========

Doing the basic regression thing.


```python
	from psi.model import MILESInterpolator as MILES
	psi = MILES(training_data='miles_prugniel.h5', normalize_labels=False)
	psi.restrict_sample({'teff':(4000.0, 9000.0)})
	psi.features = (['logt'], ['feh'], ['logg'], ['logt', 'logt'])
	psi.train()
	```


