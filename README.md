PSI: Polynomial Spectral Interpolator
========

Doing the basic regression thing.


```python
	from psi.model import MILESInterpolator as MILES
	psi = MILES(training_data='miles_prugniel.h5', normalize_labels=False)
	# Only train on warm stars
	psi.restrict_sample({'teff':(4000.0, 9000.0)})
	# Choose polynomial features to train on
	psi.features = (['logt'], ['feh'], ['logg'], ['logt', 'logt'])
	psi.train()

    # Plot a predicted spectrum
	spectrum = psi.get_star_spectrum(logt=3.617, logg=4.5, feh=0.0)
	plot(psi.wavelengths, psi.spectrum)
```


