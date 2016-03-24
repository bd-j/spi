PSI: Polynomial Spectral Interpolator
========

Doing the basic regression thing.


```python
	from model import MILESInterpolator as MILES
	psi = MILES(training_data='miles_prugniel.h5', normalize_labels=False)
	# Only train on warm stars
	psi.restrict_sample({'teff':(4000.0, 9000.0)})
	# Choose polynomial features to train on, here linear terms + logt^2
	psi.features = (['logt'], ['feh'], ['logg'], ['logt', 'logt'])
	psi.train()

    # Plot a predicted spectrum
	spectrum = psi.get_star_spectrum(logt=3.617, logg=4.5, feh=0.0)
	plot(psi.wavelengths, psi.spectrum)
```

References:
----

* Worthey, G., Faber, S. M., Gonzalez, J. J., & Burstein, D. 1994, ApJS, 94, 687 (indices)
* Wu, Y., Singh, H. P., Prugniel, P., Gupta, R., & Koleva, M. 2011, A&A, 525, A71
* Sharma, K., Prugniel, P., & Singh, H. P. 2016, A&A, 585, A64 
* Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015, ApJ, 808, 16
