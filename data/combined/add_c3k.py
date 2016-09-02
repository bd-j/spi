import sys
import numpy as np
import matplotlib.pyplot as pl
import h5py

from numpy.lib import recfunctions as rfn
from scipy.spatial import Delaunay

from psi.library_models import SimplePSIModel
from psi.utils import dict_struct, flatten_struct

from prospect.utils.smoothing import smoothspec


lightspeed = 2.998e18  # AA/s
log_rsun_cgs = np.log10(6.955) + 10
log_lsun_cgs = np.log10(3.839) + 33
log_SB_cgs = np.log10(5.670367e-5)
log_SB_solar = log_SB_cgs + 2 * log_rsun_cgs - log_lsun_cgs


def select_outside(miles, c3k, mask_mann=True, **extras):
    """
    :returns outside:
        Boolean indicating whether an object in C3K is outside the convex hull
        defined by the objects in MILES
    """
    # Read MILES
    mlib = h5py.File(miles, 'r')
    mparams = mlib['parameters'][:]
    mparams = rfn.append_fields(mparams, ['logt'], [np.log10(mparams['teff'])],
                                usemask=False)
    mlib.close()
    if mask_mann:
        mann = mparams['miles_id'] == 'mdwarf'
        mparams = mparams[~mann]

    # Read C3K
    try:
        clib = h5py.File(c3k, 'r')
        cparams = clib['parameters'][:]
        bad = np.max(clib['spectra'], axis=-1) < 1e-32
    except(IOError, AttributeError):
        cparams = c3k
        bad = np.zeros(len(cparams), dtype=bool)
    
    # Find C3K objects outside the MILES convex hull
    inside = inside_hull(mparams, cparams, **extras)

    return ~inside

    
def inside_hull(primary, secondary, use_labels=['logt', 'logg', 'feh'],
                **extras):
    L = flatten_struct(primary, use_labels=use_labels)
    l = flatten_struct(secondary, use_labels=use_labels)
    hull = Delaunay(L.T)
    return hull.find_simplex(l.T) >= 0


def broaden(wave, spec, outwave=None, break_wave=7430., inres=1e4*2.355, **extras):
    miles_resolution = 2.54 / 2.355 # in terms of dispersion
    irtf_resolution = 2000.0 * 2.355 # in terms of dispersion
    # MILES portion
    miles = outwave < break_wave
    flux_miles = [smoothspec(wave, s, miles_resolution,
                            outwave=outwave[miles], smoothtype='lambda',
                            fftsmooth=True) for s in spectra]
    # IRTF portion
    irtf = outwave >= break_wave
    flux_irtf = [smoothspec(wave, s, irtf_resolution,
                            outwave=outwave[irtf], smoothtype='R',
                            fftsmooth=True, inres=inres) for s in spectra]

    spec = np.hstack([np.array(flux_miles), np.array(flux_irtf)])
        
    return outwave, spec


def rectify_miles(miles, outname=None, **extras):
    """
    """
    with h5py.File(miles, "r") as f:
        spectra = f['spectra'][:]
        labels = f['parameters'][:]
        wave = f['wavelengths'][:]
        ancillary = f['ancillary'][:]
        try:
            unc = f['unc'][:]
            #unc[:, :4350] /= 1e12
        except:
            print('fudging errors')
            unc = 0.1 * spectra

    newfield = ['logt', 'miles_id', 'name', 'logl', 'luminosity']
    newdata = [np.log10(labels['teff']),
               ancillary['miles_id'], ancillary['name'],
               ancillary['logl'], 10**ancillary['logl']]
    labels = rfn.append_fields(labels, newfield, newdata, usemask=False)

    if outname is not None:
        write_h5(outname, wave, spectra, unc, labels)

    return wave, spectra, unc, labels        


def rectify_c3k(c3k, selection=None, miles=None, broaden=True,
                outwave=None, outname=None, mask_mann=False,
                **broaden_kwargs):
    """
    """
    # Read C3K
    with h5py.File(c3k, "r") as f:
        spectra = f['spectra'][:]
        labels = f['parameters'][:]
        wave = f['wavelengths'][:]
    
    # Add new label info
    nobj = len(labels)
    newcols = ['miles_id', 'name', 'logl', 'luminosity']
    newdata = [np.array(nobj * ['c3k']), np.array(nobj * ['c3k']),
               np.zeros(nobj), np.ones(nobj)]
    # Deal with oldstyle files that only have Z, not feh
    if 'Z' in labels.dtype.names:
        newcols += ['feh']
        newdata += [np.log10(labels['Z']/0.0134)]
    labels = rfn.append_fields(labels, newcols, newdata, usemask=False)
    
    # Ditch stars within the miles convex hull
    if selection is None:
        has_flux = np.max(spectra, axis=-1) > 1e-32
        if miles is not None:
            outside_hull = select_outside(miles, labels, mask_mann=mask_mann)
            selection = outside_hull & has_flux
        else:
            selection = has_flux
    spectra = spectra[selection, :]
    labels = labels[selection]

    # Renormalize to Lsun/Hz/solar luminosity
    logl, log4pi = 0.0, np.log10(4 * np.pi)
    twologR = (logl+log_lsun_cgs) - 4 * labels['logt'] - log_SB_cgs - log4pi
    spectra *= 10**(twologR[:, None] + 2 * log4pi - log_lsun_cgs)
    
    # Broaden
    if broaden:
        w, spec = broaden(wave / 1e4, spectra, outwave=outwave, **broaden_kwargs)
    else:
        w = outwave
        spec = np.array([np.interp(outwave, wave / 1e4, s) for s in spectra])

    # Dummy uncertainties
    unc = 0.1 * spec

    if outname is not None:
        write_h5(outname, w, spec, unc, labels)

    return w, spec, unc, labels


def combine_miles_c3k(mlib='', clib='', c3k_weight=1e-1,
                      outname='test.h5',
                      all_c3k=False, **kwargs):
    """
    """
    # get MILES
    mdat = rectify_miles(mlib, outname=None, **kwargs)
    wave, spectra, unc, labels = mdat

    # add C3K
    if all_c3k:
        # Don't remove C3K stars based on MILES locations.  We will do it later
        miles = None
    else:
        miles = mlib
    print(miles)
    cdat = rectify_c3k(clib, miles=miles, outwave=wave, inres=1e4*2.35, **kwargs)
    w, s, u, lab = cdat
    assert np.allclose(wave, w)
    spec = np.vstack([spectra, s])
    # relative weighting
    # norm = np.nanmedian(1/unc**2) * c3k_weight
    unc = np.vstack([unc, u ])
    # build a useful structured array and fill it
    newlabels = np.zeros(len(lab), dtype=labels.dtype)
    for l in labels.dtype.names:
            if l in lab.dtype.names:
                newlabels[l] = lab[l]

    labels = np.hstack([labels, newlabels])
    write_h5(outname, wave, spec, unc, labels)
    return wave, spec, unc, labels


def write_h5(outname, wave, spec, unc, label):
    with h5py.File(outname, "w") as f:
        w = f.create_dataset('wavelengths', data=wave)
        s = f.create_dataset('spectra', data=spec)
        l = f.create_dataset('parameters', data=label)
        u = f.create_dataset('uncertainties', data=unc)


if __name__ == "__main__":
        
    #clibname = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/ckc_R10k.h5'
    clibname = '/Users/bjohnson/Codes/SPS/ckc/ckc/lores/irtf/ckc14_irtf.flat.h5'
    #mlibname = '/Users/bjohnson/Projects/psi/data/combined/with_mdwarfs_culled_lib_snr_cut.h5'
    mlibname = '/Users/bjohnson/Projects/psi/data/combined/culled_libv3_w_conv_mdwarfs_w_unc_tc.h5'
    #clib = h5py.File(clibname, 'r')
    #mlib = h5py.File(mlibname, 'r')


    #outname = 'culled_lib_w_unc_w_c3k.h5'
    #combdat = combine_miles_c3k(mlibname, clibname, outname=outname, mask_mann=True, broaden=False)
    outname = 'culled_libv3tc_w_mdwarfs_w_unc_w_allc3k.h5'
    combdat = combine_miles_c3k(mlibname, clibname, outname=outname, all_c3k=True, broaden=False)

    #clib.close()
    #mlib.close()
