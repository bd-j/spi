import sys
import itertools
import numpy as np
import matplotlib.pyplot as pl
import h5py


def hypercubify(libparams, labels, output=None):
    gridpoints = {}
    for p in labels:
        gridpoints[p] = np.unique(libparams[p])

    # Digitize the library parameters
    X = np.array([np.digitize(libparams[p], bins=gridpoints[p]) - 1
                  for p in labels])

    shape = [len(gridpoints[p]) for p in labels]

    if output is None:
        # Make the output be a pointer into the original flat array
        output = np.arange(len(libparams))[:, None]
    Z = np.nan + np.zeros(shape + [output.shape[1]])
    Z[tuple(X)] = output
    points = tuple([gridpoints[p] for p in labels])

    return X, Z, points


def params_to_grid(params, labels, gridpoints):
    X = np.array([np.digitize(params[p], bins=gridpoints[p]) - 1
                  for p in labels])
    return X


def linterp_at(x, Z, itype="corners", outputs=None, points=None):
    """
    :param x:
        shape (ndim,)

    :param Z:
        shape (..., nout)

    :param outputs:
        shape (nlib, nout)
    """
    
    truth = Z[tuple(x)]

    if itype == "corners":
        # --- Corners ---
        # lower corner of hypercube
        indices = x - 1
        # all other corners of hypercube
        # note this does not include points on the face of the hypercube
        corners = np.array(list(itertools.product(*[[i, i + 2] for i in indices])))
        return combine(corners, Z, x, outputs=outputs, points=points)

    if itype == "faces":
        # --- Faces ----
        indices = x
        ndim = len(indices)
        k = np.diag(np.ones(ndim))
        faces = indices + np.vstack([k, -k])
        return combine(faces.astype(int), Z, x, outputs=outputs, points=points)

    if itype == "edges":
        # --- Edges ---
        # do ndim interpolations just along enclosing models.
        sout = []
        k = np.diag(np.ones(ndim))
        for i in range(ndim):
            lo, hi = (x + k[i]).astype(int), (x - k[i]).astype(int)
            edges = np.array([lo, hi])
            sout += combine(edges, Z, x, outputs=outputs, points=points)            
        return np.array(sout)


def combine(vertices, Z, x, outputs=None, points=None):
    """
    :param vertices:
        shape (nvert, ndim)

    :param Z:
        shape (..., nout) or (..., 1)

    :param outputs:
        shape (ngrid, nout)

    :param points:
        sequence of length ndim giving the value of the grid points in each dimension

    :returns out:
        shape (nout)
    """
    if outputs is None:
        nout = Z.shape[-1]
    else:
        nout = outputs.shape[-1]

    if np.any(vertices < 0):
        return np.nan + np.zeros(nout)
    try:
        zz = Z[tuple(vertices.T)]
    except(IndexError):
        return np.nan + np.zeros(nout)

    if points is None:
        weights = np.ones(len(vertices)) * 1.0  / len(vertices)
    else:
        target = np.array([p[x[i]] for i, p in enumerate(points)]).T
        params = np.array([p[vertices[:, i]] for i, p in enumerate(points)]).T
        prange = params.max(axis=0) - params.min(axis=0)
        weights = 1.0 - np.abs((params - target[None, :]) / prange[None, :])
        weights = weights.prod(axis=-1)
        #assert np.isclose(weights.sum(), 1.0)
        weights /= weights.sum()

    if outputs is None:
        out = np.dot(weights, zz)
    else:
        out = np.dot(weights, outputs[zz])
    return out


def renormalize(labels, spectra):
    lightspeed = 2.998e18  # AA/s
    log_rsun_cgs = np.log10(6.955) + 10
    log_lsun_cgs = np.log10(3.839) + 33
    log_SB_cgs = np.log10(5.670367e-5)
    log_SB_solar = log_SB_cgs + 2 * log_rsun_cgs - log_lsun_cgs
    logl, log4pi = 0.0, np.log10(4 * np.pi)
    # This is in cm
    twologR = (logl + log_lsun_cgs) - 4 * labels['logt'] - log_SB_cgs - log4pi

    # Now multiply by 4piR^2, with another 4pi for the solid angle
    spectra *= 10**(twologR[:, None] + 2 * log4pi)
    return spectra


def select(features, bounds):
    good = np.ones(len(features), dtype=bool)    
    for k, v in list(bounds.iteritems()):
        good = good & (features[k] > v[0]) & (features[k] < v[1])
    return good


if __name__ == "__main__":

    itype = "faces"
    fn = "/Users/bjohnson/Codes/SPS/ckc/ckc/spectra/lores/c3k_v1.3_R5K.h5"

    with h5py.File(fn, "r") as f:
        features = f["parameters"][:]
        sel = features["afe"] == 0
        output = f["spectra"][:][sel]
        features = features[sel]
        wave = f["wavelengths"][:]

    output = renormalize(features, output)
    labels = list(features.dtype.names)
    _ =labels.pop(labels.index("afe"))

    # remove a couple zero spectra
    bad = np.nanmax(output, axis=-1) <= 0
    output[bad, :] = np.nan
    
    X, cube, points = hypercubify(features, labels, output=output)
    lncube = np.log(cube)

    # --- Simple method ----
    #sparse_points = [p[::2] for p in points]
    #sparse_cube = cube[::2, ::2, ::2, :]
    #test_points = np.meshgrid(*[p[1:-1:2] for p in points], indexing="ij")
    #test_cube = cube[1:-1:2, 1:-1:2, 1:-1:2, :]
    #sparse_interp = RegularGridInterpolator(sparse_points, sparse_cube)

    lnpred = []
    for x in X.T:
        lnpred += [linterp_at(x, lncube, itype=itype, points=points)]

    
    pred = np.exp(np.array(lnpred))
    good = np.isfinite(pred[:, 100])


    bounds = {}
    bounds["Warm Dwarfs"] = {'logt': (np.log10(4251.0), np.log10(6501)),
                             'logg': (3.49, 5.01),
                             'feh': (-2.01, 0.51)}

    
    sel = good & select(features, bounds["Warm Dwarfs"])
    delta = (pred[sel] - output[sel]) / output[sel]
    delta[np.isinf(delta)] = np.nan
    mu = np.nanmean(delta, axis=0)
    sigma = np.nanstd(delta, axis=0)

    fig, ax = pl.subplots()
    ax.plot(wave, sigma * 100, label="$\sigma_\chi$")
    ax.plot(wave, mu * 100, label="$\\bar{{{}}}$".format("\chi"))
    ax.set_ylim(-5, 30)
    ax.set_xlim(0.25e4, 1.0e4)
    ax.set_xlabel("wave ($\AA$)")
    ax.set_ylabel("Fractional error (%)")
    ax.set_title("Warm Dwarfs, {} interpolation".format(itype))
    ax.legend(frameon=False)

    w = np.argmin(np.abs(wave - 3000))
    sfig, sax = pl.subplots()
    cb = sax.scatter(features[sel]['logt'], 100 * delta[:, w], c=features[sel]["feh"],
                     marker='o', cmap="viridis")
    cc = pl.colorbar(cb, ax=sax, label="[Fe/H]")
    sax.set_xlabel("logt")
    sax.set_ylabel("Fractional error (%)")
    sax.set_title("$\lambda={:5.1f}$".format(wave[w]))

    pl.show()
