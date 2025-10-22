"""Gaussian deconvolution."""

import numpy as np
from math import sin, cos, atan, sqrt, pi
from numba import njit, float64, int64, types


@njit(
    types.Tuple((float64, float64, float64, int64))(
        float64, float64, float64, float64, float64, float64
    )
)
def deconv(fmaj, fmin, fpa, cmaj, cmin, cpa):
    """Deconvolve a Gaussian "beam" from a Gaussian component.

    When we fit an elliptical Gaussian to a point in our image, we are
    actually fitting to a convolution of the physical shape of the source with
    the beam pattern of our instrument. This results in the fmaj/fmin/fpa
    arguments to this function.

    Since the shape of the (clean) beam (arguments cmaj/cmin/cpa) is known, we
    can deconvolve it from the fitted parameters to get the "real" underlying
    physical source shape, which is what this function returns.

    Parameters
    ----------
    fmaj : float
        Fitted major axis (pixels).
    fmin : float
        Fitted minor axis (pixels).
    fpa : float
        Fitted position angle of the major axis (degrees).
    cmaj : float
        Clean beam major axis (pixels).
    cmin : float
        Clean beam minor axis (pixels).
    cpa : float
        Clean beam position angle of the major axis (degrees).

    Returns
    -------
    rmaj : float
        real major axis in pixels
    rmin : float
        real minor axis in pixels
    rpa : float
        real position angle of the major axis in degress
    ierr : int
        number of components which failed to deconvolve

    Notes
    -----
    Instead of fmaj, fmin, cmaj and cmin all in pixels, one could use any
    arbitrary unit of sky angular distance, such as arcseconds or radians.
    The first two elements of the returned tuple would then have that same
    unit, while the third element (the position angle) would still be in
    degrees.

    """
    HALF_RAD = 90.0 / pi
    cmaj2 = cmaj * cmaj
    cmin2 = cmin * cmin
    fmaj2 = fmaj * fmaj
    fmin2 = fmin * fmin
    theta = (fpa - cpa) / HALF_RAD
    det = ((fmaj2 + fmin2) - (cmaj2 + cmin2)) / 2.0
    rhoc = (fmaj2 - fmin2) * cos(theta) - (cmaj2 - cmin2)
    sigic2 = 0.0
    rhoa = 0.0
    ierr = 0

    if abs(rhoc) > 0.0:
        sigic2 = atan((fmaj2 - fmin2) * sin(theta) / rhoc)
        rhoa = ((cmaj2 - cmin2) - (fmaj2 - fmin2) * cos(theta)) / (
            2.0 * cos(sigic2)
        )

    rpa = sigic2 * HALF_RAD + cpa
    rmaj = det - rhoa
    rmin = det + rhoa

    if rmaj < 0:
        ierr += 1
        rmaj = 0
    if rmin < 0:
        ierr += 1
        rmin = 0

    rmaj = sqrt(rmaj)
    rmin = sqrt(rmin)
    if rmaj < rmin:
        rmaj, rmin = rmin, rmaj
        rpa += 90

    rpa = (rpa + 900) % 180
    if not abs(rmaj):
        rpa = 0.0
    elif not abs(rmin) and (45.0 < abs(rpa - fpa) < 135.0):
        rpa = (rpa + 450.0) % 180.0

    return rmaj, rmin, rpa, ierr


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
@njit
def covariance_matrix(sigma_maj, sigma_min, theta):
    """
    Build covariance matrix for an anisotropic Gaussian.

    Parameters
    ----------
    sigma_maj, sigma_min : float
        Standard deviations along major and minor axes (same units as x,
        y grid).
    theta : float
        Position angle in radians, measured from +Y toward -X (north through east).
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[-s, -c], [c, -s]])

    # Diagonal covariance matrix of σ²
    D = np.empty((2, 2))
    D[0, 0] = sigma_maj * sigma_maj
    D[0, 1] = 0.0
    D[1, 0] = 0.0
    D[1, 1] = sigma_min * sigma_min
    return R @ D @ R.T
