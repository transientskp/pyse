"""Gaussian deconvolution."""

import numpy as np
from math import sin, cos, atan, sqrt, pi

from numba import njit, float64, int64, types, f8, b1


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
        Position angle in radians, measured CCW from +Y toward -X (north
        through east).

    Returns
    -------
    Sigma : (2,2) ndarray
        Covariance matrix.
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


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
@njit
def J_S_from_stddevs_and_pa(sigma_maj, sigma_min, theta):
    """
    Jacobian d(sxx, syy, sxy)/d(sigma_maj, sigma_min, theta).

    Parameters:
    ----------
      sigma_maj, sigma_min : float
          standard-deviations along the elliptical axes
      theta : float
          angle in radians, CCW from +Y axis

    Returns:
      J : (3,3) ndarray
          rows [dsxx/d*, dsyy/d*, dsxy/d*] and columns corresponding to [
          sigma_maj, sigma_min, theta].
    """
    phi = theta + np.pi / 2.0  # convert to math convention (CCW from +x)

    c = np.cos(phi)
    s = np.sin(phi)
    c2 = c * c
    s2 = s * s
    sc = s * c
    cos2 = np.cos(2.0 * phi)

    a = sigma_maj
    b = sigma_min

    # components
    dsxx_da = 2.0 * a * c2
    dsxx_db = 2.0 * b * s2
    dsxx_dphi = 2.0 * (b * b - a * a) * sc  # = (b^2 - a^2) * sin(2phi)

    dsyy_da = 2.0 * a * s2
    dsyy_db = 2.0 * b * c2
    dsyy_dphi = 2.0 * (a * a - b * b) * sc  # = (a^2 - b^2) * sin(2phi)

    dsxy_da = 2.0 * a * sc
    dsxy_db = -2.0 * b * sc
    dsxy_dphi = (a * a - b * b) * cos2

    J = np.array(
        [
            [dsxx_da, dsxx_db, dsxx_dphi],
            [dsyy_da, dsyy_db, dsyy_dphi],
            [dsxy_da, dsxy_db, dsxy_dphi],
        ],
        dtype=np.float64,
    )

    return J


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
@njit
def cov_p_to_cov_S(C_p, sigma_maj, sigma_min, theta):
    """
    Propagate covariance C_p (3x3) on parameters p=(sigma_maj, sigma_min,
    theta)
    to covariance on S = (sxx, syy, sxy) using analytic Jacobian.

    Parameters
    ----------
      C_p : (3,3) ndarray  (covariance in order [sigma_maj, sigma_min, theta])
      sigma_maj, sigma_min: float
      theta: float (radians)

    Returns
    -------
      C_S : (3,3) ndarray (covariance on [sxx, syy, sxy])
    """
    J = J_S_from_stddevs_and_pa(sigma_maj, sigma_min, theta)
    C_S = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                for l in range(3):
                    acc += J[i, k] * C_p[k, l] * J[j, l]
            C_S[i, j] = acc
    return C_S


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
@njit(types.Tuple((f8, f8, f8, f8[:, ::1], b1))(f8, f8, f8))
def sigma_to_stddevs_pa_and_jacobian(sxx, syy, sxy):
    """
    From covariance elements sxx, syy, sxy return (sigma_maj, sigma_min, theta,
    J, ok)
    where sigma_maj>=sigma_min are stddevs along the elliptical axes and
    theta is angle (radians) CCW from +y.
    J is the 3x3 Jacobian d[a,b,theta]/d[sxx,syy,sxy] (rows outputs,
    cols inputs).

    Parameters
    ----------
    sxx, syy, sxy : float
        Covariance matrix elements: if sigma_maj is major axis stddev,
        sigma_min is minor axis stddev, and theta the position angle (CCW
        from +Y), then
            S = [[sxx, sxy],
                 [sxy, syy]] = R @ [[sigma_maj^2, 0],
                                    [0, sigma_min^2]] @ R.T
        where R = [[-sin(theta), -cos(theta)],
                   [ cos(theta), -sin(theta)]]
        i.e. sxx = sigma_maj^2 sin^2(theta) + sigma_min^2 cos^2(theta)
             syy = sigma_maj^2 cos^2(theta) + sigma_min^2 sin^2(theta)
             sxy = -(sigma_maj^2 - sigma_min^2) sin(theta) cos(theta)
    Returns
    -------
    sigma_maj : float
        Major axis standard deviation (pixels)
    sigma_min : float
        Minor axis standard deviation (pixels)
    theta : float
        Position angle in radians, CCW from +Y axis
    J : ndarray, shape (3,3)
        Jacobian d[a,b,phi]/d[sxx,syy,sxy]
    ok : bool
        True if conversion succeeded (Sigma positive definite), else False.
    """
    # helpers
    m = 0.5 * (sxx + syy)
    d = 0.5 * (sxx - syy)
    T = np.hypot(d, sxy)  # sqrt(d^2 + sxy^2) robustly
    lam1 = m + T
    lam2 = m - T

    # check positivity
    if lam1 <= 0 or lam2 <= 0:
        return 0.0, 0.0, 0.0, np.zeros((3, 3)), False

    a = np.sqrt(lam1)
    b = np.sqrt(lam2)

    # handle very small T robustly — fall back to tiny finite differences if necessary
    if T < 1e-14:
        # numeric fallback: small finite-difference Jacobian
        base_phi = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        base = np.array([a, b, base_phi])
        J = np.zeros((3, 3))
        delta = 1e-8 * max(1.0, abs(sxx) + abs(syy) + abs(sxy))
        for j, dvec in enumerate(
            [(delta, 0, 0), (0, delta, 0), (0, 0, delta)]
        ):
            a2, b2, phi2, _, ok2 = sigma_to_stddevs_pa_and_jacobian(
                sxx + dvec[0], syy + dvec[1], sxy + dvec[2]
            )
            J[:, j] = (np.array([a2, b2, phi2]) - base) / delta
        return a, b, base_phi, J, True

    # analytical derivatives
    # dlambda1/ds*
    dl1_dsxx = 0.5 + d / (2.0 * T)
    dl1_dsyy = 0.5 - d / (2.0 * T)
    dl1_dsxy = sxy / T

    dl2_dsxx = 0.5 - d / (2.0 * T)
    dl2_dsyy = 0.5 + d / (2.0 * T)
    dl2_dsxy = -sxy / T

    da_dsxx = dl1_dsxx / (2.0 * a)
    da_dsyy = dl1_dsyy / (2.0 * a)
    da_dsxy = dl1_dsxy / (2.0 * a)

    db_dsxx = dl2_dsxx / (2.0 * b)
    db_dsyy = dl2_dsyy / (2.0 * b)
    db_dsxy = dl2_dsxy / (2.0 * b)

    # phi derivatives: phi = 0.5 * atan2(N, D) with N = 2*sxy, D = sxx - syy
    N = 2.0 * sxy
    D = sxx - syy
    Q = N * N + D * D
    dphi_dsxx = -sxy / Q
    dphi_dsyy = +sxy / Q
    dphi_dsxy = D / Q

    J = np.array(
        [
            [da_dsxx, da_dsyy, da_dsxy],
            [db_dsxx, db_dsyy, db_dsxy],
            [dphi_dsxx, dphi_dsyy, dphi_dsxy],
        ],
        dtype=np.float64,
    )

    sigma_maj = a
    sigma_min = b

    phi = 0.5 * np.arctan2(N, D) + np.pi / 2
    theta = phi

    return sigma_maj, sigma_min, theta, J, True


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
@njit
def cov_S_to_cov_r(S_dec, C_S_dec):
    """
    Convert covariance on S = [sxx, syy, sxy] to covariance on
    r = [a_dec, b_dec, phi_dec] using analytic Jacobian.

    Parameters
    ----------
    S_dec : array_like, shape (3,)
        [sxx, syy, sxy] of the deconvolved covariance matrix (pixels^2)
    C_S_dec : ndarray, shape (3,3)
        Covariance matrix of S_dec (units pixels^4)

    Returns
    -------
    r : ndarray shape (3,)
        (a_dec, b_dec, phi_dec) where phi_dec is radians (CCW from +x)
    C_r : ndarray shape (3,3)
        Covariance matrix on (a_dec, b_dec, phi_dec)
    ok : bool
        True if conversion succeeded (Sigma_dec positive definite), else False.
    """
    sxx, syy, sxy = S_dec
    a_dec, b_dec, phi_dec, J, ok = sigma_to_stddevs_pa_and_jacobian(
        sxx, syy, sxy
    )
    if not ok:
        C_r = np.full((3, 3), np.nan)
        return np.array([a_dec, b_dec, phi_dec]), C_r, False

    C_r = J @ C_S_dec @ J.T
    return np.array([a_dec, b_dec, phi_dec]), C_r, True
