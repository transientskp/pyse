"""
Source fitting routines.
"""

import math

import numpy as np
import scipy.optimize

from .gaussian import gaussian, jac_gaussian
from .stats import indep_pixels
from sourcefinder.deconv import deconv
from numba import guvectorize, float64, float32, int32, njit
from sourcefinder.utils import newton_raphson_root_finder

FIT_PARAMS = ('peak', 'xbar', 'ybar', 'semimajor', 'semiminor', 'theta')


@njit
def find_true_peak(peak, T, epsilon, msq, maxpix):
    """
    This function represents equation 2.67 from Spreeuw's thesis.

    """
    return np.log(peak / maxpix) - (epsilon / msq) * (1 + np.log(T / peak) /
                                                      (peak / T - 1))


def moments(data, fudge_max_pix_factor, beamsize, threshold=0):
    """
    Calculate source positional values using moments.

    The first moment of the distribution is the barycenter of an ellipse.
    The second moments are used to estimate the rotation angle and the
    length of the axes.

    Parameters
    ----------
    data : np.ma.MaskedArray or np.ndarray
        Actual 2D image data.
    fudge_max_pix_factor : float
        Correct for the underestimation of the peak by taking the maximum
        pixel value.
    beamsize : float
        The FWHM size of the clean beam.
    threshold : float, default: 0
        Source parameters like the semimajor and semiminor axes derived
        from moments can be underestimated if one does not take account of
        the threshold that was used to segment the source islands.

    Returns
    -------
    dict
        Dictionary containing peak, total, x barycenter, y barycenter,
        semimajor axis, semiminor axis, and theta.

    Raises
    ------
    exceptions.ValueError
        If input contains NaN values.
    """
    total = data.sum()
    x, y = np.indices(data.shape)
    xbar = float((x * data).sum() / total)
    ybar = float((y * data).sum() / total)
    xxbar = (x * x * data).sum() / total - xbar ** 2
    yybar = (y * y * data).sum() / total - ybar ** 2
    xybar = (x * y * data).sum() / total - xbar * ybar

    working1 = (xxbar + yybar) / 2.0
    working2 = math.sqrt(((xxbar - yybar) / 2) ** 2 + xybar ** 2)

    if (np.isnan(xbar) or np.isnan(ybar) or np.isnan(working1)
            or np.isnan(working2)):
        raise ValueError("Unable to estimate Gauss shape")

    maxpos = np.unravel_index(np.abs(data).argmax(), data.shape)

    # Are we fitting a -ve or +ve Gaussian?
    if data.mean() >= 0:
        peak = data[maxpos]
        # The peak is always underestimated when you take the highest or lowest
        # - for images other than Stokes I, where that may apply - pixel.
        peak *= fudge_max_pix_factor
    else:
        peak = data.min()

    # Some problems arise with the sqrt of (working1-working2) when they are
    # equal, this happens with islands that have a thickness of only one pixel
    # in at least one dimension.  Due to rounding errors this difference
    # becomes negative--->math domain error in sqrt.
    if len(data.nonzero()[0]) > 1:
        semimajor_tmp = (working1 + working2) * 2.0 * math.log(2.0)
        semiminor_tmp = (working1 - working2) * 2.0 * math.log(2.0)

        # Theta is not affected by the cut-off at the threshold, see Spreeuw's
        # thesis (2010), page 45, so we can easily compute the position angle
        # first.
        if abs(semimajor_tmp - semiminor_tmp) < 0.01:
            # short circuit!
            theta = 0.
        else:
            if xxbar != yybar:
                theta = math.atan(2. * xybar / (xxbar - yybar)) / 2.
            else:
                theta = np.sign(xybar) * math.pi / 4.0

            if theta * xybar > 0.:
                if theta < 0.:
                    theta += math.pi / 2.0
                else:
                    theta -= math.pi / 2.0

        if semiminor_tmp > 0:
            rounded_barycenter = int(round(xbar)), int(round(ybar))
            # basevalue and basepos will be needed for "tweaked moments".
            try:
                if not data.mask[rounded_barycenter]:
                    basepos = rounded_barycenter
                else:
                    basepos = maxpos
            except IndexError:
                basepos = maxpos
            except AttributeError:
                # If the island is not masked at all, we can safely set basepos to
                # the rounded barycenter position.
                basepos = rounded_barycenter
            basevalue = data[basepos]

            if np.sign(threshold) == np.sign(basevalue):
                # Implementation of "tweaked moments", equation 2.67 from
                # Spreeuw's thesis. In that formula the "base position" was the
                # maximum pixel position, though. Here that is the rounded
                # barycenter position, unless it's masked. If it's masked, it
                # will be the maximum pixel position.
                deltax, deltay = xbar - basepos[0], ybar - basepos[1]

                epsilon = np.log(2.) * ((np.cos(theta) * deltax +
                                         np.sin(theta) * deltay) ** 2
                                        + (np.cos(theta) * deltay -
                                           np.sin(theta) * deltax) ** 2
                                        * semiminor_tmp / semimajor_tmp)

                # Set limits for the root finder similar to the bounds for
                # Gaussian fits.
                if basevalue > 0:
                    low_bound = 0.5 * basevalue
                    upp_bound = 1.5 * basevalue
                else:
                    low_bound = 1.5 * basevalue
                    upp_bound = 0.5 * basevalue

                # The number of iterations used for the root finder is also
                # returned, but not used here.
                peak, _ = newton_raphson_root_finder(find_true_peak, basevalue,
                                                     low_bound, upp_bound,
                                                     1e-8, 100,
                                                     threshold, epsilon,
                                                     semiminor_tmp, basevalue)
                # The corrections below for the semi-major and semi-minor axes are
                # to compensate for the underestimate of these quantities
                # due to the cutoff at the threshold.
                ratio = threshold / peak
                semimajor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
                semiminor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))

        elif np.sign(threshold) == np.sign(peak):
            # A semi-minor axis exactly zero gives all kinds of problems.
            # For instance wrt conversion to celestial coordinates.
            # This is a quick fix.
            ratio = threshold / peak
            semimajor_tmp /= (1.0 + np.log(ratio) * ratio / (1.0 - ratio))
            semiminor_tmp = beamsize ** 2 / (np.pi ** 2 * semimajor_tmp)

        semimajor = math.sqrt(semimajor_tmp)
        semiminor = math.sqrt(semiminor_tmp)

    else:
        # This is the case when the island (or more likely subisland) has
        # a size of only one pixel.
        theta = 0.
        semiminor = np.sqrt(beamsize / np.pi)
        semimajor = np.sqrt(beamsize / np.pi)

    # NB: a dict should give us a bit more flexibility about arguments;
    # however, all those here are ***REQUIRED***.
    return {
        "peak": peak,
        "flux": total,
        "xbar": xbar,
        "ybar": ybar,
        "semimajor": semimajor,
        "semiminor": semiminor,
        "theta": theta
    }


@guvectorize([(float32[:], float32[:], int32[:], int32[:], int32[:], int32,
               int32, float32, float32, int32[:], float32, float64, float64,
               float64[:], float64, float64[:], float64, float64, float32[:, :],
               float32[:, :], float32[:, :], float32[:, :], float32[:],
               float32[:], float32[:])],
             ('(n), (n), (m), (n), (n), (), (), (), (), (m), (), (), (), (k),' +
              '(), (m), (), (), (q, r), (q, r), (l, p) -> (l, p), (), (), ()'),
             nopython=True)
def moments_enhanced(source_island, noise_island, chunkpos, posx, posy,
                     min_width, no_pixels, threshold, noise, maxpos, maxi,
                     fudge_max_pix_factor, max_pix_variance_factor, beam,
                     beamsize, correlation_lengths, clean_bias_error,
                     frac_flux_cal_error, Gaussian_islands_map,
                     Gaussian_residuals_map, dummy, computed_moments,
                     significance, chisq, reduced_chisq):
    """
    Calculate source properties using moments.

    Vectorized using the `guvectorize` decorator. Also calculates the 
    signal-to-noise ratio of detections, chi-squared and reduced 
    chi-squared statistics, and fills in maps with Gaussian islands and 
    Gaussian residuals. Uses the first moments of the distribution to 
    determine the barycenter of an ellipse, while the second moments estimate 
    rotation angle and axis lengths.

    Parameters
    ----------
    source_island : np.ndarray
        Selected from the 2D image data by taking pixels above the analysis 
        threshold, with its peak above the detection threshold. Flattened to 
        a 1D ndarray. Units: spectral brightness, typically Jy/beam.

    noise_island : np.ndarray
        Pixel values selected from the 2D RMS noise map at the positions of 
        the island. Flattened to a 1D ndarray. Units: spectral brightness, 
        typically Jy/beam.

    chunkpos : np.ndarray
        Index array of length 2 denoting the position of the top-left corner 
        of the rectangular slice encompassing the island, relative to the 
        top-left corner of the image.

    posx : np.ndarray
        Row indices of the pixels in `source_island` relative to the top-left 
        corner of the rectangular slice encompassing the island. The top-left 
        corner corresponds to `posx = 0`. Derived from the 2D image data.

    posy : np.ndarray
        Column indices of the pixels in `source_island` relative to the 
        top-left corner of the rectangular slice encompassing the island. 
        The top-left corner corresponds to `posy = 0`. Derived from the 
        2D image data.

    min_width : int
        Minimum width (in pixels) of the island, derived as the lesser of its 
        maximum width along the x and y axes.

    no_pixels : int
        Number of pixels that constitute the island.

    threshold : float
        Threshold used for segmenting source islands, which can affect 
        parameters like semimajor and semiminor axes. A higher threshold may 
        lead to a larger underestimate of the Gaussian axes. If the analysis 
        threshold is known, this underestimate can be corrected. 
        Units: spectral brightness, typically Jy/beam.

    noise : float
        Local noise, i.e., the standard deviation of the background pixel 
        values at the position of the island's peak pixel value. 
        Units: spectral brightness, typically Jy/beam.

    maxpos : np.ndarray with int32 as dtype and length 2
        The position of the maximum pixel value within the island, relative
        to the top-left corner of the rectangular slice encompassing the
        island. Units: pixels.

    maxi : float
        Peak pixel value within the island. Units: spectral brightness, 
        typically Jy/beam. To clarify: source_island[maxpos] == maxi.

    fudge_max_pix_factor : float
        Correction factor for underestimation of the peak by considering the 
        maximum pixel value.

    max_pix_variance_factor : float
        Additional variance induced by the maximum pixel method, beyond the 
        background noise.

    beam : np.ndarray
        Array of three floats: [semimajor axis, semiminor axis, theta]. 
        Units: pixels.

    beamsize : float
        FWHM size of the clean beam. Units: pixels.

    correlation_lengths : np.ndarray
        Array of two floats describing distances along the semi-major and 
        semi-minor axes of the clean beam beyond which noise is assumed 
        uncorrelated. 
        Units: pixels.
        Some background: Aperture synthesis imaging yields noise that is 
        partially correlated over the entire image. This has a considerable 
        effect on error estimates. All noise within the correlation length 
        is approximated as completely correlated, while noise beyond is 
        considered uncorrelated.

    clean_bias_error : float
        Extra source of error based on Condon (PASP 109, 166, 1997) formulae.

    frac_flux_cal_error : float
        Extra source of error based on Condon (PASP 109, 166, 1997) formulae.

    Gaussian_islands_map : np.ndarray
        Initially a 2D np.float32 array filled with zeros, same shape as the 
        astronomical image being processed. Computed Gaussian islands are 
        added to this array at pixel positions above the analysis threshold.

    Gaussian_residuals_map : np.ndarray
        Initially a 2D np.float32 array filled with zeros, same shape as 
        `Gaussian_islands_map`. Residuals are computed by subtracting 
        `Gaussian_islands_map` from the input image data.

    dummy : np.ndarray
        Empty array matching the shape of `computed_moments`, required due 
        to limitations in `guvectorize`.

    computed_moments : np.ndarray
        Array of shape (10, 2) containing moments such as peak spectral 
        brightness (Jy/beam), flux density (Jy), barycenter (pixels), 
        semi-major and semi-minor axes (pixels), and position angle 
        (radians), along with corresponding errors.

    significance : float
        The significance of a detection is defined as the maximum 
        signal-to-noise ratio across the island. Often this will be the ratio 
        of the maximum pixel value within the source island divided by the 
        noise at that position. But for extended sources, the noise can 
        perhaps decrease away from the position of the peak spectral 
        brightness more steeply than the source spectral brightness, and the 
        maximum signal-to-noise ratio can be found at a different position.

    chisq : float
        Chi-squared statistic indicating goodness-of-fit, derived in the same 
        way as in the `fitting.goodness_of_fit` method.

    reduced_chisq : float
        Reduced chi-squared statistic indicating goodness-of-fit, derived in 
        the same way as in the `fitting.goodness_of_fit` method.

    Returns
    -------
    None
        Outputs are written to `Gaussian_islands_map`, `Gaussian_residuals_map`, 
        `computed_moments`, `significance`, `chisq`, and `reduced_chisq`.

    Raises
    ------
    ValueError
        If input contains NaN values.

    """

    # Not every island has the same size. The number of columns of the array
    # containing all islands is equal to the maximum number of pxiels over
    # all islands. This containing array was created by np.empty, so better
    # dump the redundant elements that have undetermined values.
    source_island = source_island[:no_pixels]
    noise_island = noise_island[:no_pixels]
    # The significance of a source detection is determined in this way.
    significance[0] = (source_island / noise_island).max()

    total = source_island.sum()
    xbar, ybar, xxbar, yybar, xybar = 0, 0, 0, 0, 0

    for index in range(no_pixels):
        i = posx[index]
        j = posy[index]
        xbar += i * source_island[index]
        ybar += j * source_island[index]
        xxbar += i * i * source_island[index]
        yybar += j * j * source_island[index]
        xybar += i * j * source_island[index]

    xbar /= total
    ybar /= total
    xxbar /= total
    xxbar -= xbar ** 2
    yybar /= total
    yybar -= ybar ** 2
    xybar /= total
    xybar -= xbar * ybar

    working1 = (xxbar + yybar) / 2.0
    working2 = math.sqrt(((xxbar - yybar) / 2) ** 2 + xybar ** 2)

    if (np.isnan(xbar) or np.isnan(ybar) or np.isnan(working1)
            or np.isnan(working2)):
        raise ValueError("Unable to estimate Gauss shape")

    # Are we fitting a -ve or +ve Gaussian?
    if source_island.mean() >= 0:
        # The peak is always underestimated when you take the highest pixel.
        peak = maxi * fudge_max_pix_factor
    else:
        peak = source_island.min()

    # We need this width to determine Gaussian shape parameters in a meaningful
    # way.
    if min_width > 2:
        # The idea of the try except here is that, even though we require a
        # minimum width of the island, there may still be occasions where
        # working2 can be slightly higher than working1, perhaps due to rounding
        # errors.
        semimajor_tmp = (working1 + working2) * 2.0 * math.log(2.0)
        semiminor_tmp = (working1 - working2) * 2.0 * math.log(2.0)

        # Theta is not affected by the cut-off at the threshold, see Spreeuw's
        # thesis (2010), page 45, so we can easily compute the position angle
        # first.
        if abs(semimajor_tmp - semiminor_tmp) < 0.01:
            # short circuit!
            theta = 0.
        else:
            if xxbar != yybar:
                theta = math.atan(2. * xybar / (xxbar - yybar)) / 2.
            else:
                theta = np.sign(xybar) * math.pi / 4.0

            if theta * xybar > 0.:
                if theta < 0.:
                    theta += math.pi / 2.0
                else:
                    theta -= math.pi / 2.0

        if semiminor_tmp > 0:
            # We will be extrapolating from a pixel centred position to the real
            # position of the Gaussian peak.
            rounded_barycenter = int(round(xbar)), int(round(ybar))
            # First we try the pixel centred position closest to the barycenter
            # position to extrapolate from.
            if rounded_barycenter[0] in posx and rounded_barycenter[1] in posy:
                basepos = rounded_barycenter
                # We need to loop over all possible positions to find the
                # source_island index corresponding to the rounded barycenter
                # position.
                for i in (posx==rounded_barycenter[0]).nonzero()[0]:
                    if posy[i] == rounded_barycenter[1]:
                        basevalue = source_island[i]
                        break
            else:
                # The rounded barycenter position is not in source_island, so we
                # revert to the maximum pixel position, which is always included.
                basepos = maxpos[0] - chunkpos[0], maxpos[1] - chunkpos[1]
                # In this case we do not need to figure out the source_island
                # index corresponding to the maximum pixel position, because
                # maxi, the maximum pixel value has already been provided, as an
                # argument to this function.
                basevalue = maxi

            if np.sign(threshold) == np.sign(basevalue):
                # Implementation of "tweaked moments", equation 2.67 from
                # Spreeuw's thesis. In that formula the "base position" was the
                # maximum pixel position, though. Here that is the rounded
                # barycenter position, unless it's masked. If it's masked, it
                # will be the maximum pixel position.
                deltax, deltay = xbar - basepos[0], ybar - basepos[1]

                epsilon = np.log(2.) * ((np.cos(theta) * deltax +
                                         np.sin(theta) * deltay) ** 2
                                        + (np.cos(theta) * deltay -
                                           np.sin(theta) * deltax) ** 2
                                        * semiminor_tmp / semimajor_tmp)

                # Set limits for the root finder similar to the bounds for
                # Gaussian fits.
                if basevalue > 0:
                    low_bound = 0.5 * basevalue
                    upp_bound = 1.5 * basevalue
                else:
                    low_bound = 1.5 * basevalue
                    upp_bound = 0.5 * basevalue

                # The number of iterations used for the root finder is also
                # returned, but not used here.
                peak, _ = newton_raphson_root_finder(find_true_peak, basevalue,
                                                     low_bound, upp_bound,
                                                     1e-8, 100,
                                                     threshold, epsilon,
                                                     semiminor_tmp, basevalue)
                # The corrections below for the semi-major and semi-minor axes are
                # to compensate for the underestimate of these quantities
                # due to the cutoff at the threshold.
                ratio = threshold / peak
                semimajor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
                semiminor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))

        elif np.sign(threshold) == np.sign(peak):
            # A semi-minor axis exactly zero gives all kinds of problems.
            # For instance wrt conversion to celestial coordinates.
            # This is a quick fix.
            ratio = threshold / peak
            semimajor_tmp /= (1.0 + np.log(ratio) * ratio / (1.0 - ratio))
            semiminor_tmp = beamsize ** 2 / (np.pi ** 2 * semimajor_tmp)

        try:
            smaj = math.sqrt(semimajor_tmp)
            smin = math.sqrt(semiminor_tmp)
            if smin == 0:
                # A semi-minor axis exactly zero gives all kinds of problems.
                # For instance wrt conversion to celestial coordinates.
                # This is a quick fix.
                smin = beamsize / (np.pi * smaj)

        # In all cases where we hit a math domain error, we can use the clean
        # beam parameters to derive reasonable estimates for the Gaussian shape
        # parameters.
        except Exception:
            smaj = beam[0]
            smin = beam[1]
            theta = beam[2]
    else:
        # In the case that the island has insufficient width, we can also derive
        # reasonable estimates for Gaussian shape parameters using the clean
        # beam.
        smaj = beam[0]
        smin = beam[1]
        theta = beam[2]

    # Reconstruct the Gaussian profile we just derived at the pixel positions
    # of the island. Initialise a flat array to hold these values.
    Gaussian_island = np.empty(no_pixels, dtype=np.float32)
    # Likewise for the Gaussian residuals.
    Gaussian_residual = np.empty_like(Gaussian_island)

    # Compute the residuals based on the derived Gaussian parameters.
    for index in range(no_pixels):
        Gaussian_island[index] = peak * np.exp(-np.log(2) * (
            ((np.cos(theta) * (posx[index] - xbar)
             + np.sin(theta) * (posy[index] - ybar)) / smin) ** 2 +
            ((np.cos(theta) * (posy[index] - ybar)
             - np.sin(theta) * (posx[index] - xbar)) / smaj) ** 2))

        map_position = (posx[index] + chunkpos[0], posy[index] + chunkpos[1])
        Gaussian_islands_map[map_position] = Gaussian_island[index]

        Gaussian_residual[index] = source_island[index] - \
            Gaussian_island[index]
        Gaussian_residuals_map[map_position] = Gaussian_residual[index]

    # Copy code from goodness_of_fit. Here we are working with unmasked data,
    # i.e. the masks have already been applied.
    gauss_resid_normed = Gaussian_residual / noise_island
    chisq[0] = np.sum(gauss_resid_normed ** 2)
    n_indep_pix = indep_pixels(no_pixels, correlation_lengths)
    reduced_chisq[0] = chisq[0] / n_indep_pix

    #  Equivalent of param["flux"] = (np.pi * param["peak"] *
    #  param["semimajor"] * param["semiminor"] / beamsize) from extract.py.
    flux = np.pi * peak * smaj * smin / beamsize

    # Update xbar and ybar with the position of the upper left corner of the
    # chunk.
    xbar += chunkpos[0]
    ybar += chunkpos[1]

    # There is no point in proceeding with processing an image if any
    # detected peak spectral brightness is below zero since that implies
    # that part of detectionthresholdmap is below zero.
    if peak < 0:
        raise ValueError

    """Provide reasonable error estimates from the moments"""

    # The formulae below should give some reasonable estimate of the
    # errors from moments, should always be higher than the errors from
    # Gauss fitting.
    theta_B = correlation_lengths[0]
    theta_b = correlation_lengths[1]

    # This is eq. 2.81 from Spreeuw's thesis.
    rho_sq = ((16. * smaj * smin /
               (np.log(2.) * theta_B * theta_b * noise ** 2))
              * ((peak - threshold) /
                 (np.log(peak) - np.log(threshold))) ** 2)

    rho = np.sqrt(rho_sq)
    denom = np.sqrt(2. * np.log(2.)) * rho

    # Again, like above for the Condon formulae, we set the
    # positional variances to twice the theoretical values.
    error_par_major = 2. * smaj / denom
    error_par_minor = 2. * smin / denom

    # When these errors are converted to RA and Dec,
    # calibration uncertainties will have to be added,
    # like in formulae 27 of the NVSS paper.
    errorx = np.sqrt((error_par_major * np.sin(theta)) ** 2
                     + (error_par_minor * np.cos(theta)) ** 2)
    errory = np.sqrt((error_par_major * np.cos(theta)) ** 2
                     + (error_par_minor * np.sin(theta)) ** 2)

    # Note that we report errors in HWHM axes instead of FWHM axes
    # so the errors are half the errors of formula 29 of the NVSS paper.
    errorsmaj = np.sqrt(2) * smaj / rho
    errorsmin = np.sqrt(2) * smin / rho

    if smaj > smin:
        errortheta = 2.0 * (smaj * smin / (smaj ** 2 - smin ** 2)) / rho
    else:
        errortheta = np.pi
    if errortheta > np.pi:
        errortheta = np.pi

    # The peak from "moments" is just the value of the maximum pixel
    # times a correction, fudge_max_pix, for the fact that the
    # centre of the Gaussian is not at the centre of the pixel.
    # This correction is performed in fitting.py. The maximum pixel
    # method introduces a peak dependent error corresponding to the last
    # term in the expression below for errorpeaksq.
    # To this, we add, in quadrature, the errors corresponding
    # to the first and last term of the rhs of equation 37 of the
    # NVSS paper. The middle term in that equation 37 is heuristically
    # replaced by noise**2 since the threshold should not affect
    # the error from the (corrected) maximum pixel method,
    # while it is part of the expression for rho_sq above.
    # errorpeaksq = ((frac_flux_cal_error * peak) ** 2 +
    #                clean_bias_error ** 2 + noise ** 2 +
    #                utils.maximum_pixel_method_variance(
    #                    beam[0], beam[1], beam[2]) * peak ** 2)
    errorpeaksq = ((frac_flux_cal_error * peak) ** 2 +
                   clean_bias_error ** 2 + noise ** 2 +
                   max_pix_variance_factor * peak ** 2)
    errorpeak = np.sqrt(errorpeaksq)

    help1 = (errorsmaj / smaj) ** 2
    help2 = (errorsmin / smin) ** 2
    help3 = theta_B * theta_b / (4. * smaj * smin)
    errorflux = flux * np.sqrt(
        errorpeaksq / peak ** 2 + help3 * (help1 + help2))

    """Deconvolve from the clean beam"""

    # If the fitted axes are smaller than the clean beam
    # (=restoring beam) axes, the axes and position angle
    # can be deconvolved from it.
    fmaj = 2. * smaj
    fmajerror = 2. * errorsmaj
    fmin = 2. * smin
    fminerror = 2. * errorsmin
    fpa = np.degrees(theta)
    fpaerror = np.degrees(errortheta)
    cmaj = 2. * beam[0]
    cmin = 2. * beam[1]
    cpa = np.degrees(beam[2])

    rmaj, rmin, rpa, ierr = deconv(fmaj, fmin, fpa, cmaj, cmin, cpa)
    # This parameter gives the number of components that could not be
    # deconvolved, IERR from deconf.f.
    deconv_imposs = ierr
    # Now, figure out the error bars.
    if rmaj > 0:
        # In this case the deconvolved position angle is defined.
        # For convenience, we reset rpa to the interval [-90, 90].
        if rpa > 90:
            rpa = -np.mod(-rpa, 180.)
        theta_deconv = rpa

        # In the general case, where the restoring beam is elliptic,
        # calculating the error bars of the deconvolved position angle
        # is more complicated than in the NVSS case, where a circular
        # restoring beam was used.
        # In the NVSS case the error bars of the deconvolved angle are
        # equal to the fitted angle.
        rmaj1, rmin1, rpa1, ierr1 = deconv(
            fmaj, fmin, fpa + fpaerror, cmaj, cmin, cpa)
        if ierr1 < 2:
            if rpa1 > 90:
                rpa1 = -np.mod(-rpa1, 180.)
            rpaerror1 = np.abs(rpa1 - rpa)
            # An angle error can never be more than 90 degrees.
            if rpaerror1 > 90.:
                rpaerror1 = np.mod(-rpaerror1, 180.)
        else:
            rpaerror1 = np.nan
        rmaj2, rmin2, rpa2, ierr2 = deconv(
            fmaj, fmin, fpa - fpaerror, cmaj, cmin, cpa)
        if ierr2 < 2:
            if rpa2 > 90:
                rpa2 = -np.mod(-rpa2, 180.)
            rpaerror2 = np.abs(rpa2 - rpa)
            # An angle error can never be more than 90 degrees.
            if rpaerror2 > 90.:
                rpaerror2 = np.mod(-rpaerror2, 180.)
        else:
            rpaerror2 = np.nan
        if np.isnan(rpaerror1) or np.isnan(rpaerror2):
            theta_deconv_error = np.nansum(
                np.array([rpaerror1, rpaerror2]))
        else:
            theta_deconv_error = np.mean(
                np.array([rpaerror1, rpaerror2]))
        semimaj_deconv = rmaj / 2.
        rmaj3, rmin3, rpa3, ierr3 = deconv(
            fmaj + fmajerror, fmin, fpa, cmaj, cmin, cpa)
        # If rmaj>0, then rmaj3 should also be > 0,
        # if I am not mistaken, see the formulas at
        # the end of ch.2 of Spreeuw's Ph.D. thesis.
        if fmaj - fmajerror > fmin:
            rmaj4, rmin4, rpa4, ierr4 = deconv(
                fmaj - fmajerror, fmin, fpa, cmaj, cmin, cpa)
            if rmaj4 > 0:
                semimaj_deconv_error = np.mean(np.array(
                    [np.abs(rmaj3 - rmaj), np.abs(rmaj - rmaj4)]))
            else:
                semimaj_deconv_error = np.abs(rmaj3 - rmaj)
        else:
            rmin4, rmaj4, rpa4, ierr4 = deconv(
                fmin, fmaj - fmajerror, fpa, cmaj, cmin, cpa)
            if rmaj4 > 0:
                semimaj_deconv_error = np.mean(np.array(
                    [np.abs(rmaj3 - rmaj), np.abs(rmaj - rmaj4)]))
            else:
                semimaj_deconv_error = np.abs(rmaj3 - rmaj)
        if rmin > 0:
            semimin_deconv = rmin / 2.
            if fmin + fminerror < fmaj:
                rmaj5, rmin5, rpa5, ierr5 = deconv(
                    fmaj, fmin + fminerror, fpa, cmaj, cmin, cpa)
            else:
                rmin5, rmaj5, rpa5, ierr5 = deconv(
                    fmin + fminerror, fmaj, fpa, cmaj, cmin, cpa)
            # If rmin > 0, then rmin5 should also be > 0,
            # if I am not mistaken, see the formulas at
            # the end of ch.2 of Spreeuw's Ph.D. thesis.
            rmaj6, rmin6, rpa6, ierr6 = deconv(
                fmaj, fmin - fminerror, fpa, cmaj, cmin, cpa)
            if rmin6 > 0:
                semimin_deconv_error = np.mean(np.array(
                    [np.abs(rmin6 - rmin), np.abs(rmin5 - rmin)]))
            else:
                semimin_deconv_error = np.abs(rmin5 - rmin)
        else:
            semimin_deconv = np.nan
            semimin_deconv_error = np.nan
    else:
        semimaj_deconv = np.nan
        semimaj_deconv_error = np.nan
        semimin_deconv = np.nan
        semimin_deconv_error = np.nan
        theta_deconv = np.nan
        theta_deconv_error = np.nan

    computed_moments[0, :] = np.array([peak, flux, xbar, ybar, smaj,
                                       smin, theta, semimaj_deconv,
                                       semimin_deconv, theta_deconv])
    computed_moments[1, :] = np.array([errorpeak, errorflux, errorx,
                                       errory, errorsmaj, errorsmin,
                                       errortheta, semimaj_deconv_error,
                                       semimin_deconv_error,
                                       theta_deconv_error])


def fitgaussian(pixels, params, fixed=None, max_nfev=None, bounds={}):
    """
    Calculate source positional values by fitting a 2D Gaussian.

    Parameters
    ----------
    pixels : np.ma.MaskedArray
        Pixel values (with bad pixels masked)
    params : dict
        Initial fit parameters (possibly estimated using the moments() function)
    fixed : dict, default: None
        Parameters & their values to be kept frozen (ie, not fitted)
    max_nfev : int, default: None
        Maximum number of calls to the error function
    bounds : dict, default: {}
        Can be a dict such as the extract.ParamSet().bounds attribute
        generated by the extract.ParamSet().compute_bounds method, but any
        dict with keys from FIT_PARAMS and (lower_bound, upper_bound, bool)
        tuples as values will do. The boolean argument accommodates for
        loosening a bound when a fit becomes unfeasible because of the bound,
        see the scipy.optimize.Bounds documentation and source code for
        background.

    Returns
    -------
    dict
        peak, total, x barycenter, y barycenter, semimajor, semiminor,
        theta (radians)

    Raises
    ------
    exceptions.ValueError
        In case of a bad fit.

    Notes
    -----
    Perform a least squares fit to an elliptical Gaussian.

    If a dict called fixed is passed in, then parameters specified within the
    dict with the same keys as in FIT_PARAMS will be "locked" in the fitting
    process.
    """
    fixed = fixed or {}

    def wipe_out_fixed(some_dict):
        wiped_out = map(lambda key: (key, some_dict[key]),
                        filter(lambda key: key not in fixed.keys(),
                               iter(FIT_PARAMS)))
        return dict(wiped_out)

    # Collect necessary values from parameter dict; only those which aren't
    # fixed.
    initial = []
    # To avoid fits from derailing, it helps to set some reasonable bounds on
    # the fitting results.
    for param in FIT_PARAMS:
        if param not in fixed:
            if hasattr(params[param], "value"):
                initial.append(params[param].value)
            else:
                initial.append(params[param])

    def residuals(parameters):
        """
        Error function to be used in chi-squared fitting.

        Parameters
        ----------
        parameters : np.ndarray
            Fitting parameters.

        Returns
        -------
        np.ndarray
            1d-array of difference between estimated Gaussian function and the
            actual pixels. (pixel_resids is a 2d-array, but the .compressed()
            makes it 1d.)
        """
        paramlist = list(parameters)
        gaussian_args = []
        for parameter in FIT_PARAMS:
            if parameter in fixed:
                gaussian_args.append(fixed[parameter])
            else:
                gaussian_args.append(paramlist.pop(0))

        # gaussian() returns a function which takes arguments x, y and returns
        # a Gaussian with parameters gaussian_args evaluated at that point.
        g = gaussian(*gaussian_args)

        # The .compressed() below is essential so the Gaussian fit will not
        # take account of the masked values (=below threshold) at the edges
        # and corners of pixels (=(masked) array, so rectangular).
        pixel_resids = np.ma.MaskedArray(
            data=np.fromfunction(g, pixels.shape) - pixels,
            mask=pixels.mask)

        return pixel_resids.compressed()

    def jacobian_values(parameters):
        """
        The Jacobian of an anisotropic 2D Gaussian at the pixel positions.

        Parameters
        ----------
        parameters : np.ndarray
            Fitting parameters.

        Returns
        -------
        np.ndarray
            2D-array with values of the partial derivatives of the
            2D anisotropic Gaussian along its six parameters. These
            values are evaluated across the unmasked pixel positions of
            the island that constitutes the source. The number of rows
            equals the number of pixels of the flattened unmasked island.
            The number of columns equals the number of partial
            derivatives of the Gaussian (=6). For fixed Gaussian
            parameters the Jacobian component is obviously zero, so that
            results in a column of only zeroes.

        """
        paramlist = list(parameters)
        gaussian_args = []
        for parameter in FIT_PARAMS:
            if parameter in fixed:
                gaussian_args.append(fixed[parameter])
            else:
                gaussian_args.append(paramlist.pop(0))

        # jac is a list of six functions corresponding to the six partial
        # derivatives of the 2D anisotropic Gaussian profile.
        jac = jac_gaussian(gaussian_args)

        # From the six functions in jac, wipe out the ones that
        # correspond to fixed parameters, must be in sync with initial.
        jac_filtered = wipe_out_fixed(jac)

        jac_values = [np.ma.MaskedArray(
            data=np.fromfunction(jac_filtered[key], pixels.shape),
            mask=pixels.mask).compressed() for key in jac_filtered]

        return np.array(jac_values).T

    # maxfev=0, the default, corresponds to 200*(N+1) (NB, not 100*(N+1) as
    # the scipy docs state!) function evaluations, where N is the number of
    # parameters in the solution.
    # Convergence tolerances xtol and ftol established by experiment on images
    # from Paul Hancock's simulations.
    # April 2024 update.
    # scipy.optimize.leastsq was replaced by scipy.optimize.least_squares,
    # because it offers more fitting options. max_nfev instead of maxfev is one
    # of its keyword arguments, with a default value of None instead of 0.

    if bounds:
        # Wipe out fixed parameters, keep synced with initial.
        bounds_filtered = wipe_out_fixed(bounds)
        # bounds and bounds_filtered are dicts of 3-tuples with the third
        # element of those tuples a Boolean value, which can be passed to a
        # scipy.optimize.Bounds instance to allow for loosening of the bounds
        # when these bounds turn out infeasible in the fitting process. However,
        # The 3-tuples need to extracted into separate array_like objects to be
        # passed on as lb, ub and keep_feasible keyword arguments to the Bounds
        # instance.
        bounds_filtered_values = [[bounds_filtered[x][index] for x in
                                   bounds_filtered] for index in range(3)]
        lb = bounds_filtered_values[0]
        ub = bounds_filtered_values[1]
        keep_feasible = bounds_filtered_values[2]
        applied_bounds = scipy.optimize.Bounds(lb=lb, ub=ub,
                                               keep_feasible=keep_feasible)
    else:
        applied_bounds = (-np.inf, np.inf)

    Fitting_results = scipy.optimize.least_squares(
        residuals, initial, jac=jacobian_values,
        bounds=applied_bounds, method="dogbox",
        max_nfev=max_nfev, xtol=1e-4, ftol=1e-4
    )

    soln = Fitting_results["x"]
    success = Fitting_results["success"]

    if success > 4:
        raise ValueError("leastsq returned %d; bailing out" % (success,))

    # soln contains only the variable parameters; we need to merge the
    # contents of fixed into the soln list.
    # leastsq() returns either a np.float64 (if fitting a single value) or
    # a np.ndarray (if fitting multiple values); we need to turn that into
    # a list for the merger.
    try:
        # If a ndarray (or other iterable)
        soln = list(soln)
    except TypeError:
        soln = [soln]
    results = fixed.copy()
    for param in FIT_PARAMS:
        if param not in results:
            results[param] = soln.pop(0)

    if results['semiminor'] > results['semimajor']:
        # Swapped axis order is a perfectly valid fit, but inconvenient for
        # the rest of our codebase.
        results['semimajor'], results['semiminor'] = results['semiminor'], \
            results['semimajor']
        results['theta'] += np.pi / 2

    # Negative axes are a valid fit, since they are squared in the definition
    # of the Gaussian.
    results['semimajor'] = abs(results['semimajor'])
    results['semiminor'] = abs(results['semiminor'])

    return results


def goodness_of_fit(masked_residuals, noise, correlation_lengths):
    """
    Calculate the goodness-of-fit values.
    
    Parameters
    ----------
    masked_residuals : np.ma.MaskedArray
        The pixel-residuals from the fit.
    noise : float or np.ma.MaskedArray
        An estimate of the noise level. Can also be a masked numpy array
        matching the data for per-pixel noise estimates.
    correlation_lengths : tuple of two floats
        Distance along the semimajor and semiminor axes of the clean beam beyond
        which noise is assumed uncorrelated. Some background: Aperture
        synthesis imaging yields noise that is partially correlated over the
        entire image. This has a considerable effect on error estimates. We
        approximate this by considering all noise within the correlation length
        completely correlated and beyond that completely uncorrelated.
    
    Returns
    -------
    tuple
        chisq, reduced_chisq
    
    Notes
    -----
    We do not use the standard chi-squared formula for calculating these
    goodness-of-fit values, see
    <https://en.wikipedia.org/wiki/Goodness_of_fit#Regression_analysis>.
    These values are related to, but not quite the same as reduced chi-squared.
    The reduced chi-squared is statistically invalid for a Gaussian model
    from the outset (see <http://arxiv.org/abs/1012.3754>).
    
    We attempt to provide a resolution-independent estimate of goodness-of-fit
    ('reduced chi-squared') by estimating the number of independent pixels in the
    data, that we have used for fitting the Gaussian model, to normalize the
    chi-squared value.
    
    However, this will sometimes imply that we are fitting a fractional number
    of datapoints less than 1! As a result, it doesn't really make sense to try
    and apply the 'degrees-of-freedom' correction, as this would likely result
    in a negative ``reduced_chisq`` value.
    """
    gauss_resid_normed = (masked_residuals / noise).compressed()
    chisq = np.sum(gauss_resid_normed * gauss_resid_normed)
    n_fitted_pix = len(masked_residuals.compressed())
    n_indep_pix = indep_pixels(n_fitted_pix, correlation_lengths)
    reduced_chisq = chisq / n_indep_pix
    return chisq, reduced_chisq
