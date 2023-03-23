"""
Source fitting routines.
"""

import math

import numpy
import scipy.optimize

from . import utils
from .gaussian import gaussian
from .stats import indep_pixels
from numba import njit

FIT_PARAMS = ('peak', 'xbar', 'ybar', 'semimajor', 'semiminor', 'theta')


def moments(data, fudge_max_pix_factor, beamsize, threshold=0):
    """Calculate source positional values using moments

    Args:

        data (numpy.ndarray): Actual 2D image data

        fudge_max_pix_factor(float): Correct for the underestimation of the peak
                                     by taking the maximum pixel value.

        beamsize(float): The FWHM size of the clean beam

        threshold(float): source parameters like the semimajor and semiminor axes
                          derived from moments can be underestimated if one does not take
                          account of the threshold that was used to segment the source islands.

    Returns:
        dict: peak, total, x barycenter, y barycenter, semimajor
            axis, semiminor axis, theta

    Raises:
        exceptions.ValueError: in case of NaN in input.

    Use the first moment of the distribution is the barycenter of an
    ellipse. The second moments are used to estimate the rotation angle
    and the length of the axes.
    """

    # Are we fitting a -ve or +ve Gaussian?
    if data.mean() >= 0:
        # The peak is always underestimated when you take the highest pixel.
        peak = data.max() * fudge_max_pix_factor
    else:
        peak = data.min()
    ratio = threshold / peak
    total = data.sum()
    x, y = numpy.indices(data.shape)
    xbar = float((x * data).sum() / total)
    ybar = float((y * data).sum() / total)
    xxbar = (x * x * data).sum() / total - xbar ** 2
    yybar = (y * y * data).sum() / total - ybar ** 2
    xybar = (x * y * data).sum() / total - xbar * ybar

    working1 = (xxbar + yybar) / 2.0
    working2 = math.sqrt(((xxbar - yybar) / 2) ** 2 + xybar ** 2)

    # Some problems arise with the sqrt of (working1-working2) when they are
    # equal, this happens with islands that have a thickness of only one pixel
    # in at least one dimension.  Due to rounding errors this difference
    # becomes negative--->math domain error in sqrt.
    if len(data.nonzero()[0]) == 1:
        # This is the case when the island (or more likely subisland) has
        # a size of only one pixel.
        semiminor = numpy.sqrt(beamsize / numpy.pi)
        semimajor = numpy.sqrt(beamsize / numpy.pi)
    else:
        semimajor_tmp = (working1 + working2) * 2.0 * math.log(2.0)
        semiminor_tmp = (working1 - working2) * 2.0 * math.log(2.0)
        # ratio will be 0 for data that hasn't been selected according to a
        # threshold.
        if ratio != 0:
            # The corrections below for the semi-major and semi-minor axes are
            # to compensate for the underestimate of these quantities
            # due to the cutoff at the threshold.
            semimajor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
            semiminor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
        semimajor = math.sqrt(semimajor_tmp)
        semiminor = math.sqrt(semiminor_tmp)
        if semiminor == 0:
            # A semi-minor axis exactly zero gives all kinds of problems.
            # For instance wrt conversion to celestial coordinates.
            # This is a quick fix.
            semiminor = beamsize / (numpy.pi * semimajor)

    if (numpy.isnan(xbar) or numpy.isnan(ybar) or
            numpy.isnan(semimajor) or numpy.isnan(semiminor)):
        raise ValueError("Unable to estimate Gauss shape")

    # Theta is not affected by the cut-off at the threshold (see Spreeuw 2010,
    # page 45).
    if abs(semimajor - semiminor) < 0.01:
        # short circuit!
        theta = 0.
    else:
        if xxbar!=yybar:
            theta = math.atan(2. * xybar / (xxbar - yybar)) / 2.
        else:
            theta = numpy.sign(xybar) * math.pi / 4.0

        if theta * xybar > 0.:
            if theta < 0.:
                theta += math.pi / 2.0
            else:
                theta -= math.pi / 2.0

    ## NB: a dict should give us a bit more flexibility about arguments;
    ## however, all those here are ***REQUIRED***.
    return {
        "peak": peak,
        "flux": total,
        "xbar": xbar,
        "ybar": ybar,
        "semimajor": semimajor,
        "semiminor": semiminor,
        "theta": theta
    }

@njit
def moments_accelererated(island_data, posx, posy, fudge_max_pix_factor,
                          beamsize, threshold=0, clean_bias_error=0,
                          frac_flux_cal_error=0):
    """Calculate source properties using moments. Accelerated using JIT compilation.
    Also, a positional 1D index local to the island is used such that only pixels
    above the analysis threshold are addressed.

    Args:

        island_data (numpy.ndarray): Selected from the actual 2D image data, by taking
                                     pixels above the analysis threshold only, with its
                                     peak above the detection threshold. This selection
                                     results in a 1D ndarray (without a mask).

        posx: Row indices of the pixels in island_data as taken from the actual
              2D images data (rectangular slice)

        posy: Column indices of the pixels in island_data as taken from the actual
              2D images data (rectangular slice)

        fudge_max_pix_factor(float): Correct for the underestimation of the peak
                                     by taking the maximum pixel value.

        beamsize(float): The FWHM size of the clean beam

        threshold(float): source parameters like the semimajor and semiminor
                          axes derived from moments can be underestimated if
                          one does not take account of the threshold that
                          was used to segment the source islands.

        clean_bias_error: Extra source of error copied from the
                          Condon (PASP 109, 166 (1997)) formulae

        frac_flux_cal_error_error: Extra source of error copied from the
                          Condon (PASP 109, 166 (1997)) formulae
    Returns:
        dict: peak, total, x barycenter, y barycenter, semimajor
            axis, semiminor axis, theta

    Raises:
        exceptions.ValueError: in case of NaN in input.

    Use the first moment of the distribution is the barycenter of an
    ellipse. The second moments are used to estimate the rotation angle
    and the length of the axes.
    """
    # Are we fitting a -ve or +ve Gaussian?
    if island_data.mean() >= 0:
        # The peak is always underestimated when you take the highest pixel.
        peak = island_data.max() * fudge_max_pix_factor
    else:
        peak = island_data.min()
    ratio = threshold / peak
    flux = island_data.sum()
    xbar, ybar, xxbar, yybar, xybar = 0, 0, 0, 0, 0

    for index in range(island_data.size):
        i = posx[index]
        j = posy[index]
        xbar += i * island_data[index]
        ybar += j * island_data[index]
        xxbar += i * i * island_data[index]
        yybar += j * j * island_data[index]
        xybar += i * j * island_data[index]

    xbar /= flux
    ybar /= flux
    xxbar /= flux
    xxbar -= xbar ** 2
    yybar /= flux
    yybar -= ybar ** 2
    xybar /= flux
    xybar -= xbar * ybar

    working1 = (xxbar + yybar) / 2.0
    working2 = math.sqrt(((xxbar - yybar) / 2) ** 2 + xybar ** 2)

    # Some problems arise with the sqrt of (working1-working2) when they are
    # equal, this happens with islands that have a thickness of only one pixel
    # in at least one dimension.  Due to rounding errors this difference
    # becomes negative--->math domain error in sqrt.
    if len(island_data.nonzero()[0]) == 1:
        # This is the case when the island (or more likely subisland) has
        # a size of only one pixel.
        semiminor = numpy.sqrt(beamsize / numpy.pi)
        semimajor = numpy.sqrt(beamsize / numpy.pi)
    else:
        semimajor_tmp = (working1 + working2) * 2.0 * math.log(2.0)
        semiminor_tmp = (working1 - working2) * 2.0 * math.log(2.0)
        # ratio will be 0 for data that hasn't been selected according to a
        # threshold.
        if ratio != 0:
            # The corrections below for the semi-major and semi-minor axes are
            # to compensate for the underestimate of these quantities
            # due to the cutoff at the threshold.
            semimajor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
            semiminor_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
        semimajor = math.sqrt(semimajor_tmp)
        semiminor = math.sqrt(semiminor_tmp)
        if semiminor == 0:
            # A semi-minor axis exactly zero gives all kinds of problems.
            # For instance wrt conversion to celestial coordinates.
            # This is a quick fix.
            semiminor = beamsize / (numpy.pi * semimajor)

    if (numpy.isnan(xbar) or numpy.isnan(ybar) or
            numpy.isnan(semimajor) or numpy.isnan(semiminor)):
        raise ValueError("Unable to estimate Gauss shape")

    # Theta is not affected by the cut-off at the threshold (see Spreeuw 2010,
    # page 45).
    if abs(semimajor - semiminor) < 0.01:
        # short circuit!
        theta = 0.
    else:
        if xxbar!=yybar:
            theta = math.atan(2. * xybar / (xxbar - yybar)) / 2.
        else:
            theta = numpy.sign(xybar) * math.pi / 4.0

        if theta * xybar > 0.:
            if theta < 0.:
                theta += math.pi / 2.0
            else:
                theta -= math.pi / 2.0

    """Provide reasonable error estimates from the moments"""

    # The formulae below should give some reasonable estimate of the
    # errors from moments, should always be higher than the errors from
    # Gauss fitting.

    # This analysis is only possible if the peak flux is >= 0. This
    # follows from the definition of eq. 2.81 in Spreeuw's thesis. In that
    # situation, we set all errors to be infinite
    if peak < 0:
        errorpeak = float('inf')
        errorflux = float('inf')
        errorsmaj = float('inf')
        errorsmin = float('inf')
        errortheta= float('inf')

    clean_bias_error = self.clean_bias_error
    frac_flux_cal_error = self.frac_flux_cal_error
    theta_B, theta_b = correlation_lengths

    # This is eq. 2.81 from Spreeuw's thesis.
    rho_sq = ((16. * smaj * smin /
               (numpy.log(2.) * theta_B * theta_b * noise ** 2))
              * ((peak - threshold) /
                 (numpy.log(peak) - numpy.log(threshold))) ** 2)

    rho = numpy.sqrt(rho_sq)
    denom = numpy.sqrt(2. * numpy.log(2.)) * rho

    # Again, like above for the Condon formulae, we set the
    # positional variances to twice the theoretical values.
    error_par_major = 2. * smaj / denom
    error_par_minor = 2. * smin / denom

    # When these errors are converted to RA and Dec,
    # calibration uncertainties will have to be added,
    # like in formulae 27 of the NVSS paper.
    errorx = numpy.sqrt((error_par_major * numpy.sin(theta)) ** 2
                        + (error_par_minor * numpy.cos(theta)) ** 2)
    errory = numpy.sqrt((error_par_major * numpy.cos(theta)) ** 2
                        + (error_par_minor * numpy.sin(theta)) ** 2)

    # Note that we report errors in HWHM axes instead of FWHM axes
    # so the errors are half the errors of formula 29 of the NVSS paper.
    errorsmaj = numpy.sqrt(2) * smaj / rho
    errorsmin = numpy.sqrt(2) * smin / rho

    if smaj > smin:
        errortheta = 2.0 * (smaj * smin / (smaj ** 2 - smin ** 2)) / rho
    else:
        errortheta = numpy.pi
    if errortheta > numpy.pi:
        errortheta = numpy.pi

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
    errorpeak = numpy.sqrt(errorpeaksq)

    help1 = (errorsmaj / smaj) ** 2
    help2 = (errorsmin / smin) ** 2
    help3 = theta_B * theta_b / (4. * smaj * smin)
    errorflux = flux * numpy.sqrt(
        errorpeaksq / peak ** 2 + help3 * (help1 + help2))

    return numpy.array([[peak, total, xbar, ybar, semimajor, semiminor, theta],
                         [errorpeak, errorflux, errorx, errory, errorsmaj, errorsmin, errortheta]])


def fitgaussian(pixels, params, fixed=None, maxfev=0):
    """Calculate source positional values by fitting a 2D Gaussian

    Args:
        pixels (numpy.ma.MaskedArray): Pixel values (with bad pixels masked)

        params (dict): initial fit parameters (possibly estimated
            using the moments() function, above)

    Kwargs:
        fixed (dict): parameters & their values to be kept frozen (ie, not
            fitted)

        maxfev (int): maximum number of calls to the error function

    Returns:
        dict: peak, total, x barycenter, y barycenter, semimajor,
            semiminor, theta (radians)

    Raises:
        exceptions.ValueError: In case of a bad fit.

    Perform a least squares fit to an elliptical Gaussian.

    If a dict called fixed is passed in, then parameters specified within the
    dict with the same names as fit_params (below) will be "locked" in the
    fitting process.
    """
    fixed = fixed or {}

    # Collect necessary values from parameter dict; only those which aren't
    # fixed.
    initial = []
    for param in FIT_PARAMS:
        if param not in fixed:
            if hasattr(params[param], "value"):
                initial.append(params[param].value)
            else:
                initial.append(params[param])

    def residuals(paramlist):
        """Error function to be used in chi-squared fitting

        :argument paramlist: fitting parameters
        :type paramlist: numpy.ndarray
        :argument fixed: parameters to be held frozen
        :type fixed: dict

        :returns: 2d-array of difference between estimated Gaussian function
            and the actual pixels
        """
        paramlist = list(paramlist)
        gaussian_args = []
        for param in FIT_PARAMS:
            if param in fixed:
                gaussian_args.append(fixed[param])
            else:
                gaussian_args.append(paramlist.pop(0))

        # gaussian() returns a function which takes arguments x, y and returns
        # a Gaussian with parameters gaussian_args evaluated at that point.
        g = gaussian(*gaussian_args)

        # The .compressed() below is essential so the Gaussian fit will not
        # take account of the masked values (=below threshold) at the edges
        # and corners of pixels (=(masked) array, so rectangular in shape).
        pixel_resids = numpy.ma.MaskedArray(
            data=numpy.fromfunction(g, pixels.shape) - pixels,
            mask=pixels.mask)
        return pixel_resids.compressed()

    # maxfev=0, the default, corresponds to 200*(N+1) (NB, not 100*(N+1) as
    # the scipy docs state!) function evaluations, where N is the number of
    # parameters in the solution.
    # Convergence tolerances xtol and ftol established by experiment on images
    # from Paul Hancock's simulations.
    soln, success = scipy.optimize.leastsq(
        residuals, initial, maxfev=maxfev, xtol=1e-4, ftol=1e-4
    )

    if success > 4:
        raise ValueError("leastsq returned %d; bailing out" % (success,))

    # soln contains only the variable parameters; we need to merge the
    # contents of fixed into the soln list.
    # leastsq() returns either a numpy.float64 (if fitting a single value) or
    # a numpy.ndarray (if fitting multiple values); we need to turn that into
    # a list for the merger.
    try:
        # If an ndarray (or other iterable)
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
        results['theta'] += numpy.pi / 2

    # Negative axes are a valid fit, since they are squared in the definition
    # of the Gaussian.
    results['semimajor'] = abs(results['semimajor'])
    results['semiminor'] = abs(results['semiminor'])

    return results


def goodness_of_fit(masked_residuals, noise, correlation_lengths):
    """
    Calculates the goodness-of-fit values, `chisq` and `reduced_chisq`.

    .. Warning::
        We do not use the `standard chi-squared
        formula <https://en.wikipedia.org/wiki/Goodness_of_fit#Regression_analysis>`_
        for calculating these goodness-of-fit
        values, and should probably rename them in the next release.
        See below for details.


    These goodness-of-fit values are related to, but not quite the same as
    reduced chi-squared.
    Strictly speaking the reduced chi-squared is statistically
    invalid for a Gaussian model from the outset
    (see `arxiv:1012.3754 <http://arxiv.org/abs/1012.3754>`_).
    We attempt to provide a resolution-independent estimate of goodness-of-fit
    ('reduced chi-squared'), by using the same 'independent pixels' correction
    as employed when estimating RMS levels, to normalize the chi-squared value.
    However, as applied to the standard formula this will sometimes
    imply that we are fitting a fractional number of datapoints less than 1!
    As a result, it doesn't really make sense to try and apply the
    'degrees-of-freedom' correction, as this would likely result in a
    negative ``reduced_chisq`` value.
    (And besides, the 'degrees of freedom' concept is invalid for non-linear
    models.) Finally, note that when called from
    :func:`.source_profile_and_errors`, the noise-estimate at the peak-pixel
    is supplied, so will typically over-estimate the noise and
    hence under-estimate the chi-squared values.

    Args:
        masked_residuals(numpy.ma.MaskedArray): The pixel-residuals from the fit
        noise (float): An estimate of the noise level. Could also be set to
            a masked numpy array matching the data, for per-pixel noise
            estimates.

        correlation_lengths(tuple): Tuple of two floats describing the distance along the semimajor
                                    and semiminor axes of the clean beam beyond which noise
                                    is assumed uncorrelated. Some background: Aperture synthesis imaging
                                    yields noise that is partially correlated
                                    over the entire image. This has a considerable effect on error
                                    estimates. We approximate this by considering all noise within the
                                    correlation length completely correlated and beyond that completely
                                    uncorrelated.

    Returns:
        tuple: chisq, reduced_chisq

    """
    gauss_resid_normed = (masked_residuals / noise).compressed()
    chisq = numpy.sum(gauss_resid_normed * gauss_resid_normed)
    n_fitted_pix = len(masked_residuals.compressed().ravel())
    n_indep_pix = indep_pixels(n_fitted_pix, correlation_lengths)
    reduced_chisq = chisq / n_indep_pix
    return chisq, reduced_chisq
