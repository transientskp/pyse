"""
Source fitting routines.
"""

import math

import numpy
import scipy.optimize
from .gaussian import gaussian, jac_gaussian
from .stats import indep_pixels
from sourcefinder.deconv import deconv
from numba import guvectorize, float64, float32, int32

FIT_PARAMS = ('peak', 'xbar', 'ybar', 'semimajor', 'semiminor', 'theta')


def moments(data, fudge_max_pix_factor, beamsize, threshold=0):
    """Calculate source positional values using moments

    Use the first moment of the distribution is the barycenter of an
    ellipse. The second moments are used to estimate the rotation angle
    and the length of the axes.

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
        if xxbar != yybar:
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


@guvectorize([(float32[:], int32[:], int32[:], int32[:], int32, int32, float32,
               float32, float32, float64, float64, float64[:], float64,
               float64[:], float64, float64, float32[:, :], float32[:, :])],
              ('(n), (m), (n), (n), (), (), (), (), (), (), (), (k), (), (m), ' +
               '(), (), (l, p) -> (l, p)'), nopython=True)
def moments_enhanced(island_data, chunkpos, posx, posy, min_width, no_pixels,
                     threshold, noise, maxi, fudge_max_pix_factor,
                     max_pix_variance_factor, beam, beamsize,
                     correlation_lengths,
                     clean_bias_error, frac_flux_cal_error, dummy,
                     computed_moments):
    """Calculate source properties using moments. Vectorized using the
    guvectorize decorator.

    Use the first moment of the distribution is the barycenter of an
    ellipse. The second moments are used to estimate the rotation angle
    and the length of the axes.

    Also, a positional 2D index local to the island is used such that every
    pixel value can be linked to a position relative to a corner of the island.

    Args:

        island_data (numpy.ndarray): Selected from the actual 2D image data,
                                     by taking pixels above the analysis
                                     threshold only, with its peak above the
                                     detection threshold. This selection
                                     results in a 1D ndarray (without a mask).

        chunkpos (numpy.ndarray): Index array of length 2 denoting the position
                                  of the top left corner of the rectangular
                                  slice encompassing the island relative to the
                                  top left corner of the image, which has pixel
                                  coordinates (0, 0), i.e. we need chunkpos
                                  to return to absolute pixel coordinates.

        posx (numpy.ndarray): Row indices of the pixels in island_data as taken
                              from the actual 2D images data
                              (rectangular slice).

        posy  (numpy.ndarray): Column indices of the pixels in island_data as
                               taken from the actual 2D images data (rectangular
                                slice).

        min_width (integer): The minimum width (in pixels) of the island. This
                             was derived by calculating the maximum width of the
                             island over x and y and then taking the minimum of
                             those two numbers.

        no_pixels (integer): The number of pixels that constitute the island.

        threshold(float): source parameters like the semimajor and semiminor
                          axes derived from moments can be underestimated if
                          one does not take account of the threshold that
                          was used to segment the source islands.

        noise(float): local noise, i.e. the standard deviation of the
                      background pixel values, at the position of the
                      peak pixel value of the island.

        maxi(float): peak pixel value from island.

        fudge_max_pix_factor(float): Correct for the underestimation of the peak
                                     by taking the maximum pixel value.

        max_pix_variance_factor (float): Take account of additional variance
                                        induced by the maximum pixel method,
                                        on top of the background noise.

        beam(numpy.ndarray): array from three floats: semimaj, semimin, theta.

        beamsize(float): The FWHM size of the clean beam

        correlation_lengths(numpy.ndarray): array from two floats describing the
                                    distance along the semi-major and semi-minor
                                    axes of the clean beam beyond which noise is
                                    assumed uncorrelated. Some background:
                                    Aperture synthesis imaging yields noise that
                                    is partially correlated over the entire
                                    image. This has a considerable effect on
                                    error estimates. We approximate this by
                                    considering all noise within the
                                    correlation length completely correlated
                                    and beyond that completely uncorrelated.

        clean_bias_error: Extra source of error copied from the
                          Condon (PASP 109, 166 (1997)) formulae

        frac_flux_cal_error: Extra source of error copied from the
                          Condon (PASP 109, 166 (1997)) formulae

        dummy (numpy.ndarray): Empty array with the same shape as
                               computed_moments needed because of a flau
                               in guvectorize: There is no other way to tell
                               guvectorize what the shape of the output array
                               will be. Therefore, we define an otherwise
                               redundant input array with the same shape as
                               the desired output array.

        computed_moments(numpy.ndarray): a (10, 2) array of floats containing
                                the computed moments, i.e.peak flux density,
                                total flux, x barycenter, y barycenter,
                                semimajor axis, semiminor axis, position angle
                                and the deconvolved counterparts of the latter
                                three quantities. This constitutes a total of
                                ten quantities and their corresponding errors.

    Returns:
        None (because of the guvectorize decorator), but computed_moments is
             filled with values.

    Raises:
        exceptions.ValueError: in case of NaN in input.

    """

    # Not every island has the same size. The number of columns of the array
    # containing all islands is equal to the maximum number of pxiels over
    # all islands. This containing array was created by numpy.empty, so better
    # dump the redundant elements that have undetermined values.
    island_data = island_data[:no_pixels]
    # Are we fitting a -ve or +ve Gaussian?
    if island_data.mean() >= 0:
        # The peak is always underestimated when you take the highest pixel.
        peak = maxi * fudge_max_pix_factor
    else:
        peak = island_data.min()

    ratio = threshold / peak
    total = island_data.sum()
    xbar, ybar, xxbar, yybar, xybar = 0, 0, 0, 0, 0

    for index in range(no_pixels):
        i = posx[index]
        j = posy[index]
        xbar += i * island_data[index]
        ybar += j * island_data[index]
        xxbar += i * i * island_data[index]
        yybar += j * j * island_data[index]
        xybar += i * j * island_data[index]

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
    smaj_tmp = (working1 + working2) * 2.0 * math.log(2.0)
    smin_tmp = (working1 - working2) * 2.0 * math.log(2.0)
    # ratio will be 0 for data that hasn't been selected according to a
    # threshold.
    if ratio != 0:
        # The corrections below for the semi-major and semi-minor axes are
        # to compensate for the underestimate of these quantities
        # due to the cutoff at the threshold.
        smaj_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))
        smin_tmp /= (1.0 + math.log(ratio) * ratio / (1.0 - ratio))

    # We need this width to determine Gaussian shape parameters in a meaningful
    # way.
    if min_width > 2:
        # The idea of the try except here is that, even though we require a
        # minimum width of the island, there may still be occasions where
        # working2 can be slightly higher than working1, perhaps due to rounding
        # errors.
        try:
            smaj = math.sqrt(smaj_tmp)
            smin = math.sqrt(smin_tmp)
            if smin == 0:
                # A semi-minor axis exactly zero gives all kinds of problems.
                # For instance wrt conversion to celestial coordinates.
                # This is a quick fix.
                smin = beamsize / (numpy.pi * smaj)

            if (numpy.isnan(xbar) or numpy.isnan(ybar) or
                    numpy.isnan(smaj) or numpy.isnan(smin)):
                raise ValueError("Unable to estimate Gauss shape")

            # Theta is not affected by the cut-off at the threshold (see Spreeuw 2010,
            # page 45).
            if abs(smaj - smin) < 0.01:
                # short circuit!
                theta = 0.
            else:
                if xxbar != yybar:
                    theta = math.atan(2. * xybar / (xxbar - yybar)) / 2.
                else:
                    theta = numpy.sign(xybar) * math.pi / 4.0

                if theta * xybar > 0.:
                    if theta < 0.:
                        theta += math.pi / 2.0
                    else:
                        theta -= math.pi / 2.0

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

    #  Equivalent of param["flux"] = (numpy.pi * param["peak"] *
    #  param["semimajor"] * param["semiminor"] / beamsize) from extract.py.
    flux = numpy.pi * peak * smaj * smin / beamsize

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

    """Deconvolve from the clean beam"""

    # If the fitted axes are smaller than the clean beam
    # (=restoring beam) axes, the axes and position angle
    # can be deconvolved from it.
    fmaj = 2. * smaj
    fmajerror = 2. * errorsmaj
    fmin = 2. * smin
    fminerror = 2. * errorsmin
    fpa = numpy.degrees(theta)
    fpaerror = numpy.degrees(errortheta)
    cmaj = 2. * beam[0]
    cmin = 2. * beam[1]
    cpa = numpy.degrees(beam[2])

    rmaj, rmin, rpa, ierr = deconv(fmaj, fmin, fpa, cmaj, cmin, cpa)
    # This parameter gives the number of components that could not be
    # deconvolved, IERR from deconf.f.
    deconv_imposs = ierr
    # Now, figure out the error bars.
    if rmaj > 0:
        # In this case the deconvolved position angle is defined.
        # For convenience we reset rpa to the interval [-90, 90].
        if rpa > 90:
            rpa = -numpy.mod(-rpa, 180.)
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
                rpa1 = -numpy.mod(-rpa1, 180.)
            rpaerror1 = numpy.abs(rpa1 - rpa)
            # An angle error can never be more than 90 degrees.
            if rpaerror1 > 90.:
                rpaerror1 = numpy.mod(-rpaerror1, 180.)
        else:
            rpaerror1 = numpy.nan
        rmaj2, rmin2, rpa2, ierr2 = deconv(
            fmaj, fmin, fpa - fpaerror, cmaj, cmin, cpa)
        if ierr2 < 2:
            if rpa2 > 90:
                rpa2 = -numpy.mod(-rpa2, 180.)
            rpaerror2 = numpy.abs(rpa2 - rpa)
            # An angle error can never be more than 90 degrees.
            if rpaerror2 > 90.:
                rpaerror2 = numpy.mod(-rpaerror2, 180.)
        else:
            rpaerror2 = numpy.nan
        if numpy.isnan(rpaerror1) or numpy.isnan(rpaerror2):
            theta_deconv_error = numpy.nansum(
                numpy.array([rpaerror1, rpaerror2]))
        else:
            theta_deconv_error = numpy.mean(
                numpy.array([rpaerror1, rpaerror2]))
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
                semimaj_deconv_error = numpy.mean(numpy.array(
                    [numpy.abs(rmaj3 - rmaj), numpy.abs(rmaj - rmaj4)]))
            else:
                semimaj_deconv_error = numpy.abs(rmaj3 - rmaj)
        else:
            rmin4, rmaj4, rpa4, ierr4 = deconv(
                fmin, fmaj - fmajerror, fpa, cmaj, cmin, cpa)
            if rmaj4 > 0:
                semimaj_deconv_error = numpy.mean(numpy.array(
                    [numpy.abs(rmaj3 - rmaj), numpy.abs(rmaj - rmaj4)]))
            else:
                semimaj_deconv_error = numpy.abs(rmaj3 - rmaj)
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
                semimin_deconv_error = numpy.mean(numpy.array(
                    [numpy.abs(rmin6 - rmin), numpy.abs(rmin5 - rmin)]))
            else:
                semimin_deconv_error = numpy.abs(rmin5 - rmin)
        else:
            semimin_deconv = numpy.nan
            semimin_deconv_error = numpy.nan
    else:
        semimaj_deconv = numpy.nan
        semimaj_deconv_error = numpy.nan
        semimin_deconv = numpy.nan
        semimin_deconv_error = numpy.nan
        theta_deconv = numpy.nan
        theta_deconv_error= numpy.nan

    computed_moments[0, :] = numpy.array([peak, flux, xbar, ybar, smaj,
                                          smin, theta, semimaj_deconv,
                                          semimin_deconv, theta_deconv])
    computed_moments[1, :] = numpy.array([errorpeak, errorflux, errorx,
                                          errory, errorsmaj, errorsmin,
                                          errortheta, semimaj_deconv_error,
                                          semimin_deconv_error,
                                          theta_deconv_error])


def fitgaussian(pixels, params, fixed=None, max_nfev=None, bounds={}):
    """Calculate source positional values by fitting a 2D Gaussian

    :Args:
        pixels (numpy.ma.MaskedArray): Pixel values (with bad pixels masked)

        params (dict): initial fit parameters (possibly estimated
            using the moments() function, above)

    :Kwargs:
        fixed (dict): parameters & their values to be kept frozen (ie, not
            fitted)

        max_nfev (int): maximum number of calls to the error function

        bounds (dict): can be a dict such as the extract.ParamSet().bounds
            attribute generated by the extract.ParamSet().compute_bounds method,
            but any dict with keys from FIT_PARAMS and
            (lower_bound, upper_bound, bool) tuples as values will do. The
            boolean argument accommodates for loosening a bound when a fit
            becomes unfeasible because of the bound, see the
            scipy.optimize.Bounds documentation and source code for background.
            lower_bound and upper_bound are floats.

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

    def residuals(params):
        """Error function to be used in chi-squared fitting

        Args:
            params(numpy.ndarray): fitting parameters

        Returns:
            1d-array of difference between estimated Gaussian function
            and the actual pixels. (pixel_resids is a 2d-array, but the
            .compressed() makes it 1d.)
        """
        paramlist = list(params)
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
        # and corners of pixels (=(masked) array, so rectangular).
        pixel_resids = numpy.ma.MaskedArray(
            data=numpy.fromfunction(g, pixels.shape) - pixels,
            mask=pixels.mask)

        return pixel_resids.compressed()

    def jacobian_values(params):
        """The Jacobian of an anisotropic 2D Gaussian at the pixel positions.

        Args:
            params(numpy.ndarray): fitting parameters

        Returns:
             2d-array with values of the partial derivatives of the
             2d anisotropic Gaussian along its six parameters. These
             values are evaluated across the unmasked pixel positions of
             the island that constitutes the source. The number of rows
             equals the number of pixels of the flattened unmasked island.
             The number of columns equals the number of partial
             derivatives of the Gaussian (=6). For fixed Gaussian
             parameters the Jacobian component is obviously zero, so that
             results in a column of only zeroes.

        """
        paramlist = list(params)
        gaussian_args = []
        for param in FIT_PARAMS:
            if param in fixed:
                gaussian_args.append(fixed[param])
            else:
                gaussian_args.append(paramlist.pop(0))

        # jac is a list of six functions corresponding to the six partial
        # derivatives of the 2D anisotropic Gaussian profile.
        jac = jac_gaussian(gaussian_args)

        # From the six functions in jac, wipe out the ones that
        # correspond to fixed parameters, must be in sync with initial.
        jac_filtered = wipe_out_fixed(jac)

        jac_values = [numpy.ma.MaskedArray(
            data=numpy.fromfunction(jac_filtered[key], pixels.shape),
            mask=pixels.mask).compressed() for key in jac_filtered]

        return numpy.array(jac_values).T

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
        applied_bounds = (-numpy.inf, numpy.inf)

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
    # leastsq() returns either a numpy.float64 (if fitting a single value) or
    # a numpy.ndarray (if fitting multiple values); we need to turn that into
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
