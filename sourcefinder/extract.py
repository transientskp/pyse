"""
Source Extraction Helpers.

These are used in conjunction with image.ImageData.
"""

from sourcefinder.deconv import deconv
from sourcefinder.utility import coordinates
from sourcefinder.utility.uncertain import Uncertain
from .gaussian import gaussian
from . import measuring
from . import utils
import logging
from collections.abc import MutableMapping
from numba import guvectorize, float64, float32, int32
import numpy as np
np.seterr(divide="raise", invalid="raise")

try:
    import ndimage
except ImportError:
    from scipy import ndimage

logger = logging.getLogger(__name__)

# This is used as a dummy value, -BIGNUM values will be always be masked.
# As such, it should be larger than the expected range of real values,
# since we will get negative values representing real data after
# background subtraction, etc.
BIGNUM = 99999.0


class Island(object):
    """
    The source extraction process forms islands, which it then fits.
    Each island needs to know its position in the image (ie, x, y pixel
    value at one corner), the threshold above which it is detected
    (analysis_threshold by default, but will increase if the island is
    the result of deblending), and a data array.

    The island should provide a means of deblending: splitting itself
    apart and returning multiple sub-islands, if necessary.
    """

    def __init__(self, data, rms, chunk, analysis_threshold, detection_map,
                 beam, deblend_nthresh, deblend_mincont, structuring_element,
                 rms_orig=None, flux_orig=None, subthrrange=None
                 ):

        # deblend_nthresh is the number of subthresholds used when deblending.
        self.deblend_nthresh = deblend_nthresh
        # If we deblend too far, we hit the recursion limit. And it's slow.
        if self.deblend_nthresh > 300:
            logger.warning("Limiting to 300 deblending subtresholds")
            self.deblend_nthresh = 300
        else:
            logger.debug("Using %d subthresholds", deblend_nthresh)

        # Deblended components of this island must contain at least
        # deblend_mincont times the total flux of the original to be regarded
        # as significant.
        self.deblend_mincont = deblend_mincont

        # The structuring element defines connectivity between pixels.
        self.structuring_element = structuring_element

        # NB we have set all unused data to -(lots) before passing it to
        # Island().
        mask = np.where(data > -BIGNUM / 10.0, 0, 1)
        self.data = np.ma.array(data, mask=mask)
        self.rms = rms
        self.chunk = chunk
        self.analysis_threshold = analysis_threshold
        self.detection_map = detection_map
        self.beam = beam
        self.max_pos = ndimage.maximum_position(self.data.filled(fill_value=0))
        self.position = (self.chunk[0].start, self.chunk[1].start)
        if not isinstance(rms_orig, np.ndarray):
            self.rms_orig = self.rms
        else:
            self.rms_orig = rms_orig
        # The idea here is to retain the flux of the original, unblended
        # island. That flux is used as a criterion for deblending.
        if not isinstance(flux_orig, float):
            self.flux_orig = self.data.sum()
        else:
            self.flux_orig = flux_orig
        if isinstance(subthrrange, np.ndarray):
            self.subthrrange = subthrrange
        else:
            self.subthrrange = utils.generate_subthresholds(
                self.data.min(), self.data.max(), self.deblend_nthresh
            )

    def deblend(self, niter=0):
        """Return a decomposed numpy array of all the subislands.

        Iterate up through subthresholds, looking for our island
        splitting into two. If it does, start again, with two or more
        separate islands.
        """

        logger.debug("Deblending source")
        for level in self.subthrrange[niter:]:

            # The idea is to retain the parent island when no significant
            # subislands are found and jump to the next subthreshold
            # using niter.
            # Deblending is started at a level higher than the lowest
            # pixel value in the island.
            # Deblending at the level of the lowest pixel value will
            # likely not yield anything, because the island was formed at
            # threshold just below that.
            # So that is why we use niter+1 (>=1) instead of niter (>=0).

            if level > self.data.max():
                # level is above the highest pixel value...
                # Return the current island.
                break
            clipped_data = np.where(
                self.data.filled(fill_value=0) >= level, 1, 0)
            labels, number = ndimage.label(clipped_data, self.structuring_element)

            # If we have more than one island, then we need to make subislands.
            if number > 1:
                subislands = []
                label = 0
                for chunk in ndimage.find_objects(labels):
                    label += 1
                    newdata = np.where(
                        labels == label,
                        self.data.filled(fill_value=-BIGNUM), -BIGNUM
                    )
                    # NB: In class Island(object), rms * analysis_threshold
                    # is taken as the threshold for the bottom of the island.
                    # Everything below that level is masked.
                    # For subislands, this product should be equal to level
                    # and flat, i.e., horizontal.
                    # We can achieve this by setting rms=level*ones and
                    # analysis_threshold=1.
                    island = Island(
                        newdata[chunk],
                        (np.ones(self.data[chunk].shape) * level),
                        (
                            slice(self.chunk[0].start + chunk[0].start,
                                  self.chunk[0].start + chunk[0].stop),
                            slice(self.chunk[1].start + chunk[1].start,
                                  self.chunk[1].start + chunk[1].stop)
                        ),
                        1,
                        self.detection_map[chunk],
                        self.beam,
                        self.deblend_nthresh,
                        self.deblend_mincont,
                        self.structuring_element,
                        self.rms_orig[chunk[0].start:chunk[0].stop,
                                      chunk[1].start:chunk[1].stop],
                        self.flux_orig,
                        self.subthrrange
                    )

                    subislands.append(island)
                # This line should filter out any subisland with insufficient
                # flux, in about the same way as SExtractor.
                # Sufficient means: the flux of the branch above the
                # subthreshold (=level) must exceed some user given fraction
                # of the composite object, i.e., the original island.
                subislands = [isl for isl in subislands if (isl.data - np.ma.array(
                    np.ones(isl.data.shape) * level,
                    mask=isl.data.mask)).sum() > self.deblend_mincont *
                              self.flux_orig]
                # Discard subislands below detection threshold
                subislands = [isl for isl in subislands if (isl.data - isl.detection_map).max() >= 0]
                numbersignifsub = len(subislands)
                # Proceed with the previous island, but make sure the next
                # subthreshold is higher than the present one.
                # Or we would end up in an infinite loop...
                if numbersignifsub > 1:
                    if niter + 1 < self.deblend_nthresh:
                        # Apparently, the map command always results in
                        # nested lists.
                        return list(utils.flatten([island.deblend(niter=niter + 1) for island in subislands]))
                    else:
                        return subislands
                elif numbersignifsub == 1 and niter + 1 < self.deblend_nthresh:
                    return Island.deblend(self, niter=niter + 1)
                else:
                    # In this case we have numbersignifsub == 0 or
                    # (1 and reached the highest subthreshold level).
                    # Pull out of deblending loop, return current island.
                    break
        # We've not found any subislands: just return this island.
        return self

    def threshold(self):
        """Threshold"""
        return self.noise() * self.analysis_threshold

    def noise(self):
        """Noise at maximum position"""
        return self.rms[self.max_pos]

    def sig(self):
        """Deviation"""
        return (self.data / self.rms_orig).max()

    def fit(self, fudge_max_pix_factor, beamsize, correlation_lengths,
            fixed=None):
        """Fit the position"""
        try:
            measurement, gauss_island, gauss_residual = \
                source_profile_and_errors(self.data, self.threshold(),
                                          self.rms, self.noise(), self.beam,
                                          fudge_max_pix_factor,
                                          beamsize, correlation_lengths,
                                          fixed=fixed)
        except ValueError:
            # Fitting failed
            logger.error("Moments & Gaussian fitting failed at %s" % (
                str(self.position)))
            return None
        measurement["xbar"] += self.position[0]
        measurement["ybar"] += self.position[1]
        measurement.sig = self.sig()
        return measurement, gauss_island, gauss_residual


class ParamSet(MutableMapping):
    """
    All the source fitting methods should go to produce a ParamSet, which
    gives all the information necessary to make a Detection.
    """

    def __init__(self, clean_bias=0.0, clean_bias_error=0.0,
                 frac_flux_cal_error=0.0, alpha_maj1=2.5, alpha_min1=0.5,
                 alpha_maj2=0.5, alpha_min2=2.5, alpha_maj3=1.5,
                 alpha_min3=1.5):

        self.clean_bias = clean_bias
        self.clean_bias_error = clean_bias_error
        self.frac_flux_cal_error = frac_flux_cal_error
        self.alpha_maj1 = alpha_maj1
        self.alpha_min1 = alpha_min1
        self.alpha_maj2 = alpha_maj2
        self.alpha_min2 = alpha_min2
        self.alpha_maj3 = alpha_maj3
        self.alpha_min3 = alpha_min3

        self.measurements = {
            'peak': Uncertain(),
            'flux': Uncertain(),
            'xbar': Uncertain(),
            'ybar': Uncertain(),
            'semimajor': Uncertain(),
            'semiminor': Uncertain(),
            'theta': Uncertain(),
            'semimaj_deconv': Uncertain(),
            'semimin_deconv': Uncertain(),
            'theta_deconv': Uncertain()
        }
        # This parameter gives the number of components that could not be
        # deconvolved, IERR from deconf.f.
        self.deconv_imposs = 2

        # These flags are used to indicate where the values stored in this
        # parameterset have come from: we set them to True if & when moments
        # and/or Gaussian fitting succeeds.
        self.moments = False
        # Gaussian fits are preferably performed with bounds for better
        # stability, but to establish those bounds in a meaningful manner
        # moments estimation is required, i.e. self.moments = True.
        self.bounds = {}
        self.gaussian = False

        # It is instructive to keep a record of the minimal cross-section of
        # the island, i.e. is its minimal width large enough for at least a
        # proper moments computation? Set thin_detection to True as default.
        # Will be set to False before the start of a source measurement if the
        # island is wide enough.
        self.thin_detection = True

        # More metadata about the fit: only valid for Gaussian fits:
        self.chisq = None
        self.reduced_chisq = None
        self.sig = None

    def __getitem__(self, item):
        return self.measurements[item]

    def __setitem__(self, item, value):
        if item in self.measurements:
            if isinstance(value, Uncertain):
                self.measurements[item] = value
            else:
                self.measurements[item].value = value
        else:
            raise AttributeError("Invalid parameter")

    def __delitem__(self, key):
        del self.measurements[key]

    def __iter__(self):
        return iter(self.measurements)

    def __len__(self):
        return len(self.measurements)

    def keys(self):
        """ """
        return list(self.measurements.keys())

    def compute_bounds(self, data_shape):
        """ Calculate bounds for 'safer' Gauss fitting, i.e. a smaller chance
        on runaway solutions. The bounds are largely based on moments
        estimation, so it only makes sense to impose bounds if moments
        estimation was successful.

        Args:
            data_shape: (int32, int32) or (int64, int64) tuple describing the
                        shape of the rectangular area (slice) encompassing the
                        island used for the fit. Used to set bounds for the
                        position of the source in the fitting process.

        Creates dict of (float, float, bool) tuples, i.e. for a maximum of six
        Gaussian fit parameters a (lower_bound, upper_bound, bool) tuple. The
        boolean entry is used to loosen a bound when a fit becomes unfeasible,
        see the documentation on scipy.optimize.Bounds.
        """
        if hasattr(self["peak"], "value"):
            self.bounds["peak"] = (0.5 * self["peak"].value,
                                   1.5 * self["peak"].value, False)
        else:
            self.bounds["peak"] = (0.5 * self["peak"], 1.5 * self["peak"],
                                   False)
        # Accommodate for negative heights. The "bounds" argument in
        # scipy.optimize.least-squares demands that the lower bound is
        # smaller than the upper bound and with the (0.5, 1.5) fractions
        # this does not work out well for negative heights. Background:
        # only to be expected in images from visibilities from polarization
        # products other than Stokes I.
        if self.bounds["peak"][0] > self.bounds["peak"][1]:
            true_upper = self.bounds["peak"][0]
            true_lower = self.bounds["peak"][1]
            self.bounds["peak"] = (true_lower, true_upper, False)

        if hasattr(self["semimajor"], "value"):
            self.bounds["semimajor"] = (0.5 * self["semimajor"].value,
                                        1.5 * self["semimajor"].value,
                                        False)
        else:
            self.bounds["semimajor"] = (0.5 * self["semimajor"],
                                        1.5 * self["semimajor"], False)

        if hasattr(self["semiminor"], "value"):
            self.bounds["semiminor"] = (0.5 * self["semiminor"].value,
                                        1.5 * self["semiminor"].value,
                                        False)
        else:
            self.bounds["semiminor"] = (0.5 * self["semiminor"],
                                        1.5 * self["semiminor"], False)

        self.bounds["xbar"] = (0., data_shape[0], False)
        self.bounds["ybar"] = (0., data_shape[1], False)
        # The upper bound for theta is a bit odd, one would expect
        # np.pi/2 here, but that will yield imperfect fits in
        # cases where the axes are aligned with the coordinate axes,
        # which, in turn, breaks the AxesSwapGaussTest.testFitHeight
        # and AxesSwapGaussTest.testFitSize unit tests. Thus, some
        # margin needs to be applied.
        # keep_feasibly is True here to accommodate for a fitting process
        # where the margin is exceeded.
        self.bounds["theta"] = (-np.pi/2, np.pi, True)

        return self

    def calculate_errors(self, noise, correlation_lengths, threshold):
        """Calculate positional errors

        Uses _condon_formulae() if this object is based on a Gaussian fit,
        _error_bars_from_moments() if it's based on moments.
        """

        if self.gaussian:
            return self._condon_formulae(noise, correlation_lengths)
        else:
            # If thin_detection == True, _error_bars_from_moments probably
            # gives a better estimate of the true errors than errors from
            # Gaussian fits, i.e. from _condon_formulae, since they are
            # generally larger. _error_bars_from_moments could still be
            # underestimating the true errors in case of thin detections.
            if not threshold:
                threshold = 0
            return self._error_bars_from_moments(noise, correlation_lengths,
                                                 threshold)

    def _condon_formulae(self, noise, correlation_lengths):
        """Returns the errors on parameters from Gaussian fits according to
        the Condon (PASP 109, 166 (1997)) formulae.

        These formulae are not perfect, but we'll use them for the
        time being.  (See Refregier and Brown (astro-ph/9803279v1) for
        a more rigorous approach.) It also returns the corrected peak.
        The peak is corrected for the overestimate due to the local
        noise gradient.
        """
        peak = self['peak'].value
        flux = self['flux'].value
        smaj = self['semimajor'].value
        smin = self['semiminor'].value
        theta = self['theta'].value

        theta_B, theta_b = correlation_lengths

        rho_sq1 = ((smaj * smin / (theta_B * theta_b)) *
                   (1. + (theta_B / (2. * smaj)) ** 2) ** self.alpha_maj1 *
                   (1. + (theta_b / (2. * smin)) ** 2) ** self.alpha_min1 *
                   (peak / noise) ** 2)
        rho_sq2 = ((smaj * smin / (theta_B * theta_b)) *
                   (1. + (theta_B / (2. * smaj)) ** 2) ** self.alpha_maj2 *
                   (1. + (theta_b / (2. * smin)) ** 2) ** self.alpha_min2 *
                   (peak / noise) ** 2)
        rho_sq3 = ((smaj * smin / (theta_B * theta_b)) *
                   (1. + (theta_B / (2. * smaj)) ** 2) ** self.alpha_maj3 *
                   (1. + (theta_b / (2. * smin)) ** 2) ** self.alpha_min3 *
                   (peak / noise) ** 2)

        rho1 = np.sqrt(rho_sq1)
        rho2 = np.sqrt(rho_sq2)

        denom1 = np.sqrt(2. * np.log(2.)) * rho1
        denom2 = np.sqrt(2. * np.log(2.)) * rho2

        # Here you get the errors parallel to the fitted semi-major and
        # semi-minor axes as taken from the NVSS paper (Condon et al. 1998,
        # AJ, 115, 1693), formula 25.
        # Those variances are twice the theoreticals, so the errors in
        # position are sqrt(2) as large as one would get from formula 21
        # of the Condon (1997) paper.
        error_par_major = 2. * smaj / denom1
        error_par_minor = 2. * smin / denom2

        # When these errors are converted to RA and Dec,
        # calibration uncertainties will have to be added,
        # like in formulae 27 of the NVSS paper.
        errorx = np.sqrt((error_par_major * np.sin(theta)) ** 2 +
                            (error_par_minor * np.cos(theta)) ** 2)
        errory = np.sqrt((error_par_major * np.cos(theta)) ** 2 +
                            (error_par_minor * np.sin(theta)) ** 2)

        # Note that we report errors in HWHM axes instead of FWHM axes
        # so the errors are half the errors of formula 29 of the NVSS paper.
        errorsmaj = np.sqrt(2) * smaj / rho1
        errorsmin = np.sqrt(2) * smin / rho2

        if smaj > smin:
            errortheta = 2.0 * (smaj * smin / (smaj ** 2 - smin ** 2)) / rho2
        else:
            errortheta = np.pi
        if errortheta > np.pi:
            errortheta = np.pi

        # Formula (34) of the NVSS paper, without the general fitting bias,
        # since we don't observe that. On the contrary, there may be peak
        # fitting bias, but that has the opposite sign, i.e. fitted peaks are
        # systematically biased too low, but more investigation is needed to
        # quantify that effect.
        peak += self.clean_bias

        # Formula (37) of the NVSS paper.
        errorpeaksq = ((self.frac_flux_cal_error * peak) ** 2 +
                       self.clean_bias_error ** 2 +
                       2. * peak ** 2 / rho_sq3)

        errorpeak = np.sqrt(errorpeaksq)

        help1 = (errorsmaj / smaj) ** 2
        help2 = (errorsmin / smin) ** 2
        help3 = theta_B * theta_b / (4. * smaj * smin)

        # Since we recomputed the peak spectral brightness following formula
        # (34) from the NVSS paper, we also need to recompute the flux density
        # following formula (35) of that paper.
        flux *= peak / self['peak'].value

        errorflux = np.abs(flux) * np.sqrt(
            errorpeaksq / peak ** 2 + help3 * (help1 + help2))

        self['peak'] = Uncertain(peak, errorpeak)
        self['flux'] = Uncertain(flux, errorflux)
        self['xbar'].error = errorx
        self['ybar'].error = errory
        self['semimajor'].error = errorsmaj
        self['semiminor'].error = errorsmin
        self['theta'].error = errortheta

        return self

    def _error_bars_from_moments(self, noise, correlation_lengths,
                                 threshold):
        """Provide reasonable error estimates from the moments"""

        # The formulae below should give some reasonable estimate of the
        # errors from moments, should always be higher than the errors from
        # Gauss fitting.
        peak = self['peak'].value
        flux = self['flux'].value
        smaj = self['semimajor'].value
        smin = self['semiminor'].value
        theta = self['theta'].value

        # There is no point in proceeding with processing an image if any
        # detected peak spectral brightness is below zero since that implies
        # that part of detectionthresholdmap is below zero.
        if peak < 0:
            raise ValueError((f"Peak from moments = {peak:.2e}, something is "
                              "possibly wrong with detectionthresholdmap, "
                              "since a negative peak from moments analysis "
                              "should not occur."))

        clean_bias_error = self.clean_bias_error
        frac_flux_cal_error = self.frac_flux_cal_error
        theta_B, theta_b = correlation_lengths

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
        # This correction is performed in measuring.py. The maximum pixel
        # method introduces a peak dependent error corresponding to the last
        # term in the expression below for errorpeaksq.
        # To this, we add, in quadrature, the errors corresponding
        # to the first and last term of the rhs of equation 37 of the
        # NVSS paper. The middle term in that equation 37 is heuristically
        # replaced by noise**2 since the threshold should not affect
        # the error from the (corrected) maximum pixel method,
        # while it is part of the expression for rho_sq above.
        errorpeaksq = ((frac_flux_cal_error * peak) ** 2 +
                       clean_bias_error ** 2 + noise ** 2)
        errorpeak = np.sqrt(errorpeaksq)

        help1 = (errorsmaj / smaj) ** 2
        help2 = (errorsmin / smin) ** 2
        help3 = theta_B * theta_b / (4. * smaj * smin)
        errorflux = flux * np.sqrt(
            errorpeaksq / peak ** 2 + help3 * (help1 + help2))

        self['peak'].error = errorpeak
        self['flux'].error = errorflux
        self['xbar'].error = errorx
        self['ybar'].error = errory
        self['semimajor'].error = errorsmaj
        self['semiminor'].error = errorsmin
        self['theta'].error = errortheta

        return self

    def deconvolve_from_clean_beam(self, beam):
        """Deconvolve with the clean beam"""

        # If the fitted axes are larger than the clean beam
        # (=restoring beam) axes, the axes and position angle
        # can be deconvolved from it.
        fmaj = 2. * self['semimajor'].value
        fmajerror = 2. * self['semimajor'].error
        fmin = 2. * self['semiminor'].value
        fminerror = 2. * self['semiminor'].error
        fpa = np.degrees(self['theta'].value)
        fpaerror = np.degrees(self['theta'].error)
        cmaj = 2. * beam[0]
        cmin = 2. * beam[1]
        cpa = np.degrees(beam[2])

        rmaj, rmin, rpa, ierr = deconv(fmaj, fmin, fpa, cmaj, cmin, cpa)
        # This parameter gives the number of components that could not be
        # deconvolved, IERR from deconf.f.
        self.deconv_imposs = ierr
        # Now, figure out the error bars.
        if rmaj > 0:
            # In this case the deconvolved position angle is defined.
            # For convenience we reset rpa to the interval [-90, 90].
            if rpa > 90:
                rpa = -np.mod(-rpa, 180.)
            self['theta_deconv'].value = rpa

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
                self['theta_deconv'].error = np.nansum(
                    [rpaerror1, rpaerror2])
            else:
                self['theta_deconv'].error = np.mean(
                    [rpaerror1, rpaerror2])
            self['semimaj_deconv'].value = rmaj / 2.
            rmaj3, rmin3, rpa3, ierr3 = deconv(
                fmaj + fmajerror, fmin, fpa, cmaj, cmin, cpa)
            # If rmaj>0, then rmaj3 should also be > 0,
            # if I am not mistaken, see the formulas at
            # the end of ch.2 of Spreeuw's Ph.D. thesis.
            if fmaj - fmajerror > fmin:
                rmaj4, rmin4, rpa4, ierr4 = deconv(
                    fmaj - fmajerror, fmin, fpa, cmaj, cmin, cpa)
                if rmaj4 > 0:
                    self['semimaj_deconv'].error = np.mean(
                        [np.abs(rmaj3 - rmaj), np.abs(rmaj - rmaj4)])
                else:
                    self['semimaj_deconv'].error = np.abs(rmaj3 - rmaj)
            else:
                rmin4, rmaj4, rpa4, ierr4 = deconv(
                    fmin, fmaj - fmajerror, fpa, cmaj, cmin, cpa)
                if rmaj4 > 0:
                    self['semimaj_deconv'].error = np.mean(
                        [np.abs(rmaj3 - rmaj), np.abs(rmaj - rmaj4)])
                else:
                    self['semimaj_deconv'].error = np.abs(rmaj3 - rmaj)
            if rmin > 0:
                self['semimin_deconv'].value = rmin / 2.
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
                    self['semimin_deconv'].error = np.mean(
                        [np.abs(rmin6 - rmin), np.abs(rmin5 - rmin)])
                else:
                    self['semimin_deconv'].error = np.abs(rmin5 - rmin)
            else:
                self['semimin_deconv'] = Uncertain(
                    np.nan, np.nan)
        else:
            self['semimaj_deconv'] = Uncertain(np.nan, np.nan)
            self['semimin_deconv'] = Uncertain(np.nan, np.nan)
            self['theta_deconv'] = Uncertain(np.nan, np.nan)

        return self


def source_profile_and_errors(data, threshold, rms, noise, beam,
                              fudge_max_pix_factor, beamsize,
                              correlation_lengths, fixed=None):
    """Return a number of measurable properties with errorbars

    Given an island of pixels it will return a number of measurable
    properties including errorbars.  It will also compute residuals
    from Gauss fitting and export these to a residual map.

    In addition to handling the initial parameter estimation, and any fits
    which fail to converge, this function runs the goodness-of-fit
    calculations -
    see :func:`sourcefinder.measuring.goodness_of_fit` for details.

    Args:

        data: np.ma.Maskedarray or np.ndarray
            Array of pixel values, can be a masked
            array, which is necessary for proper Gauss fitting,
            because the pixels below the threshold in the corners and
            along the edges should not be included in the fitting
            process

        threshold (float): Threshold used for selecting pixels for the
            source (ie, building an island)

        rms (np.ndarray or np.ma.MaskedArray): noise levels at pixel positions
            corresponding to the data array, determined from the interpolated
            grid of standard deviations of the background pixels.

        noise (float): rms (noise level) at the maximum pixel position

        beam (tuple): beam parameters (semimaj,semimin,theta)

        fudge_max_pix_factor(float): Correct for the underestimation of the peak
                                     by taking the maximum pixel value.

        beamsize(float): The FWHM size of the clean beam

        correlation_lengths(tuple): Tuple of two floats describing the distance
                                    along the semimajor and semiminor axes of
                                    the clean beam beyond which noise is assumed
                                    uncorrelated. Some background: Aperture
                                    synthesis imaging yields noise that is
                                    partially correlated over the entire image.
                                    This has a considerable effect on error
                                    estimates. We approximate this by
                                    considering all noise within the
                                    correlation length completely correlated
                                     and beyond that completely uncorrelated.

    Kwargs:

        fixed (dict): Parameters (and their values) to hold fixed while fitting.
            Passed on to measuring.fitgaussian().

    Returns:
        tuple: a populated ParamSet, an islands map and a residuals map.
            Note that both the islands and residuals maps are regular ndarrays,
            where masked (unfitted) regions have been filled with 0-values.

    """

    if fixed is None:
        fixed = {}
    param = ParamSet()

    if threshold is None:
        moments_threshold = 0
    else:
        moments_threshold = threshold

    # We can always find the maximum pixel value and derive the barycenter
    # position. The other three Gaussian parameters we can copy from the clean
    # beam.

    peak = data.max()
    # Are we fitting a -ve or +ve Gaussian?
    if peak <= 0:
        peak = data.min()
    # The peak is always underestimated when you take the highest (or lowest)
    # pixel.
    peak *= fudge_max_pix_factor

    total = data.sum()
    x, y = np.indices(data.shape)
    xbar = float((x * data).sum() / total)
    ybar = float((y * data).sum() / total)

    param.update({
        "peak": peak,
        "flux": np.pi * peak * beam[0] * beam[1] / beamsize,
        "xbar": xbar,
        "ybar": ybar,
        "semimajor": beam[0],
        "semiminor": beam[1],
        "theta": beam[2]
    })

    if fixed:
        param.update(fixed)

    # data_as_ones is constructed to help determine if the island has enough
    # width for Gauss fitting.
    try:
        if data.mask.shape == data.data.shape:
            data_as_ones = np.where(~data.mask == 1, 1, 0)
        else:
            data_as_ones = np.where(data.data > moments_threshold, 1, 0)
    except AttributeError:
        data_as_ones = np.where(data > moments_threshold, 1, 0)
    max_along_x = np.sum(data_as_ones, axis=0).max()
    max_along_y = np.sum(data_as_ones, axis=1).max()
    minimum_width = min(max_along_x, max_along_y)

    if minimum_width > 2:
        # We can realistically compute moments and perform Gaussian fits if
        # the island has a thickness of more than 2 in both dimensions.
        param.thin_detection = False
        if not fixed:
            # Moments can only be computed if no parameters are fixed.
            try:
                param.update(measuring.moments(data, fudge_max_pix_factor, beam,
                                             beamsize, moments_threshold))
                if moments_threshold:
                    # A complete moments computation is only possible if we have
                    # imposed a non-zero threshold (and no parameters are
                    # fixed).
                    param.moments = True
                    param.compute_bounds(data.shape)
            except ValueError:
                logger.warning('Moments computations failed, use defaults.')
        try:
            gaussian_soln = measuring.fitgaussian(data, param, fixed=fixed,
                                                bounds=param.bounds)
            param.update(gaussian_soln)
            param.gaussian = True
            logger.debug('Gaussian fitting was successful.')
        except ValueError:
            logger.warning('Gaussian fitting failed.')
    else:
        logger.debug("Unable to estimate gaussian parameters from moments."
                     " Proceeding with defaults %s""",
                     str(param))

    param["flux"] = (np.pi * param["peak"] * param["semimajor"] *
                     param["semiminor"] / beamsize)
    param.calculate_errors(noise, correlation_lengths, threshold)
    param.deconvolve_from_clean_beam(beam)

    # Calculate residuals
    # NB this works even if Gaussian fitting fails, we generate the model from
    # the moments parameters.
    gauss_arg = (param["peak"].value,
                 param["xbar"].value,
                 param["ybar"].value,
                 param["semimajor"].value,
                 param["semiminor"].value,
                 param["theta"].value)

    try:
        gauss_island_masked = \
            np.ma.array(gaussian(*gauss_arg)(*np.indices(data.shape)),
                           mask=data.mask)
        gauss_resid_masked = data - gauss_island_masked
        gauss_island_filled = gauss_island_masked.filled(fill_value=0.)
        gauss_resid_filled = gauss_resid_masked.filled(fill_value=0.)
    except AttributeError:
        gauss_island_masked = gaussian(*gauss_arg)(*np.indices(data.shape))
        gauss_resid_masked = data - gauss_island_masked
        gauss_island_filled = gauss_island_masked
        gauss_resid_filled = gauss_resid_masked

    param.chisq, param.reduced_chisq = measuring.goodness_of_fit(
        gauss_resid_masked, rms, correlation_lengths)

    return param, gauss_island_filled, gauss_resid_filled


class Detection(object):
    """The result of a measurement at a given position in a given image."""

    def __init__(self, paramset, imagedata, chunk=None, eps_ra=0, eps_dec=0):

        self.eps_ra = eps_ra
        self.eps_dec = eps_dec

        self.imagedata = imagedata
        # self.wcs = imagedata.wcs
        self.chunk = chunk

        self.peak = paramset['peak']
        self.flux = paramset['flux']
        self.x = paramset['xbar']
        self.y = paramset['ybar']
        self.smaj = paramset['semimajor']
        self.smin = paramset['semiminor']
        self.theta = paramset['theta']
        # This parameter gives the number of components that could not
        # be deconvolved, IERR from deconf.f.
        self.dc_imposs = paramset.deconv_imposs
        self.smaj_dc = paramset['semimaj_deconv']
        self.smin_dc = paramset['semimin_deconv']
        self.theta_dc = paramset['theta_deconv']
        self.error_radius = None
        self.gaussian = paramset.gaussian
        self.chisq = paramset.chisq
        self.reduced_chisq = paramset.reduced_chisq

        self.sig = paramset.sig

        try:
            self._physical_coordinates()
        except RuntimeError:
            logger.warning("Physical coordinates failed at %f, %f" % (
                self.x, self.y))
            raise

    def __getstate__(self):
        return {
            'imagedata': self.imagedata,
            'chunk': (self.chunk[0].start, self.chunk[0].stop,
                      self.chunk[1].start, self.chunk[1].stop),
            'peak': self.peak,
            'flux': self.flux,
            'x': self.x,
            'y': self.y,
            'smaj': self.smaj,
            'smin': self.smin,
            'theta': self.theta,
            'sig': self.sig,
            'error_radius': self.error_radius,
            'gaussian': self.gaussian,
        }

    def __setstate__(self, attrdict):
        self.imagedata = attrdict['imagedata']
        self.chunk = (slice(attrdict['chunk'][0], attrdict['chunk'][1]),
                      slice(attrdict['chunk'][2], attrdict['chunk'][3]))
        self.peak = attrdict['peak']
        self.flux = attrdict['flux']
        self.x = attrdict['x']
        self.y = attrdict['y']
        self.smaj = attrdict['smaj']
        self.smin = attrdict['smin']
        self.theta = attrdict['theta']
        self.sig = attrdict['sig']
        self.error_radius = attrdict['error_radius']
        self.gaussian = attrdict['gaussian']

        try:
            self._physical_coordinates()
        except RuntimeError as e:
            logger.warning("Physical coordinates failed at %f, %f" % (
                self.x, self.y))
            raise

    def __getattr__(self, attrname):
        # Backwards compatibility for "errquantity" attributes
        if attrname[:3] == "err":
            return self.__getattribute__(attrname[3:]).error
        else:
            raise AttributeError(attrname)

    def __str__(self):
        return "(%.2f, %.2f) +/- (%.2f, %.2f): %g +/- %g" % (
            self.ra.value, self.dec.value, self.ra.error * 3600,
            self.dec.error * 3600,
            self.peak.value, self.peak.error)

    def __repr__(self):
        return str(self)

    def _physical_coordinates(self):
        """Convert the pixel parameters for this object into something
        physical."""

        # First, the RA & dec.
        self.ra, self.dec = [Uncertain(x) for x in self.imagedata.wcs.p2s(
            [self.x.value, self.y.value])]
        if np.isnan(self.dec.value) or abs(self.dec) > 90.0:
            raise ValueError("object falls outside the sky")

        # First, determine local north.
        help1 = np.cos(np.radians(self.ra.value))
        help2 = np.sin(np.radians(self.ra.value))
        help3 = np.cos(np.radians(self.dec.value))
        help4 = np.sin(np.radians(self.dec.value))
        center_position = np.array([help3 * help1, help3 * help2, help4])

        # The length of this vector is chosen such that it touches
        # the tangent plane at center position.
        # The cross product of the local north vector and the local east
        # vector will always be aligned with the center_position vector.
        if center_position[2] != 0:
            local_north_position = np.array(
                [0., 0., 1. / center_position[2]])
        else:
            # If we are right on the equator (ie dec=0) the division above
            # will blow up: as a workaround, we use something Really Big
            # instead.
            local_north_position = np.array([0., 0., 99e99])
        # Next, determine the orientation of the y-axis wrt local north
        # by incrementing y by a small amount and converting that
        # to celestial coordinates. That small increment is conveniently
        # chosen to be an increment of 1 pixel.

        endy_ra, endy_dec = self.imagedata.wcs.p2s(
            [self.x.value, self.y.value + 1.])
        help5 = np.cos(np.radians(endy_ra))
        help6 = np.sin(np.radians(endy_ra))
        help7 = np.cos(np.radians(endy_dec))
        help8 = np.sin(np.radians(endy_dec))
        endy_position = np.array([help7 * help5, help7 * help6, help8])

        # Extend the length of endy_position to make it touch the plane
        # tangent at center_position.
        endy_position /= np.dot(center_position, endy_position)

        diff1 = endy_position - center_position
        diff2 = local_north_position - center_position

        cross_prod = np.cross(diff2, diff1)

        length_cross_sq = np.dot(cross_prod, cross_prod)

        normalization = np.dot(diff1, diff1) * np.dot(diff2, diff2)

        # The length of the cross product equals the product of the lengths of
        # the vectors times the sine of their angle.
        # This is the angle between the y-axis and local north,
        # measured eastwards.
        # yoffset_angle = np.degrees(
        #    np.arcsin(np.sqrt(length_cross_sq/normalization)))
        # The formula above is commented out because the angle computed
        # in this way will always be 0<=yoffset_angle<=90.
        # We'll use the dotproduct instead.
        yoffs_rad = (np.arccos(np.dot(diff1, diff2) /
                                  np.sqrt(normalization)))

        # The multiplication with -sign_cor makes sure that the angle
        # is measured eastwards (increasing RA), not westwards.
        sign_cor = (np.dot(cross_prod, center_position) /
                    np.sqrt(length_cross_sq))
        yoffs_rad *= -sign_cor
        yoffset_angle = np.degrees(yoffs_rad)

        # Now that we have the BPA, we can also compute the position errors
        # properly, by projecting the errors in pixel coordinates (x and y)
        # on local north and local east.
        errorx_proj = np.sqrt(
            (self.x.error * np.cos(yoffs_rad)) ** 2 +
            (self.y.error * np.sin(yoffs_rad)) ** 2)
        errory_proj = np.sqrt(
            (self.x.error * np.sin(yoffs_rad)) ** 2 +
            (self.y.error * np.cos(yoffs_rad)) ** 2)

        # Now we have to sort out which combination of errorx_proj and
        # errory_proj gives the largest errors in RA and Dec.
        try:
            end_ra1, end_dec1 = self.imagedata.wcs.p2s(
                [self.x.value + errorx_proj, self.y.value])
            end_ra2, end_dec2 = self.imagedata.wcs.p2s(
                [self.x.value, self.y.value + errory_proj])
            # Here we include the position calibration errors
            self.ra.error = self.eps_ra + max(
                np.fabs(self.ra.value - end_ra1),
                np.fabs(self.ra.value - end_ra2))
            self.dec.error = self.eps_dec + max(
                np.fabs(self.dec.value - end_dec1),
                np.fabs(self.dec.value - end_dec2))
        except RuntimeError:
            # We get a runtime error from wcs.p2s if the errors place the
            # limits outside the image.
            # In which case we set the RA / DEC uncertainties to infinity
            self.ra.error = float('inf')
            self.dec.error = float('inf')

        # Estimate an absolute angular error on our central position.
        self.error_radius = utils.get_error_radius(
            self.imagedata.wcs, self.x.value, self.x.error, self.y.value,
            self.y.error
        )

        # Now we can compute the BPA, east from local north.
        # That these angles can simply be added is not completely trivial.
        # First, the Gaussian in gaussian.py must be such that theta is
        # measured from the positive y-axis in the direction of negative x.
        # Secondly, x and y are defined such that the direction
        # positive y-->negative x-->negative y-->positive x is the same
        # direction (counterclockwise) as (local) north-->east-->south-->west.
        # If these two conditions are matched, the formula below is valid.
        # Of course, the formula is also valid if theta is measured
        # from the positive y-axis towards positive x
        # and both of these directions are equal (clockwise).
        self.theta_celes = Uncertain(
            (np.degrees(self.theta.value) + yoffset_angle) % 180,
            np.degrees(self.theta.error))
        if not np.isnan(self.theta_dc.value):
            self.theta_dc_celes = Uncertain(
                (self.theta_dc.value + yoffset_angle) % 180,
                np.degrees(self.theta_dc.error))
        else:
            self.theta_dc_celes = Uncertain(np.nan, np.nan)

        # Next, the axes.
        # Note that the signs of np.sin and np.cos in the
        # four expressions below are arbitrary.
        self.end_smaj_x = (self.x.value - np.sin(self.theta.value) *
                           self.smaj.value)
        self.start_smaj_x = (self.x.value + np.sin(self.theta.value) *
                             self.smaj.value)
        self.end_smaj_y = (self.y.value + np.cos(self.theta.value) *
                           self.smaj.value)
        self.start_smaj_y = (self.y.value - np.cos(self.theta.value) *
                             self.smaj.value)
        self.end_smin_x = (self.x.value + np.cos(self.theta.value) *
                           self.smin.value)
        self.start_smin_x = (self.x.value - np.cos(self.theta.value) *
                             self.smin.value)
        self.end_smin_y = (self.y.value + np.sin(self.theta.value) *
                           self.smin.value)
        self.start_smin_y = (self.y.value - np.sin(self.theta.value) *
                             self.smin.value)

        def pixel_to_spatial(x, y):
            try:
                return self.imagedata.wcs.p2s([x, y])
            except RuntimeError:
                logger.debug("pixel_to_spatial failed at %f, %f" % (x, y))
                return np.nan, np.nan

        end_smaj_ra, end_smaj_dec = pixel_to_spatial(self.end_smaj_x,
                                                     self.end_smaj_y)
        end_smin_ra, end_smin_dec = pixel_to_spatial(self.end_smin_x,
                                                     self.end_smin_y)

        smaj_asec = coordinates.angsep(self.ra.value, self.dec.value,
                                       end_smaj_ra, end_smaj_dec)
        scaling_smaj = smaj_asec / self.smaj.value
        errsmaj_asec = scaling_smaj * self.smaj.error
        self.smaj_asec = Uncertain(smaj_asec, errsmaj_asec)

        smin_asec = coordinates.angsep(self.ra.value, self.dec.value,
                                       end_smin_ra, end_smin_dec)
        scaling_smin = smin_asec / self.smin.value
        errsmin_asec = scaling_smin * self.smin.error
        self.smin_asec = Uncertain(smin_asec, errsmin_asec)

    def distance_from(self, x, y):
        """Distance from center"""
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5

    def serialize(self, ew_sys_err, ns_sys_err):
        """
        Return source properties suitable for database storage.

        We manually add ew_sys_err, ns_sys_err

        returns: a list of tuples containing all relevant fields
        """
        return [
            self.ra.value,
            self.dec.value,
            self.ra.error,
            self.dec.error,
            self.peak.value,
            self.peak.error,
            self.flux.value,
            self.flux.error,
            self.sig,
            self.smaj_asec.value,
            self.smin_asec.value,
            self.theta_celes.value,
            ew_sys_err,
            ns_sys_err,
            self.error_radius,
            self.gaussian,
            self.chisq,
            self.reduced_chisq
        ]


@guvectorize([(float64[:], float64[:], float32[:], float32[:], float32[:],
               float32[:])], '(n), (n), (n), (l), (m) -> (m)', nopython=True)
def first_part_of_celestial_coordinates(ra_dec, endy_ra_dec,
                                        xbar_ybar_error,
                                        xbar_ybar_smaj_smin_theta,
                                        dummy, return_values):
    """
    Similar to extract.Detection._physical_coordinates, but vectorized and
    mainly the first part, until we need another call to wcs.all_pix2world,
    based on the output from this part.
    What we have learned from moments_enhanced is that measuring the islands
    can be done very rapidly, by vectorizing the measurements rather than
    parallellizing, although one can do both by using the guvectorize
    decorator from Numba. Contrary to vectorizing moments, vectorizing
    the transformation to celestial coordinates is harder, since it involves
    wcs.wcs_pix2world from Astropy, which Numba cannot compile. The way out
    of this is to use wcs.all_pix2world from Astropy, which is a fast
    vectorized function, but this has to be done outside the routine compiled
    by Numba, i.e. this routine. We will be needing a number of calls to
    wcs.all_pix2world to traverse all the steps from
    extract.Detection._physical_coordinates.

    Args:
        ra_dec (np.ndarray): array of floats of length 2, containing the
            right ascension (degrees) and the declination
            (degrees) corresponding to [xbar, ybar] of the source.

        endy_ra_dec (np.ndarray): array of floats of length 2, containing
            the right ascension (degrees) and the declination
            (degrees) corresponding to [xbar, ybar + 1] of the source.

        xbar_ybar_error (np.ndarray): array of floats of length 2,
            with the errors on the barycentric positions, in both dimensions.

        xbar_ybar_smaj_smin_theta (np.ndarray): array of floats of length 5,
            with the barycentric positions, the semi-major and semi-minor axes
            and the position angles.

        dummy (np.ndarray): Same shape as return_values, to accommodate for a
            missing feature in the guvectorize decorator, i.e. this is a trick
            to pass on information about the shape of the output array.

    Returns:
        return_values (np.ndarray): Strictly speaking nothing is returned,
        because of the guvectorize decorator, but return_values is filled with
        xerror_proj, yerror_proj, yoffset_angle, end_smaj_x, start_smaj_x,
        end_smaj_y,  start_smaj_y, end_smin_x, start_smin_x, end_smin_y and
        start_smin_y.
    """

    ra, dec = ra_dec
    if np.isnan(dec) or abs(dec) > 90.0:
        raise ValueError("object falls outside the sky")

    # First, determine local north.
    help1 = np.cos(np.radians(ra))
    help2 = np.sin(np.radians(ra))
    help3 = np.cos(np.radians(dec))
    help4 = np.sin(np.radians(dec))
    center_position = np.array([help3 * help1, help3 * help2, help4])

    # The length of this vector is chosen such that it touches
    # the tangent plane at center position.
    # The cross product of the local north vector and the local east
    # vector will always be aligned with the center_position vector.
    if center_position[2] != 0:
        local_north_position = np.array(
            [0., 0., 1. / center_position[2]])
    else:
        # If we are right on the equator (ie dec=0) the division above
        # will blow up: as a workaround, we use something Really Big
        # instead.
        local_north_position = np.array([0., 0., 99e99])

    endy_ra, endy_dec = endy_ra_dec
    help5 = np.cos(np.radians(endy_ra))
    help6 = np.sin(np.radians(endy_ra))
    help7 = np.cos(np.radians(endy_dec))
    help8 = np.sin(np.radians(endy_dec))
    endy_position = np.array([help7 * help5, help7 * help6, help8])

    # Extend the length of endy_position to make it touch the plane
    # tangent at center_position.
    endy_position /= np.dot(center_position, endy_position)

    diff1 = endy_position - center_position
    diff2 = local_north_position - center_position

    cross_prod = np.cross(diff2, diff1)

    length_cross_sq = np.dot(cross_prod, cross_prod)

    normalization = np.dot(diff1, diff1) * np.dot(diff2, diff2)

    # The length of the cross product equals the product of the lengths of
    # the vectors times the sine of their angle.
    # This is the angle between the y-axis and local north,
    # measured eastwards.
    # yoffset_angle = np.degrees(
    #    np.arcsin(np.sqrt(length_cross_sq/normalization)))
    # The formula above is commented out because the angle computed
    # in this way will always be 0<=yoffset_angle<=90.
    # We'll use the dotproduct instead.
    yoffs_rad = (np.arccos(np.dot(diff1, diff2) /
                              np.sqrt(normalization)))

    # The multiplication with -sign_cor makes sure that the angle
    # is measured eastwards (increasing RA), not westwards.
    sign_cor = (np.dot(cross_prod, center_position) /
                np.sqrt(length_cross_sq))
    yoffs_rad *= -sign_cor
    yoffset_angle = np.degrees(yoffs_rad)

    # Now that we have the BPA, we can also compute the position errors
    # properly, by projecting the errors in pixel coordinates (x and y)
    # on local north and local east.
    xbar_error, ybar_error = xbar_ybar_error
    errorx_proj = np.sqrt(
        (xbar_error * np.cos(yoffs_rad)) ** 2 +
        (ybar_error * np.sin(yoffs_rad)) ** 2)
    errory_proj = np.sqrt(
        (xbar_error * np.sin(yoffs_rad)) ** 2 +
        (ybar_error * np.cos(yoffs_rad)) ** 2)

    # Next, the axes.
    # Note that the signs of np.sin and np.cos in the
    # four expressions below are arbitrary.
    xbar, ybar, smaj, smin, theta = xbar_ybar_smaj_smin_theta
    end_smaj_x = xbar - np.sin(theta) * smaj
    start_smaj_x = xbar + np.sin(theta) * smaj
    end_smaj_y = ybar + np.cos(theta) * smaj
    start_smaj_y = ybar - np.cos(theta) * smaj
    end_smin_x = xbar + np.cos(theta) * smin
    start_smin_x = xbar - np.cos(theta) * smin
    end_smin_y = ybar + np.sin(theta) * smin
    start_smin_y = ybar - np.sin(theta) * smin

    return_values[:] = np.array([errorx_proj, errory_proj, yoffset_angle,
                                 end_smaj_x, start_smaj_x, end_smaj_y, start_smaj_y,
                                 end_smin_x, start_smin_x, end_smin_y, start_smin_y])


@guvectorize([(float32[:, :], float32[:, :], int32[:], int32[:, :], int32[:],
               int32[:], float32[:], float32[:], int32[:], int32[:], int32[:],)],
             '(n, m), (n, m), (l), (n, m), (), (), (k) -> (k), (k), (k), ()')
def insert_sources_and_noise(some_image, noise_map, inds, labelled_data, label,
                             npix, source, noise, xpos, ypos, min_width):
    """
    We want to copy the relevant source and noise data into input arrays for
    measuring.moments_enhanced. Simultaneously calculate the minimum width of
    each island; when determining source properties this is an important
    integer, i.e. with sufficient width all six Gaussian parameters can be
    calculated in a robust manner.

    :param some_image: The 2D ndarray with all the pixel values, typically
                       self.data_bgsubbed.data.

    :param noise_map: The 2D ndarray with all the noise values, i.e. an
                      estimate of the standard deviation of the background
                      pixels, from an interpolation across the background
                      grid, centered on subimages. Typically
                      self.rmsmap.data.

    :param inds: A ndarray of four indices indicating the slice encompassing
                 an island. Such a slice would typically be a pick from a
                 list of slices from a call to scipy.ndimage.find_objects.
                 Since we are attempting vectorized processing here, the
                 slice should have been replaced by its four coordinates
                 through a call to slices_to_indices.

    :param labelled_data: A ndarray with the same shape as some_image, with
                          labelled islands with integer values and zeroes
                          for all background pixels.

    :param label: The label (integer value) corresponding to the slice
                  encompassing the island. Or actually it should be the
                  other way round, since there can be multiple islands
                  within one rectangular slice.

    :param npix: Number of pixels comprising the island with label=label.

    :param source: 1D ndarray of float32 pixel values of the island with
                   label = label. Length = maxpix, the number of pixels in the
                   largest possible island, indicating that there will be a
                   number of unassigned values for the highest indices, for some
                   islands.

    :param noise: 1D ndarray of float32 rms noise values - local estimates of
                  the standard deviation of the background noise - of the
                  island with label = label. Length = maxpix, the number of
                  pixels in the largest possible island, indicating that there
                  will be a number of unassigned values for the highest indices,
                  for some islands.

    :param xpos: 1D ndarray of integers indicating the row indices of the
                 pixels of the island with label = label, relative to the
                 position of pixel [0, 0] of the rectangular slice
                 encompassing the island. Must have same order as pixel
                 values in island. Length = maxpix, the number of pixels in
                 the largest possible island, indicating that there will be
                 a number of unassigned values for the highest indices.

    :param ypos: 1D ndarray of integers indicating the column indices of the
                 pixels of the island with label = label, relative to the
                 position of pixel [0, 0]  of the rectangular slice
                 encompassing the island. Must have same order as
                 pixel values in island. Length = maxpix, the number of
                 pixels in the largest possible island, indicating that
                 there will be a number of unassigned values for the highest
                 indices.

    :param min_width: int32 indicating the minimum width of the island, in order
                 to assess whether the island has sufficient width to determine
                 Gaussian parameters. Calculated from the maximum widths of the
                 island over x and y and subsequently taking the minimum of
                 those two maximum widths.

    :return: No return values, because of the use of the guvectorize
             decorator: 'guvectorize() functions don’t return their
             result value: they take it as an array argument,
             which must be filled in by the function'. In this case
             source, noise, xpos, ypos and min_width will be filled with values.
    """

    image_chunk = some_image[inds[0]:inds[1], inds[2]:inds[3]]
    labelled_data_chunk = labelled_data[inds[0]:inds[1], inds[2]:inds[3]]
    noise_chunk = noise_map[inds[0]:inds[1], inds[2]:inds[3]]

    segmented_island = np.where(labelled_data_chunk == label[0], 1, 0)

    max_along_x = np.sum(segmented_island, axis=0).max()
    max_along_y = np.sum(segmented_island, axis=1).max()
    min_width[0] = min(max_along_x, max_along_y)

    # pos = "positions", i.e. the row and column indices of the island pixels.
    pos = segmented_island.nonzero()

    for i in range(npix[0]):
        index = pos[0][i], pos[1][i]
        source[i] = image_chunk[index]
        noise[i] = noise_chunk[index]
        xpos[i], ypos[i] = index


def source_measurements_pixels_and_celestial_vectorised(num_islands, npixs,
                                                        maxposs, maxis,
                                                        data_bgsubbeddata,
                                                        rmsdata,
                                                        analysisthresholddata,
                                                        indices,
                                                        labelled_data, labels, wcs,
                                                        fudge_max_pix_factor,
                                                        beam, beamsize,
                                                        correlation_lengths,
                                                        eps_ra, eps_dec):
    """
    From islands of pixels above the analysis threshold with peaks above the
    detection threshold source parameters are extracted, including error bars
    These quantities are transformed to celestial coordinates in a vectorized
    way, making it suitable for tens of thousands of sources per image.
    Per island one can extract a maximum of 10 quantities: peak spectral
    brightness, flux density, i.e. the spectral brightness integrated over
    the source, position in the sky along both axes and the source shape
    parameters: semi-major and semi-minor axes and the position angle of the
    major axis, measured east from local north. The latter three quantities can
    be deconvolved from the clean beam if the source is resolved. That gives a
    total af 20 numbers, 10 measurements and their error bars. 8 of these
    have to be transformed to celestial coordinates, which adds another 16
    numbers.
    This function is a duplicate of code elsewhere, in particular of
    'measuring.moments' and 'extract.Detection._physical_coordinates' but with
    a data layout suitable for vectorisation.

    :param eps_ra:
    :param num_islands: integer equal to the number of islands detected.

    :param npixs: 1D int32 ndarray of length num_islands representing the number
                  of pixels comprising every island.

    :param maxposs: 2D int32 ndarray of shape (num_islands, 2) with the peak
                    positions per island

    :param maxis: 1D float32 ndarray of length num_islands with the peak pixel
                  values corresponding to the maxposs positions.

    :param data_bgsubbeddata: 2D ndarray of float32, representing the
                              image. These are the data of
                              ImageData.data_bgsubbed, i.e.
                              ImageData.data_bgsubbed.data, without the mask,
                              since we assume that any mask has already
                              been properly propagated in selecting the islands
                              pixels. The interpolated mean background has been
                              subtracted from the image data.

    :param rmsdata: 2D ndarray of float32 or float64, same shape as the image
                    (data_bgsubbeddata). rms is jargon in image processing, it
                    refers to "Root mean square", but is taken as the
                    standard deviation of the background noise. The standard
                    deviation of the background noise is determined after kappa,
                    sigma clipping subimages of appropriate size and
                    interpolating this across the image while including masks.
                    This is ImageData.rmsmap.data, i.e. without the mask, since
                    we assume that any mask has already been properly propagated
                    in selecting the islands pixels.

    :param analysisthresholddata: 2D ndarray of float32 or float64, same shape
                                  as data_bgsubbeddata. These are the data of
                                  the analysisthresholdmap, i.e.
                                  analysisthresholdmap.data, without the mask,
                                  since we assume that any mask has already
                                  been properly propagated in the process
                                  of selecting the island pixels.

    :param indices: 2D int32 ndarray of shape (num_islands, 4) representing
                    positions of the corners of the "chunks", i.e. slices
                    encompassing every island, relative to upper left corner
                    of the image, i.e. relative to array element [0,0] of
                    data_bgsubbeddata.

    :param labelled_data: 2D int32 ndarray of shape data_bgsubbeddata.shape,
                          representing all unmasked island pixels above the
                          analysis threshold, with labels (integers) > 0.
                          All background or masked pixels from
                          ImageData.data_bgsubbed should have value (label) = 0.

    :param labels: 1D int32 or int64 ndarray representing the labels in
                   labelled_data corresponding to islands in
                   data_bgsubbeddata that have peak values above the local
                   detection threshold. Thus, it is a subset of
                   np.unique(labelled_data) because the
                   'insignificant' islands have been filtered out, i.e.
                   the islands above the analysis threshold with peak
                   values below the (local) detection threshold. Also,
                   labels does not contain the zero label corresponding
                   to background or masked pixels in ImageData.data_bgsubbed.

    :param wcs:  A utility.coordinates.wcs instance, filled with appropriate
                 values from the image header, issued e.g. through
                 accessors.fitsimage.FitSImage. With numbers representing
                 the center of the image and the width of the pixels in
                 celestial coordinates plus the type of projection used,
                 we can transform from pixel coordinates to celestial
                 coordinates.

    :param fudge_max_pix_factor: float to correct for the underestimate of the
                 true peak when the maximum pixel method is used. It is a
                 statistical correction, so on average correct over a large
                 ensemble of unresolved sources, when a circular restoring beam
                 is appropriate.

    :param beam: tuple of 3 floats describing the restoring beam in terms of
                 its semi-major and semi-minor axes (in pixels) and the position
                 angle of the semi-major axis, east from local north, in
                 radians.

    :param beamsize: area (float) of the restoring beam, in pixels.

    :param correlation_lengths: tuple of 2 floats describing over which
             distance (in pixels) noise should be considered correlated, along
             both principal axes of the Gaussian profile of the restoring
             beam.

    :param eps_ra, eps_dec: floats (degrees) that reflect an extra positional
            uncertainty from calibration errors along right ascension and
            declination.

    :return:
             moments_of_sources: a 3D float32 ndarray, with a row index
             corresponding to each island, i.e. num_islands rows, a 0 or 1
             column index corresponding to values and their error bars,
             respectively and a third index [0...9] which represents the
             measurements of the sources, i.e. peak specific brightness,
             in Jy/beam and the integrated specific brightness a.k.a. flux
             density in Jy. All the other numbers in moments_of_sources are
             in pixel coordinates. These are position on the sky, along both
             axes, semi-major and semi-minor axis of the Gaussian profile of
             the source and the position angle of the major axis measured east
             from local north. These three Gaussian profile parameters can be
             deconvolved from the clean beam if the source is resolved. This
             adds up to ten quantities and corresponding error bars for each
             island.

             sky_barycenters: a (num_islands, 2) ndarray of floats: each row
                              has an entry for Right ascension (float64) and
                              Declination (float64).

             chunk_positions: a (num_islands, 2) ndarray of integers denoting
                              the indices that correspond to the upper left
                              corners of the islands. They are used in this
                              module and returned to the calling module.
                              However, their return is only useful if the user
                              requires a residual map.

             ra_errors, dec_errors: Both are 1D float64 ndarrays corresponding
                                    to the two columns in sky_barycenters.

             error_radii: a 1D float64 ndarray representing an absolute error
                          bar on the position of every source. This is an
                          important quantity to associate an observation of a
                          source with its previous observations.

             smaj_asec, errsmaj_asec, smin_asec, errsmin_asec: Four 1D float64
                          ndarrays representing the convolved semi-major and
                          semi-minor axes - i.e. not deconvolved from the
                          restoring beam - transformed from pixel coordinates
                          to celestial coordinates and their 1-sigma error bars.

             theta_celes_values and theta_celes_errors:  Both are 1D float32
                          ndarrays representing the position angle of the
                          semi-major axis of the convolved Gaussian profile,
                          i.e. not yet deconvolved from the restoring beam and
                          its estimated 1-sigma error bar.

             theta_dc_celes_values and theta_dc_celes_errors: Both are 1D
                          float32 ndarrays representing the position angle of
                          the semi-major axis of the deconvolved Gaussian
                          profile, i.e. deconvolved from the restoring beam and
                          its estimated 1-sigma error bar. The function we are
                          describing here is mimicking the Detection class as
                          closely as possible, including the
                          ._physical_coordinates function, that is why we are
                          including these two quantities. Oddly enough, these
                          are not propagated into the serialize function, which
                          aggregates quantities for database storage, nor are
                          the deconvolved semi-major and semi-minor axes
                          transformed to celestial coordinates. It would be
                          straightforward to calculate these and to include
                          them for database storage, together with
                          theta_dc_celes_values and theta_dc_celes_errors. A
                          reason not to pass them on for database storage may
                          be that most sources are unresolved, so the
                          deconvolved source shape parameters will be nans.

        Gaussian_islands_map (np.ndarray): A 2D np.float32 ndarray
            with the same shape as data_bgsubbed from the ImageData class
            instantiation. The derived Gaussian parameters have been used to
            reconstruct Gaussian profiles at the pixel positions of the islands
            and zero outside those islands. This array can be saved e.g. as a
            FITS image for inspection of the source processing.

        Gaussian_residuals_map (np.ndarray): Similar to Gaussian_islands_map,
        but as residuals, i.e. the Gaussian islands have been subtracted from
        the image data at the pixel positions of the islands - and zero outside
        the islands. This array can be saved e.g. as a FITS image for
        inspection of the source processing.

        sig(ndarray): np.float32 numbers indicating the significances of the
            detections. Often this will be the ratio of the peak spectral
            brightness of the source island divided by the noise at the position
            of that peak.  But for extended sources, the noise can perhaps
            decrease away from the position of the peak spectral brightness more
            steeply than the source spectral brightness and the maximum
            signal-to-noise ratio can be found at a different position.

        chisq(ndarray): np.float32 numbers representing chi-squared statistics
            reflecting an indication of the goodness-of-fit, equivalent to chisq
            as calculated in measuring.goodness_of_fit.

        reduced_chisq(ndarray): np.float32 numbers representing reduced
            chi-squared statistics reflecting an indication of the
            goodness-of-fit, equivalent to reduced_chisq as calculated in
            measuring.goodness_of_fit.

    """
    # This is the conditional route to the fastest algorithm for source
    # measurements, with no forced_beam and deblending options and no
    # Gauss fitting, so a coarser way of measuring sources.

    # Make 2D input data suitable for moments_enhanced with guvectorize
    # decorator, Each row will contain the pixel values of a flattened source.
    # First, determine the correct dimensions for the input
    # The number of rows will be num_labels = number of islands
    # (sources).
    # The number of columns will be max_pixels = the maximum number
    # of pixels an island can have.
    # We will also be needing auxiliary arrays.
    # Two of those will be index arrays, xpositions and ypositions.
    # These arrays will have shape (num_labels, max_pixels).
    # I.e. these are 2D arrays of integers, indicating the relative
    # positions of the source pixel relative to the top left corner of the slice
    # encompassing the island.
    # Later we will be needing another 2D array of floats with a number
    # of quantities related to sky position.

    max_pixels = npixs.max()

    sources = np.empty((num_islands, max_pixels), dtype=np.float32)
    noises = np.empty_like(sources)

    # xpositions and ypositions are relative to the upper left corner of
    # the chunk.
    xpositions = np.empty((num_islands, max_pixels), dtype=np.int32)
    ypositions = np.empty((num_islands, max_pixels), dtype=np.int32)

    # It makes sense to calculate the minimum width of each island of pixels
    # upfront such that we are not trying to estimate any Gaussian parameters,
    # either through moments estimation or through fitting, when the island has
    # a width of less than 3 pixels along any axis. For these "thin" detections
    # we can still determine the peak spectral brightness and position, but the
    # other four Gaussian parameters, i.e. the flux density, the axes and the
    # position angle will have to be derived with the help of the clean beam
    # parameters, i.e. by assuming the source is unresolved.
    minimum_widths = np.empty(num_islands, dtype=np.int32)

    insert_sources_and_noise(
        data_bgsubbeddata.astype(np.float32, copy=False),
        rmsdata.astype(np.float32, copy=False), indices, labelled_data,
        labels.astype(np.int32, copy=False), npixs, sources, noises,
        xpositions, ypositions, minimum_widths)

    # Make sure we get a single precision array of floats, which
    # measuring.moments_enhanced expects.
    thresholds = analysisthresholddata[maxposs[:, 0], maxposs[:, 1]].astype(
                                      np.float32, copy=False)

    local_noise_levels = rmsdata[maxposs[:, 0], maxposs[:, 1]].astype(
                                 np.float32, copy=False)

    # In order to convert to celestial coordinates, at a later stage, we
    # need to keep a record of the positions of the upper left corners
    # of the chunks. moments_enhanced starts by calculating xbar and
    # ybar as if those upper left corners have indices [0, 0] in the
    # image. We can add the indices of the chunks as arguments of
    # moments_enhanced to correct for that.
    chunk_positions = np.empty((num_islands, 2), dtype=np.int32)
    chunk_positions[:, 0] = indices[:, 0]
    chunk_positions[:, 1] = indices[:, 2]

    # The result will be put in an array 'moments_of_sources' containing
    # ten quantities and their uncertainties: peak flux density,
    # integrated flux, xbar, ybar, semi-major axis, semi-minor axis,
    # gaussian position angle and the deconvolved equivalents of the latter
    # three quantities.
    moments_of_sources = np.empty((num_islands, 2, 10),
                                     dtype=np.float32)
    dummy = np.empty_like(moments_of_sources)

    # sig is to be filled with the significances of each detection, i.e. the
    # maximum signal-to-noise ratio across all island pixels, for each island.
    sig = np.empty(num_islands, dtype=np.float32)
    chisq = np.empty_like(sig)
    reduced_chisq = np.empty_like(sig)

    Gaussian_islands = np.zeros_like(data_bgsubbeddata)
    Gaussian_residuals = np.zeros_like(Gaussian_islands)

    # This is a workaround for an unresolved issue:
    # https://github.com/numba/numba/issues/6690
    # The output shape can apparently not be set as fixed numbers.
    # So we will add a dummy array with shape corresponding
    # to the output array (moments_of_sources), as (useless) input
    # array. In this way Numba can infer the shape of the output array.
    measuring.moments_enhanced(sources, noises, chunk_positions, xpositions,
                             ypositions, minimum_widths, npixs, thresholds,
                             local_noise_levels, maxposs, maxis,
                             fudge_max_pix_factor, np.array(beam), beamsize,
                             np.array(correlation_lengths), 0, 0,
                             Gaussian_islands, Gaussian_residuals, dummy,
                             moments_of_sources, sig, chisq, reduced_chisq)

    barycentric_pixel_positions = moments_of_sources[:, 0, 2:4]
    # Convert the barycentric positions to celestial_coordinates.
    sky_barycenters = wcs.all_p2s(barycentric_pixel_positions)
    # We need to determine the orientation of the y-axis wrt local north
    # by incrementing y by a small amount and converting that
    # to celestial coordinates. That small increment is conveniently
    # chosen to be an increment of 1 pixel.
    endy_barycentric_positions = barycentric_pixel_positions.copy()
    endy_barycentric_positions[:, 1] += 1
    endy_sky_coordinates = wcs.all_p2s(endy_barycentric_positions)

    input_for_second_part = \
        np.empty((num_islands, 11), dtype=np.float32)
    # Unfortunately, the use of the guvectorize decorator again
    # requires a dummy input with the same shape as the output,
    # such that Numba can infer the shape of the output array.
    dummy = np.empty_like(input_for_second_part)

    first_part_of_celestial_coordinates(sky_barycenters,
                                        endy_sky_coordinates,
                                        moments_of_sources[:, 1, 2:4],
                                        moments_of_sources[:, 0, 2:7],
                                        dummy, input_for_second_part)

    # Derive an absolute angular error on position, as
    # utils.get_error_radius does, but less involved.
    # Simply derive the angle between the celestial positions
    # corresponding to [xbar, ybar] and [xbar + errorx,
    # ybar + errory], that should suffice.
    error_radii = np.empty(num_islands, dtype=np.float64)
    try:
        # Compute sky positions corresponding to [x_bar + x_error,
        #                                         y_bar + y_error]
        pix_offset = moments_of_sources[:, 0, 2:4] + \
                     moments_of_sources[:, 1, 2:4]
        sky_offset = wcs.all_p2s(pix_offset)
        coordinates.angsep_vectorized(sky_barycenters, sky_offset,
                                      error_radii)

    except RuntimeError:
        # Mimic error handling from utils.get_error_radius.
        # The downside is that all error radii will be infinite
        # also when only the calculation of the error radius for a
        # single source gives a RuntimeError.
        error_radii.fill(float('inf'))

    # Now we have to sort out which combination of errorx_proj and
    # errory_proj gives the largest errors in RA and Dec.
    try:
        # Derive end_ra1, end_dec1, end_ra2, end_dec2 as in
        # extract.Detection._physical_coordinates.
        # We need to add errorx_proj to xbar and zero to ybar.
        # So we need to extract the first column from
        # input_for_second_part and append a column with zeros
        # in order to perform the addition.
        helper1 = input_for_second_part[:, :1]
        errorx_proj_and_zeros = np.hstack((helper1,
                                              np.zeros((helper1.shape[0], 1),
                                                          dtype=helper1.dtype)))
        pix_x_plus_errorx_proj = (moments_of_sources[:, 0, 2:4] +
                                  errorx_proj_and_zeros)
        end_ra1_end_dec1 = wcs.all_p2s(pix_x_plus_errorx_proj)

        # Now we need to add errory_proj to ybar and zero to xbar.
        # So we need to extract the second column from
        # input_for_second_part and prepend a column with zeros
        # in order to perform the addition.
        helper2 = input_for_second_part[:, 1:2]
        zeros_and_errory_proj = np.hstack((
            np.zeros((helper2.shape[0], 1), dtype=helper2.dtype),
            helper2))
        pix_y_plus_errory_proj = moments_of_sources[:, 0, 2:4] + \
            zeros_and_errory_proj

        end_ra2_end_dec2 = wcs.all_p2s(pix_y_plus_errory_proj)

        # Here we include the position calibration errors
        ra_errors = eps_ra + np.maximum(
            np.fabs(sky_barycenters[:, :1] - end_ra1_end_dec1[:, :1]),
            np.fabs(sky_barycenters[:, :1] - end_ra2_end_dec2[:, :1]))

        dec_errors = eps_dec + np.maximum(
            np.fabs(sky_barycenters[:, 1:2] - end_ra1_end_dec1[:, 1:2]),
            np.fabs(sky_barycenters[:, 1:2] - end_ra2_end_dec2[:, 1:2]))
    except RuntimeError:
        # We get a runtime error from wcs.all_p2s if the errors place the
        # limits outside the image.
        # In which case we set the RA / DEC uncertainties to infinity.
        # The downside of this vectorized approach is that the position
        # errors for all the sources will be set to infinity.
        ra_errors = np.empty(num_islands).fill(np.inf)
        dec_errors = np.empty(num_islands).fill(np.inf)

    theta_celes_values = (np.degrees(moments_of_sources[:, 0, 6:7]) +
                          input_for_second_part[:, 2:3]) % 180

    theta_celes_errors = np.degrees(moments_of_sources[:, 1, 6:7])

    # This should also work for any nan value of theta_dc.
    theta_dc_celes_values = \
        (np.degrees(moments_of_sources[:, 0, 9:10]) +
         input_for_second_part[:, 2:3]) % 180

    theta_dc_celes_errors = np.degrees(moments_of_sources[:, 1, 9:10])

    try:
        end_smaj_ra_dec = wcs.all_p2s(input_for_second_part[:, [3, 5]])
        end_smin_ra_dec = wcs.all_p2s(input_for_second_part[:, [7, 9]])

    except RuntimeError:
        # Here we are again facing a downside of the vectorized
        # approach: if the FWHM ends of any source give a problem
        # wrt to pixel to sky conversion, all sky coordinates have to
        # be set to nans.
        logger.debug("pixel_to_spatial failed")
        end_smaj_ra_dec = \
            np.empty((num_islands, 2)).fill(np.nan)
        end_smin_ra_dec = \
            np.empty((num_islands, 2)).fill(np.nan)

    smaj_asec = np.empty(num_islands, dtype=np.float64)
    coordinates.angsep_vectorized(sky_barycenters, end_smaj_ra_dec,
                                  smaj_asec)

    scaling_smaj = smaj_asec / moments_of_sources[:, 0, 4]

    errsmaj_asec = scaling_smaj * moments_of_sources[:, 1, 4]

    smin_asec = np.empty(num_islands, dtype=np.float64)
    coordinates.angsep_vectorized(sky_barycenters, end_smin_ra_dec,
                                  smin_asec)

    scaling_smin = smin_asec / moments_of_sources[:, 0, 5]

    errsmin_asec = scaling_smin * moments_of_sources[:, 1, 5]

    return (moments_of_sources, sky_barycenters, ra_errors, dec_errors,
            error_radii, smaj_asec, errsmaj_asec, smin_asec, errsmin_asec,
            theta_celes_values, theta_celes_errors, theta_dc_celes_values,
            theta_dc_celes_errors, Gaussian_islands, Gaussian_residuals,
            sig, chisq, reduced_chisq)
