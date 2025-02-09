import math
import unittest

from sourcefinder.extract import Detection
from sourcefinder.extract import ParamSet
from sourcefinder.utility.coordinates import WCS
from sourcefinder.utility.uncertain import Uncertain
from sourcefinder.utils import calculate_correlation_lengths


class DummyImage(object):
    pass


def get_paramset():
    paramset = ParamSet()
    paramset.sig = 1
    paramset.measurements = {
        'peak': Uncertain(10, 0),
        'flux': Uncertain(10, 0),
        'semimajor': Uncertain(10, 1),
        'semiminor': Uncertain(10, 1),
        'theta': Uncertain(0, 1),
        'semimaj_deconv': Uncertain(10, 1),
        'semimin_deconv': Uncertain(10, 1),
        'theta_deconv': Uncertain(10, 1),
        'xbar': Uncertain(10, 1),
        'ybar': Uncertain(10, 1)
    }
    return paramset


class TestFluxErrors(unittest.TestCase):
    # We should never return a negative uncertainty
    def setUp(self):
        self.p = get_paramset()
        self.beam = (1.0, 1.0, 0.0)
        self.noise = 1.0  # RMS at position of source
        self.threshold = 3.0  # significance * rms at position of source
        self.correlation_lengths = calculate_correlation_lengths(self.beam[0], self.beam[1])

    def test_positive_flux_condon(self):
        self.p._condon_formulae(self.noise, self.correlation_lengths)
        self.assertGreaterEqual(self.p['peak'].error, 0)
        self.assertGreaterEqual(self.p['flux'].error, 0)

    def test_positive_flux_moments(self):
        self.p._error_bars_from_moments(self.noise, self.correlation_lengths,
                                        self.threshold)
        self.assertGreaterEqual(self.p['peak'].error, 0)
        self.assertGreaterEqual(self.p['flux'].error, 0)

    def test_negative_flux_condon(self):
        self.p['peak'] *= -1
        self.p['flux'] *= -1
        self.p._condon_formulae(self.noise, self.correlation_lengths)
        self.assertGreaterEqual(self.p['peak'].error, 0)
        self.assertGreaterEqual(self.p['flux'].error, 0)

    def test_negative_flux_moments(self):
        # A negative peak spectral brightness from moments computations implies
        # serious problems with the detectionthresholdmap and, in turn, with the
        # rms noisemap: it must have negative values, which is physically
        # impossible. Cancel any further processing of the image.
        self.p['peak'] *= -1
        self.assertRaises(ValueError, self.p._error_bars_from_moments,
                          self.noise, self.correlation_lengths, self.threshold)


class TestPositionErrors(unittest.TestCase):
    def setUp(self):
        # An image pointing at the north celestial pole
        self.ncp_image = DummyImage()
        self.ncp_image.wcs = WCS()
        self.ncp_image.wcs.cdelt = (-0.009722222222222, 0.009722222222222)
        self.ncp_image.wcs.crota = (0.0, 0.0)
        self.ncp_image.wcs.crpix = (1025, 1025)  # Pixel position of reference
        self.ncp_image.wcs.crval = (
        15.0, 90.0)  # Celestial position of reference
        self.ncp_image.wcs.ctype = ('RA---SIN', 'DEC--SIN')
        self.ncp_image.wcs.cunit = ('deg', 'deg')

        # And an otherwise identical one at the equator
        self.equator_image = DummyImage()
        self.equator_image.wcs = WCS()
        self.equator_image.wcs.cdelt = self.ncp_image.wcs.cdelt
        self.equator_image.wcs.crota = self.ncp_image.wcs.crota
        self.equator_image.wcs.crpix = self.ncp_image.wcs.crpix
        self.equator_image.wcs.crval = (
        15.0, 0.0)  # Celestial position of reference
        self.equator_image.wcs.ctype = self.ncp_image.wcs.ctype
        self.equator_image.wcs.cunit = self.ncp_image.wcs.cunit

        self.p = get_paramset()

    def test_ra_error_scaling(self):
        # Demonstrate that RA errors scale with declination.
        # See HipChat discussion of 2013-08-28.
        # Values of all parameters are dummies except for the pixel position.
        # First, construct a source at the NCP
        self.p.measurements['xbar'] = Uncertain(1025, 1)
        self.p.measurements['ybar'] = Uncertain(1025, 1)
        d_ncp = Detection(self.p, self.ncp_image)

        # Then construct a source somewhere away from the NCP
        self.p.measurements['xbar'] = Uncertain(125, 1)
        self.p.measurements['ybar'] = Uncertain(125, 1)
        d_not_ncp = Detection(self.p, self.ncp_image)

        # One source is at higher declination
        self.assertGreater(d_ncp.dec.value, d_not_ncp.dec.value)

        # The RA error at higher declination must be greater
        self.assertGreater(d_ncp.ra.error, d_not_ncp.ra.error)

    def test_error_radius_value(self):
        # Demonstrate that we calculate the expected value for the error
        # radius
        self.p.measurements['xbar'] = Uncertain(1025, 1)
        self.p.measurements['ybar'] = Uncertain(1025, 1)
        d_ncp = Detection(self.p, self.ncp_image)

        # cdelt gives the per-pixel increment along the axis in degrees
        # Detection.error_radius is in arcsec
        expected_error_radius = math.sqrt(
            (self.p.measurements['xbar'].error * self.ncp_image.wcs.cdelt[
                0] * 3600) ** 2 +
            (self.p.measurements['ybar'].error * self.ncp_image.wcs.cdelt[
                1] * 3600) ** 2
        )

        self.assertAlmostEqual(d_ncp.error_radius, expected_error_radius, 6)

    def test_error_radius_with_dec(self):
        self.p.measurements['xbar'] = Uncertain(1025, 1)
        self.p.measurements['ybar'] = Uncertain(1025, 1)
        d_ncp = Detection(self.p, self.ncp_image)
        d_equator = Detection(self.p, self.equator_image)
        self.assertEqual(d_ncp.error_radius, d_equator.error_radius)
