"""
Tests for simulated LOFAR datasets.
"""

import os
import unittest

from sourcefinder import accessors
from sourcefinder.accessors.fitsimage import FitsImage
from sourcefinder.testutil.decorators import requires_data
from test.conftest import DATAPATH


class PyfitsFitsImage(unittest.TestCase):

    @requires_data(os.path.join(DATAPATH, "observed-all.fits"))
    @requires_data(os.path.join(DATAPATH, "correlated_noise.fits"))
    def testOpen(self):
        fits_file = os.path.join(DATAPATH, "observed-all.fits")
        image = FitsImage(fits_file, beam=(54.0 / 3600, 54.0 / 3600, 0.0))
        self.assertAlmostEqual(image.beam[0], 0.225)
        self.assertAlmostEqual(image.beam[1], 0.225)
        self.assertAlmostEqual(image.beam[2], 0.0)
        self.assertAlmostEqual(image.wcs.crval[0], 350.85)
        self.assertAlmostEqual(image.wcs.crval[1], 58.815)
        self.assertAlmostEqual(image.wcs.crpix[0], 1440.0)
        self.assertAlmostEqual(image.wcs.crpix[1], 1440.0)
        self.assertAlmostEqual(image.wcs.cdelt[0], -0.03333333)
        self.assertAlmostEqual(image.wcs.cdelt[1], 0.03333333)
        self.assertEqual(image.wcs.ctype[0], "RA---SIN")
        self.assertEqual(image.wcs.ctype[1], "DEC--SIN")
        # Beam included in image
        fits_file = os.path.join(DATAPATH, "correlated_noise.fits")
        image = FitsImage(fits_file)
        self.assertAlmostEqual(image.beam[0], 2.7977999)
        self.assertAlmostEqual(image.beam[1], 2.3396999)
        self.assertAlmostEqual(image.beam[2], -0.869173967)
        self.assertAlmostEqual(image.wcs.crval[0], 266.363244382)
        self.assertAlmostEqual(image.wcs.crval[1], -29.9529359725)
        self.assertAlmostEqual(image.wcs.crpix[0], 127.0)
        self.assertAlmostEqual(image.wcs.crpix[1], 128.0)
        self.assertAlmostEqual(image.wcs.cdelt[0], -0.003333333414)
        self.assertAlmostEqual(image.wcs.cdelt[1], 0.003333333414)
        self.assertEqual(image.wcs.ctype[0], "RA---SIN")
        self.assertEqual(image.wcs.ctype[1], "DEC--SIN")

    @requires_data(os.path.join(DATAPATH, "observed-all.fits"))
    def testSFImageFromFITS(self):
        fits_file = os.path.join(DATAPATH, "observed-all.fits")
        image = FitsImage(fits_file, beam=(54.0 / 3600, 54.0 / 3600, 0.0))
        sfimage = accessors.sourcefinder_image_from_accessor(image)


class TestFitsImage(unittest.TestCase):

    @requires_data(os.path.join(DATAPATH, "observed-all.fits"))
    @requires_data(os.path.join(DATAPATH, "correlated_noise.fits"))
    def testOpen(self):
        # Beam specified by user
        fits_file = os.path.join(DATAPATH, "observed-all.fits")
        image = FitsImage(fits_file, beam=(54.0 / 3600, 54.0 / 3600, 0.0))
        self.assertEqual(
            image.telescope, "LOFAR20"
        )  # God knows why it's 'LOFAR20'
        self.assertAlmostEqual(image.beam[0], 0.225)
        self.assertAlmostEqual(image.beam[1], 0.225)
        self.assertAlmostEqual(image.beam[2], 0.0)
        self.assertAlmostEqual(image.wcs.crval[0], 350.85)
        self.assertAlmostEqual(image.wcs.crval[1], 58.815)
        self.assertAlmostEqual(image.wcs.crpix[0], 1440.0)
        self.assertAlmostEqual(image.wcs.crpix[1], 1440.0)
        self.assertAlmostEqual(image.wcs.cdelt[0], -0.03333333)
        self.assertAlmostEqual(image.wcs.cdelt[1], 0.03333333)
        self.assertEqual(image.wcs.ctype[0], "RA---SIN")
        self.assertEqual(image.wcs.ctype[1], "DEC--SIN")

        # Beam included in image
        image = FitsImage(os.path.join(DATAPATH, "correlated_noise.fits"))
        self.assertAlmostEqual(image.beam[0], 2.7977999)
        self.assertAlmostEqual(image.beam[1], 2.3396999)
        self.assertAlmostEqual(image.beam[2], -0.869173967)
        self.assertAlmostEqual(image.wcs.crval[0], 266.363244382)
        self.assertAlmostEqual(image.wcs.crval[1], -29.9529359725)
        self.assertAlmostEqual(image.wcs.crpix[0], 127.0)
        self.assertAlmostEqual(image.wcs.crpix[1], 128.0)
        self.assertAlmostEqual(image.wcs.cdelt[0], -0.003333333414)
        self.assertAlmostEqual(image.wcs.cdelt[1], 0.003333333414)
        self.assertEqual(image.wcs.ctype[0], "RA---SIN")
        self.assertEqual(image.wcs.ctype[1], "DEC--SIN")

    @requires_data(os.path.join(DATAPATH, "observed-all.fits"))
    def testSFImageFromFITS(self):
        image = FitsImage(
            os.path.join(DATAPATH, "observed-all.fits"),
            beam=(54.0 / 3600, 54.0 / 3600, 0.0),
        )
        sfimage = accessors.sourcefinder_image_from_accessor(image)


class FrequencyInformation(unittest.TestCase):
    @requires_data(os.path.join(DATAPATH, "missing_metadata.fits"))
    def testFreqinfo(self):
        # Frequency information is required by the data accessor.
        self.assertRaises(
            TypeError,
            FitsImage,
            os.path.join(DATAPATH, "missing_metadata.fits"),
        )


if __name__ == "__main__":
    unittest.main()
