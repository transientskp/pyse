"""
Tests for simulated LOFAR datasets.
"""

import gc
import os
import unittest

from sourcefinder.config import Conf, ImgConf

from .conftest import DATAPATH
from sourcefinder.testutil.decorators import requires_data

import sourcefinder.accessors.fitsimage
import sourcefinder.image as image
import sourcefinder.utility.coordinates as coords

# The simulation code causes a factor of 2 difference in the
# measured flux.
FUDGEFACTOR = 0.5

corrected_fits = os.path.join(DATAPATH, 'corrected-all.fits')
observed_fits = os.path.join(DATAPATH, 'observed-all.fits')
all_fits = os.path.join(DATAPATH, 'model-all.fits')


# This module appears to be the most memory-intensive of the TKP unit-tests
# Probably because a large (~30mb) image is being repeatedly loaded into the
# sourcefinder. This was intermittently breaking the Travis-CI system until
# the addition of the manual garbage-collection steps, which is why they are
# only used in this module.


class L15_12hConstObs(unittest.TestCase):
    # Single, constant 1 Jy source at centre of image.
    @requires_data(observed_fits)
    def setUp(self):
        # Beam here is derived from a Gaussian fit to the central (unresolved)
        # source.
        fitsfile = sourcefinder.accessors.fitsimage.FitsImage(observed_fits,
                                                              beam=(0.2299,
                                                                    0.1597,
                                                                    -23.87))
        self.image = image.ImageData(fitsfile.data, fitsfile.beam, fitsfile.wcs)
        self.results = self.image.extract(det=10, anl=3.0)

    def tearDown(self):
        del self.results
        del self.image
        gc.collect()

    @requires_data(observed_fits)
    def testNumSources(self):
        self.assertEqual(len(self.results), 1)

    @requires_data(observed_fits)
    def testSourceProperties(self):
        mysource = self.results.closest_to(1440, 1440)[0]
        self.assertAlmostEqual(mysource.peak, 1.0 * FUDGEFACTOR, 1)


class L15_12hConstCor(unittest.TestCase):
    # Cross shape of 5 sources, 2 degrees apart, at centre of image.
    def setUp(self):
        # Beam here is derived from a Gaussian fit to the central (unresolved)
        # source.
        fitsfile = sourcefinder.accessors.fitsimage.FitsImage(corrected_fits,
                                                              beam=(0.2299,
                                                                    0.1597,
                                                                    -23.87))
        self.image = image.ImageData(fitsfile.data, fitsfile.beam, fitsfile.wcs)
        self.results = self.image.extract(det=10.0, anl=3.0)

    def tearDown(self):
        del self.image
        del self.results
        gc.collect()

    @requires_data(corrected_fits)
    def testNumSources(self):
        self.assertEqual(len(self.results), 5)

    @requires_data(corrected_fits)
    def testBrightnesses(self):
        # All sources in this image are supposed to have the same peak spectral
        # brightness. But they don't, because the simulation is broken, so this
        # test checks they fall in a vaguely plausible range.
        for mysource in self.results:
            self.assertTrue(mysource.peak.value > 0.35)
            self.assertTrue(mysource.peak.value < 0.60)

    @requires_data(corrected_fits)
    def testSeparation(self):
        centre = self.results.closest_to(1440, 1440)[0]
        # How accurate should the '2 degrees' be?
        for mysource in filter(lambda src: src != centre, self.results):
            self.assertAlmostEqual(round(
                coords.angsep(centre.ra, centre.dec, mysource.ra,
                              mysource.dec) /
                60 ** 2), 2)


class L15_12hConstMod(unittest.TestCase):
    # 1 Jy constant source at centre; 1 Jy (peak) transient 3 degrees away.
    def setUp(self):
        # This image is of the whole sequence, so obviously we won't see the
        # transient varying. In fact, due to a glitch in the simulation
        # process, it will appear smeared out & shouldn't be identified at
        # all.
        # Beam here is derived from a Gaussian fit to the central (unresolved)
        # source.
        fitsfile = sourcefinder.accessors.fitsimage.FitsImage(all_fits,
                                                              beam=(0.2299,
                                                                    0.1597,
                                                                    -23.87))
        self.image = image.ImageData(
            fitsfile.data, fitsfile.beam, fitsfile.wcs, Conf(ImgConf(radius=100), {})
        )
        self.results = self.image.extract(det=5, anl=3.0)

    def tearDown(self):
        del (self.results)
        del (self.image)
        gc.collect()

    @requires_data(all_fits)
    def testNumSources(self):
        self.assertEqual(len(self.results), 1)

    @requires_data(all_fits)
    def testFluxes(self):

        #self.results.sort(lambda x, y: (y.peak > x.peak) - (y.peak < x.peak))
        self.results.sort(key=lambda x: x.peak)
        self.assertAlmostEqual(self.results[0].peak.value, 1.0 * FUDGEFACTOR, 1)


class FitToPointTestCase(unittest.TestCase):
    def setUp(self):
        # FWHM of PSF taken from fit to unresolved source.
        fitsfile = sourcefinder.accessors.fitsimage.FitsImage(corrected_fits,
                                                              beam=(2. * 500.099 / 3600,
                                                                    2. * 319.482 / 3600,
                                                                    168.676))
        self.my_im = image.ImageData(fitsfile.data, fitsfile.beam,
                                     fitsfile.wcs)

    def tearDown(self):
        del self.my_im
        gc.collect()

    @requires_data(corrected_fits)
    def testFixed(self):
        d = self.my_im.fit_to_point(1379.00938273, 1438.38801493, 20,
                                    threshold=2, fixed='position')
        self.assertAlmostEqual(d.x.value, 1379.00938273)
        self.assertAlmostEqual(d.y.value, 1438.38801493)

    @requires_data(corrected_fits)
    def testUnFixed(self):
        d = self.my_im.fit_to_point(1379.00938273, 1438.38801493, 20,
                                    threshold=2, fixed=None)
        self.assertAlmostEqual(d.x.value, 1379.00938273, 0)
        self.assertAlmostEqual(d.y.value, 1438.38801493, 0)


if __name__ == '__main__':
    unittest.main()
