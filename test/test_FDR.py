"""
Here we test the FDR algorithm using two maps with pure Gaussian noise.

In one map the pixels are not spatially correlated, in the other one they
are.  In fact, the second map has been made by convolving a map similar to
the first with the dirty beam of a certain VLA observation.

Analysis shows that the pixel values of the convolved map do not follow an
exact Gaussian distribution in the sense that the number of extreme pixel
values, based on some reasonable estimate of the number of independent
pixels, i.e., based on some reasonable estimate of the correlation length,
exceeds the expected number from Gaussian statistics.  Nevertheless, also
for the convolved image, image.fd_extract will not find any sources for any
reasonable value of alpha.

A similar conclusion was drawn in paragraph 3.8 of Spreeuw's thesis although
these maps were made in a slightly different manner, i.e., by adding Gaussian
noise to the visibilities and, subsequently, an FFT.  Here the Gaussian noise
in an image was convolved with a dirty beam, although FFTs were used to speed
things up (I used scipy.signal.fftconvolve).  We adjusted the header of
sourcefinder/simulations/uncorrelated_noise.fits, by adding values for BMAJ
and BMIN (and BPA, but that is redundant) to make sure that 0.25 * pi* BMAJ *
BMIN = -CDELT1 * CDELT2, i.e., that the correlated area (with the default
equations from config.py) equals the area of exactly one pixel.

Strictly speaking the FDR algorithm applies to the number of falsely
detected pixels as a fraction of all detected pixels.  in the presence of
uncorrelated noise. The algorithm has been modified somewhat to apply it to
correlated noise, but there is no rigorous statistical proof, see Hopkins et
al. (2002), AJ 123, 1086, paragraph 3.1.  Also, it should be noted that the
validity of the FDR algorithm refers to large ensembles.  This means that in
indivual maps the fraction of falsely detected pixels can exceed the
threshold (alpha).  For these unit tests, we'll be bold and use the number
of detected sources in the presence of correlated noise in a single map
(TEST_DECONV.FITS).
"""

import os
import unittest

from sourcefinder import accessors
from .conftest import DATAPATH
from sourcefinder.testutil.decorators import requires_data, duration

from sourcefinder import image

NUMBER_INSERTED = float(3969)

uncorr_path = os.path.join(DATAPATH, 'uncorrelated_noise.fits')
corr_path = os.path.join(DATAPATH, 'correlated_noise.fits')
deconv_path = os.path.join(DATAPATH, 'deconvolved.fits')


@requires_data(uncorr_path)
@requires_data(corr_path)
@requires_data(deconv_path)
@duration(100)
class test_maps(unittest.TestCase):
    def setUp(self):
        uncorr_map = accessors.open(uncorr_path)
        corr_map = accessors.open(corr_path)
        map_with_sources = accessors.open(deconv_path)

        # The FDR algorithm computes a threshold level as a factor for multiplying
        # the rms noise background map. This multiplication factor is the same for
        # the entire image.
        # The 'deconvolved.fits' image comes from 'correlated_noise.fits' with
        # sources inserted. The average noise in 'correlated_noise.fits' is 5.3 Jy,
        # but if you compute the rms noise on 32 * 32 subimages, the grid values range
        # between 2.4 and 11.1. It is not clear if FDR is applicable if the image is
        # split up into subimages, all with their local noise levels.
        # For now I set it back_size_x and back_size_y to 2048. This gives a uniform noise
        # level of 6.1 Jy. Not very close to the ground truth of 5.3 Jy, but no spatial
        # variation as for the 32 by 32 subimage sizes.
        # FDR may still be applicable, but I was unable to find any research on this.
        # Other aspects include non-normal aspects of the distribution of the correlated
        # noise, see paragraphs 3.6, 3.8 and the last column of table 3.4 of my thesis.
        # The last aspect is that the FDR algorithm holds for an average over a large
        # ensemble of images. So for a single image it may fail. Lastly, FDR applies to
        # pixels, not to sources.
        # These are all arguments to make unit tests for our FDR implementation a bit
        # less strict as coded previously.
        self.uncorr_image = image.ImageData(uncorr_map.data, uncorr_map.beam,
                                            uncorr_map.wcs,
                                            back_size_x = 2048, back_size_y = 2048)
        self.corr_image = image.ImageData(corr_map.data, uncorr_map.beam,
                                          uncorr_map.wcs,
                                          back_size_x = 2048, back_size_y = 2048)
        self.image_with_sources = image.ImageData(map_with_sources.data,
                                                  map_with_sources.beam,
                                                  map_with_sources.wcs,
                                                  back_size_x=2048, back_size_y=2048)

    def test_normal(self):
        self.number_detections_uncorr = len(self.uncorr_image.fd_extract(1e-2))
        self.number_detections_corr = len(self.corr_image.fd_extract(1e-2))
        self.assertEqual(self.number_detections_uncorr, 0)
        self.assertEqual(self.number_detections_corr, 0)

    def test_alpha0_1(self):
        self.number_alpha_10pc = len(
            self.image_with_sources.fd_extract(alpha=0.1))
        self.assertTrue((self.number_alpha_10pc - NUMBER_INSERTED) /
                        NUMBER_INSERTED < 0.1)

    def test_alpha0_01(self):
        self.number_alpha_1pc = len(
            self.image_with_sources.fd_extract(alpha=0.01))
        self.assertTrue((self.number_alpha_1pc - NUMBER_INSERTED) /
                        NUMBER_INSERTED < 0.01)

    def test_alpha0_001(self):
        self.number_alpha_point1pc = len(
            self.image_with_sources.fd_extract(alpha=0.001))
        self.assertTrue((self.number_alpha_point1pc - NUMBER_INSERTED) /
                        NUMBER_INSERTED < 0.001)


if __name__ == '__main__':
    unittest.main()
