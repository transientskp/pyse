"""
Try the in memory fits stream Accessor
"""

import os
import unittest
from astropy.io.fits import open as fitsopen
from sourcefinder.accessors import open as tkpopen
from sourcefinder.testutil.decorators import requires_data
from sourcefinder.accessors.fitsimageblob import FitsImageBlob
from .conftest import DATAPATH

FITS_FILE = os.path.join(DATAPATH, 'accessors/aartfaac.fits')


@requires_data(FITS_FILE)
class PyfitsFitsImage(unittest.TestCase):

    def setUp(self):
        self.hudelist = fitsopen(FITS_FILE)

    @staticmethod
    def test_tkp_open():
        accessor = tkpopen(FITS_FILE)

    def test_fits_blob_accessor(self):
        accessor = FitsImageBlob(self.hudelist)
