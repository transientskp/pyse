import os
import warnings
import sourcefinder

HERE = os.path.dirname(__file__)
DEFAULT = os.path.abspath(os.path.join(HERE, '../../test/data'))
DATAPATH = os.environ.get('TKP_TESTPATH', DEFAULT)

if not os.access(DATAPATH, os.X_OK):
    warnings.warn("can't access " + DATAPATH)

# A arbitrary fits file which can be used for playing around
fits_file = os.path.join(DATAPATH, "NCP_sample_image_1.fits")

