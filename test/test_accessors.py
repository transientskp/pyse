import os.path

from numpy.testing import assert_almost_equal
import pytest

from sourcefinder import accessors

from .conftest import DATAPATH

lofar_casatable = os.path.join(
    DATAPATH,
    ("casatable/L55596_000TO009_skymodellsc_wmax6000_noise_mult10_cell40_" +
     "npix512_wplanes215.img.restored.corr"),
)


@pytest.mark.skipif(not os.path.exists(lofar_casatable),
                    reason="data file not in repo")
def test_no_injection():
    original_ms = accessors.open(lofar_casatable)
    assert_almost_equal(original_ms.tau_time, 58141509)
