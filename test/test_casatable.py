import os

import unittest

import pytest
import sourcefinder.accessors as accessors
from sourcefinder.testutil.decorators import requires_data
from .conftest import DATAPATH

casatable = os.path.join(DATAPATH, 'accessors/casa.table')


@requires_data(casatable)
class TestLofarCasaImage(unittest.TestCase):
    # CasaImages can't be directly instantiated, since they don't provide the
    # DataAccessor interface.
    def test_casaimage(self):
        self.assertRaises(OSError, accessors.open, casatable)
