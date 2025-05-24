"""
This module implements the CASA kat7 data container format.
"""

import logging

from casacore.tables import table as casacore_table

from sourcefinder.accessors.casaimage import CasaImage

logger = logging.getLogger(__name__)


class AmiCasaImage(CasaImage):
    """
    Use casacore to pull image data out of a CASA table as produced by AMI-LA.

    Parameters
    ----------
    url : str
        Location of the CASA table.
    plane : int, default: 0
        If the data is a cube, specifies which plane to use.
    beam : tuple, default: None
        Beam parameters in degrees, in the form (bmaj, bmin, bpa). If not
        supplied, an attempt is made to read them from the header.

    Notes
    -----
    - AMI-LA does not currently include image duration in its headers, so a
      placeholder value of 1 is used.
    - The start time is taken from the CASA coords record and may not be valid
      if the image is composed of multiple observations.
    """

    def __init__(self, url, plane=0, beam=None):
        super().__init__(url, plane, beam)
        table = casacore_table(self.url, ack=False)
        self.taustart_ts = self.parse_taustartts(table)
        self.tau_time = 1  # Placeholder value until properly implemented
