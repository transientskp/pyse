import logging

import numpy

from sourcefinder.utils import is_valid_beam_tuple
from sourcefinder.accessors.fitsimage import FitsImage

logger = logging.getLogger(__name__)


class FitsImageBlob(FitsImage):
    """
    A FITS image blob. Same as ``sourcefinder.accessors.fitsimage.FitsImage``
    but constructed from an in-memory FITS file, not a FITS file on disk.

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList
        The HDU list representing the in-memory FITS file.
    plane : int, default: None
        If the data is a datacube, specifies which plane to use.
    beam : tuple, default: None
        Beam parameters in degrees, in the form (bmaj, bmin, bpa). If not 
        supplied, the method will attempt to read these from the header.
    hdu_index : int, default: 0
        The index of the HDU to use from the HDU list.
    """
    def __init__(self, hdulist, plane=None, beam=None, hdu_index=0):
        # Set the URL in case we need it during header parsing for error
        # logging.
        self.url = "AARTFAAC streaming image"

        self.header = self._get_header(hdulist, hdu_index)
        self.wcs = self.parse_coordinates()
        self.data = self.read_data(hdulist, hdu_index, plane)
        self.taustart_ts, self.tau_time = self.parse_times()
        self.freq_eff, self.freq_bw = self.parse_frequency()
        self.pixelsize = self.parse_pixelsize()

        elements = "memory://AARTFAAC", self.taustart_ts, self.tau_time, \
                   self.freq_eff, self.freq_bw
        self.url = "_".join([str(x) for x in elements])

        if is_valid_beam_tuple(beam) or not is_valid_beam_tuple(self.beam):
            # An argument-supplied beam overrides a beam derived from
            # (bmaj, bmin, bpa) in a config.toml. Only if those two options
            # are not specified, we parse the beam from the header.
            bmaj, bmin, bpa = beam if is_valid_beam_tuple(beam) else (
                self.parse_beam())
            self.beam = self.degrees2pixels(
                bmaj, bmin, bpa, self.pixelsize[0], self.pixelsize[1]
            )

        self.centre_ra, self.centre_decl = self.calculate_phase_centre()

        # Bonus attribute
        if 'TELESCOP' in self.header:
            self.telescope = self.header['TELESCOP']

    def _get_header(self, *args):
        """
        Retrieve the header from the specified HDU.

        Parameters
        ----------
        *args : tuple
            Positional arguments where:

            - args[0] is the HDU list (astropy.io.fits.HDUList).
            - args[1] is the index of the HDU to use.

        Returns
        -------
        astropy.io.fits.Header
            The header of the specified HDU.
        """
        hdulist = args[0]
        hdu_index = args[1]
        return hdulist[hdu_index].header

    def read_data(self, *args):
        """
        Read and process the data from the specified HDU.

        Parameters
        ----------
        *args : tuple
            Positional arguments where:

            - args[0] is the HDU list (astropy.io.fits.HDUList).
            - args[1] is the index of the HDU to use.
            - args[2] is the plane index (int) if the data is a datacube.

        Returns
        -------
        numpy.ndarray
            The processed 2D data array. Processing here means remove axes of
            length 1, select the plane index from the datacube if needed, and
            transpose.
        """
        hdulist = args[0]
        hdu_index = args[1]
        plane = args[2]
        hdu = hdulist[hdu_index]
        data = numpy.float64(hdu.data.squeeze())
        if plane is not None and len(data.shape) > 2:
            data = data[plane].squeeze()
        n_dim = len(data.shape)
        if n_dim != 2:
            logger.warning((
                "Loaded datacube with %s dimensions, assuming Stokes I and "
                "taking plane 0" % n_dim))
            data = data[0, :, :]
        data = data.transpose()
        return data
