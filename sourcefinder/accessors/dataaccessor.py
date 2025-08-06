import logging

import numpy as np
from math import degrees, sqrt, sin, pi, cos
from dataclasses import dataclass, field
from sourcefinder.utils import is_valid_beam_tuple
from sourcefinder.utility.coordinates import WCS
from sourcefinder.config import ImgConf
from typing import Optional, cast

logger = logging.getLogger(__name__)


@dataclass
class DataAccessor:
    """Base class for accessors used with
    :class:`sourcefinder.image.ImageData`.

    Data accessors provide a uniform way for the ImageData class (i.e.,
    generic image representation) to access the various ways in which
    images may be stored (FITS files, arrays in memory, potentially HDF5,
    etc.).

    This class cannot be instantiated directly, but should be subclassed
    and the abstract properties provided. Note that all abstract
    properties are required to provide a valid accessor.

    Additional properties may also be provided by subclasses. However,
    TraP components are required to degrade gracefully in the absence of
    these optional properties.

    Attributes
    ----------
    beam : tuple
        Restoring beam. Tuple of three floats: semi-major axis (in pixels),
        semi-minor axis (pixels), and position angle (radians).
    centre_ra : float
        Right ascension at the central pixel of the image. Units of J2000
        decimal degrees.
    centre_decl : float
        Declination at the central pixel of the image. Units of J2000
        decimal degrees.
    data : numpy.ndarray
        Two-dimensional numpy.ndarray of floating point pixel values.
    freq_bw : float
        The frequency bandwidth of this image in Hz.
    freq_eff : float
        Effective frequency of the image in Hz. That is, the mean frequency
        of all the visibility data which comprises this image.
    pixelsize : tuple
        (x, y) tuple representing the size of a pixel along each axis in
        units of degrees.
    tau_time : float
        Total time on sky in seconds.
    taustart_ts : float
        Timestamp of the first integration which constitutes part of this
        image. MJD in seconds.
    url : str
        A URL representing the location of the image at the time of
        processing.
    wcs : :class:`sourcefinder.utility.coordinates.WCS`
        An instance of :py:class:`sourcefinder.utility.coordinates.WCS`,
        describing the mapping from data pixels to sky-coordinates.

    Notes
    -----
    The class also provides some common functionality: static methods used
    for parsing data files, and an 'extract_metadata' function which
    provides key info in a simple dict format.

    """

    centre_ra: float
    centre_decl: float
    data: np.ndarray
    freq_bw: float
    freq_eff: float
    pixelsize: tuple
    tau_time: float
    taustart_ts: float
    url: str
    wcs: WCS
    beam: Optional[tuple[float, float, float]] = field(default=None)
    conf: Optional[ImgConf] = field(default=None, repr=False)

    def __post_init__(self):
        if self.conf is not None:
            beam_tuple = (self.conf.bmaj, self.conf.bmin, self.conf.bpa)
            if is_valid_beam_tuple(beam_tuple):
                deltax, deltay = self.pixelsize
                self.beam = DataAccessor.degrees2pixels(
                    *beam_tuple, deltax, deltay
                )

    def extract_metadata(self) -> dict:
        """Massage the class attributes into a flat dictionary with
        database-friendly values.

        While rather tedious, this is easy to serialize and store separately
        to the actual image data.

        May be extended by subclasses to return additional data.

        Returns
        -------
        dict
            A dictionary containing key-value pairs of class attributes
            formatted for database storage.

        """
        metadata = {
            "tau_time": self.tau_time,
            "freq_eff": self.freq_eff,
            "freq_bw": self.freq_bw,
            "taustart_ts": self.taustart_ts,
            "url": self.url,
            "centre_ra": self.centre_ra,
            "centre_decl": self.centre_decl,
            "deltax": self.pixelsize[0],
            "deltay": self.pixelsize[1],
        }

        if is_valid_beam_tuple(self.beam):
            beam = cast(tuple[float, float, float], self.beam)
            metadata.update(
                {
                    "beam_smaj_pix": beam[0],
                    "beam_smin_pix": beam[1],
                    "beam_pa_rad": beam[2],
                }
            )

        return metadata

    def parse_pixelsize(self) -> tuple[float, float]:
        """Parse pixel size.

        Returns
        -------
        deltax : float
            Pixel size along the x axis in degrees.
        deltay : float
            Pixel size along the y axis in degrees.

        """
        wcs = self.wcs
        # Check that pixels are square
        # (Would have to be pretty strange data for this not to be the case)
        assert wcs.cunit[0] == wcs.cunit[1]
        if wcs.cunit[0] == "deg":
            deltax = wcs.cdelt[0]
            deltay = wcs.cdelt[1]
        elif wcs.cunit[0] == "rad":
            deltax = degrees(wcs.cdelt[0])
            deltay = degrees(wcs.cdelt[1])
        else:
            raise ValueError("Unrecognised WCS co-ordinate system")

        # NB. What's a reasonable epsilon here?
        eps = 1e-7
        if abs(abs(deltax) - abs(deltay)) > eps:
            raise ValueError(
                "Image WCS header suggests non-square pixels."
                "This is an untested use case, and may break "
                "things - specifically the skyregion tracking "
                "but possibly other stuff too."
            )
        return deltax, deltay

    @staticmethod
    def degrees2pixels(
        bmaj, bmin, bpa, deltax, deltay
    ) -> tuple[float, float, float]:
        """Convert beam in degrees to beam in pixels and radians.
        For example, FITS beam parameters are in degrees.

        Parameters
        ----------
        bmaj : float
            Beam major axis in degrees.
        bmin : float
            Beam minor axis in degrees.
        bpa : float
            Beam position angle in degrees.
        deltax : float
            Pixel size along the x-axis in degrees.
        deltay : float
            Pixel size along the y-axis in degrees.

        Returns
        -------
        semimaj : float
            Beam semi-major axis in pixels.
        semimin : float
            Beam semi-minor axis in pixels.
        theta : float
            Beam position angle in radians.

        """
        theta = pi * bpa / 180
        semimaj = (bmaj / 2.0) * (
            sqrt(
                (sin(theta) ** 2) / (deltax**2)
                + (cos(theta) ** 2) / (deltay**2)
            )
        )
        semimin = (bmin / 2.0) * (
            sqrt(
                (cos(theta) ** 2) / (deltax**2)
                + (sin(theta) ** 2) / (deltay**2)
            )
        )
        return semimaj, semimin, theta
