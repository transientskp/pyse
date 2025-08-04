"""
Mock / synthetic data objects for use in testing.
"""
import numpy as np
from sourcefinder.accessors.dataaccessor import DataAccessor
from sourcefinder.utility.coordinates import WCS
import datetime

class Mock(object):
    def __init__(self, returnvalue=None):
        self.callcount = 0
        self.callvalues = []
        self.returnvalue = returnvalue

    def __call__(self, *args, **kwargs):
        self.callcount += 1
        self.callvalues.append((args, kwargs))
        return self.returnvalue


def make_wcs(crval=None,
             cdelt=None,
             crpix=None
             ):
    """
    Make a WCS object for insertion into a synthetic image.
    
    Args:
        crval : tuple, default : None
            Tuple of (RA, Dec) in decimal degrees at the reference position.
        crpix : tuple, default : None
            Tuple of (x, y) coordinates describing the reference pixel location
            corresponding to the crval sky-position.
        cdelt : tuple, default : None
            Tuple of (cdelt0, cdelt1) in decimal degrees. This is the pixel 
            width in degrees of arc, but not necessarily aligned to RA, 
            Dec unless `crota` is (0, 0). If that *is* the case, 
            then typically cdelt0 is negative since the x-axis is in the 
            direction of West (decreasing RA).
    """
    # For any arguments not set we simply assign an arbitrary valid value:
    if crval is None:
        crval = 100., 45.
    if cdelt is None:
        pixel_width_arcsec = 40
        pixel_width_deg = pixel_width_arcsec / 3600.
        cdelt = (-pixel_width_deg, pixel_width_deg)
    if crpix is None:
        crpix = (256.0, 256.0)
    wcs = WCS()
    wcs.cdelt = cdelt
    wcs.crota = (0.0, 0.0)
    wcs.crpix = crpix
    wcs.crval = crval
    wcs.ctype = ('RA---SIN', 'DEC--SIN')
    wcs.cunit = ('deg', 'deg')
    return wcs


class SyntheticImage(DataAccessor):
    def __init__(self,
                 wcs=None,
                 data=None,
                 beam=(1.5,1.5,0),
                 freq_eff=150e6,
                 freq_bw=2e6,
                 tau_time=1800,
                 taustart_ts=datetime.datetime(2015,1,1)
                 ):
        """Generate a synthetic image for use in tests.

        Parameters
        ----------
        wcs : sourcefinder.utility.coordinates.WCS, default: None
            WCS instance for the image.
        data : array_like, default: None
            Data for the image. Default is a 512x512 array of zeroes.
        beam : tuple, default: (1.5, 1.5, 0)
            Beam semi-major axis (in pixels), semi-minor axis (in pixels),
            and position angle (in radians).
        freq_eff : float, default: 150e6
            Effective frequency of the image in Hz. This is the mean frequency
            of all the visibility data which comprises this image.
        freq_bw : float, default: 2e6
            The frequency bandwidth of this image in Hz.
        tau_time : float, default: 1800
            Total time on sky in seconds.
        taustart_ts : datetime.datetime, default: datetime.datetime(2015, 1, 1)
            Timestamp of the first integration which constitutes part of this
            image.

        """
        self.url = "SyntheticImage"
        self.wcs = wcs
        if self.wcs is None:
            self.wcs = make_wcs()
        self.data = data
        if self.data is None:
            self.data = np.zeros((512,512))
        self.beam = beam
        self.freq_eff = freq_eff
        self.freq_bw = freq_bw
        self.tau_time = tau_time
        self.taustart_ts = taustart_ts

        self.pixelsize = self.parse_pixelsize()
        self.centre_ra, self.centre_decl = self.calculate_phase_centre()

    def calculate_phase_centre(self):
        """Calculate the equatorial coordinates of the center of the
        synthetic image, based on the image dimensions.

        Returns
        -------
        tuple
            A tuple containing the right ascension and declination of the
            image center in degrees.

        """
        x, y = self.data.shape
        centre_ra, centre_decl = self.wcs.p2s((x / 2, y / 2))
        return centre_ra, centre_decl
