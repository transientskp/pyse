import numpy
import pytz
import datetime
import dateutil.parser
import logging
import re

import astropy.io.fits as pyfits

from sourcefinder.utils import is_valid_beam_tuple
from sourcefinder.accessors.dataaccessor import DataAccessor
from sourcefinder.utility.coordinates import WCS

logger = logging.getLogger(__name__)


class FitsImage(DataAccessor):
    """Use PyFITS to pull image data out of a FITS file.

    Provide standard attributes, as per :class:`DataAccessor`. In addition, we
    provide a ``telescope`` attribute if the FITS file has a ``TELESCOP``
    header.

    Parameters
    ----------
    url : Path or str
        The path or URL to the FITS file.
    plane : int, default: None
        If the data is a datacube, specifies which plane to use.
    beam : tuple, default: None
        Beam parameters in degrees, in the form (bmaj, bmin, bpa). If not
        supplied, the method will attempt to read these from the header.
    hdu_index : int, default: 0
        The index of the HDU to use from the HDU list.

    """

    def __init__(self, url, plane=None, beam=None, hdu_index=0):
        self.url = url
        self.header = self._get_header(hdu_index)
        self.wcs = self.parse_coordinates()
        self.data = self.read_data(hdu_index, plane)
        self.taustart_ts, self.tau_time = self.parse_times()
        self.freq_eff, self.freq_bw = self.parse_frequency()
        self.pixelsize = self.parse_pixelsize()

        if is_valid_beam_tuple(beam) or not is_valid_beam_tuple(self.beam):
            # An argument-supplied beam overrides a beam derived from
            # (bmaj, bmin, bpa) in a config.toml. Only if those two options
            # are not specified, we parse the beam from the header.
            bmaj, bmin, bpa = (
                beam if is_valid_beam_tuple(beam) else (self.parse_beam())
            )
            self.beam = self.degrees2pixels(
                bmaj, bmin, bpa, self.pixelsize[0], self.pixelsize[1]
            )

        self.centre_ra, self.centre_decl = self.calculate_phase_centre()

        # Bonus attribute
        if "TELESCOP" in self.header:
            self.telescope = self.header["TELESCOP"]

    def _get_header(self, hdu_index):
        """Retrieve the header from the specified HDU in the FITS
        file.

        Parameters
        ----------
        hdu_index : int
            The index of the Header Data Unit (HDU) to extract the header from.

        Returns
        -------
        astropy.io.fits.Header
            A copy of the header from the specified HDU.

        """
        with pyfits.open(self.url) as hdulist:
            hdu = hdulist[hdu_index]
        return hdu.header.copy()

    def read_data(self, hdu_index, plane):
        """Read data from our FITS file.

        Parameters
        ----------
        hdu_index : int
            The index of the Header Data Unit (HDU) to extract the data from.
        plane : int, default: None
            If the data is a datacube, specifies which plane to use.

        Returns
        -------
        numpy.ndarray
            The processed 2D data array, transposed for intuitive display.

        Notes
        -----
        PyFITS reads the data into an array indexed as [y][x]. We
        take the transpose to make this more intuitively reasonable and
        consistent with (eg) ds9 display of the FitsFile. Transpose back
        before viewing the array with RO.DS9, saving to a FITS file,
        etc.

        """
        with pyfits.open(self.url) as hdulist:
            hdu = hdulist[hdu_index]
            data = numpy.float64(hdu.data.squeeze())
        if plane is not None and len(data.shape) > 2:
            data = data[plane].squeeze()
        n_dim = len(data.shape)
        if n_dim != 2:
            logger.warning(
                (
                    f"Loaded datacube with {n_dim:d} dimensions, "
                    "assuming Stokes I and taking plane 0."
                )
            )
            data = data[0, :, :]
        data = data.transpose()
        return data

    def parse_coordinates(self):
        """Parse header to return a WCS (World Coordinate System)
        object.

        Returns
        -------
        WCS
            A WCS object containing the coordinate system information
            extracted from the FITS file header.

        Raises
        ------
        TypeError
            If the coordinate system is not specified in the FITS header.

        Notes
        -----
        If units are not specified in the header, degrees are assumed by
        default.

        """
        header = self.header
        wcs = WCS()
        try:
            wcs.crval = header["crval1"], header["crval2"]
            wcs.crpix = header["crpix1"] - 1, header["crpix2"] - 1
            wcs.cdelt = header["cdelt1"], header["cdelt2"]
        except KeyError:
            msg = "Coordinate system not specified in FITS"
            logger.error(msg)
            raise TypeError(msg)
        try:
            wcs.ctype = header["ctype1"], header["ctype2"]
        except KeyError:
            wcs.ctype = "unknown", "unknown"
        try:
            wcs.crota = float(header["crota1"]), float(header["crota2"])
        except KeyError:
            wcs.crota = 0.0, 0.0
        try:
            wcs.cunit = header["cunit1"], header["cunit2"]
        except KeyError:
            # The "Definition of the Flexible Image Transport System", version
            # 3.0, tells us that "units for celestial coordinate systems defined
            # in this Standard must be degrees", so we assume that if nothing
            # else is specified.
            msg = "WCS units unknown; using degrees"
            logger.warning(msg)
            wcs.cunit = "deg", "deg"
        return wcs

    def calculate_phase_centre(self):
        """Calculate the phase center of the FITS image.

        The phase center is determined by finding the central pixel and
        converting that position to celestial coordinates.

        Returns
        -------
        tuple[float]
            A tuple containing the right ascension and declination of
            the phase center in degrees.

        """
        x, y = self.data.shape
        centre_ra, centre_decl = self.wcs.p2s((x / 2, y / 2))
        return float(centre_ra), float(centre_decl)

    def parse_frequency(self):
        """Set some 'shortcut' variables for access to the frequency
        parameters in the FITS file header.

        Returns
        -------
        freq_eff : float
            The effective frequency extracted from the FITS header.
        freq_bw : float
            The bandwidth extracted from the FITS header.

        """
        try:
            header = self.header
            if "RESTFRQ" in header:
                freq_eff = header["RESTFRQ"]
                if "RESTBW" in header:
                    freq_bw = header["RESTBW"]
                else:
                    logger.warning(
                        "bandwidth header missing in image {},"
                        " setting to 1 MHz".format(self.url)
                    )
                    freq_bw = 1e6
            elif ("CTYPE3" in header) and (
                header["CTYPE3"] in ("FREQ", "VOPT")
            ):
                freq_eff = header["CRVAL3"]
                freq_bw = header["CDELT3"]
            elif ("CTYPE4" in header) and (
                header["CTYPE4"] in ("FREQ", "VOPT")
            ):
                freq_eff = header["CRVAL4"]
                freq_bw = header["CDELT4"]
            else:
                freq_eff = header["RESTFREQ"]
                freq_bw = 1e6
        except KeyError:
            msg = "Frequency not specified in headers for {}".format(self.url)
            logger.error(msg)
            raise TypeError(msg)

        return freq_eff, freq_bw

    def parse_beam(self):
        """Read and return the beam properties bmaj, bmin and bpa
        values from the FITS header.

        Returns
        -------
        bmaj : float
            the major axis of the beam in degrees.
        bmin : float
            the minor axis of the beam in degrees.
        bpa : float
            the position angle of the beam in degrees.

        Notes
        -----
        AIPS FITS file: stored in the history section

        """
        beam_regex = re.compile(
            r"""
            BMAJ
            \s*=\s*
            (?P<bmaj>[-\d\.eE]+)
            \s*
            BMIN
            \s*=\s*
            (?P<bmin>[-\d\.eE]+)
            \s*
            BPA
            \s*=\s*
            (?P<bpa>[-\d\.eE]+)
            """,
            re.VERBOSE,
        )

        bmaj, bmin, bpa = None, None, None
        header = self.header
        try:
            # MIRIAD FITS file
            bmaj = header["BMAJ"]
            bmin = header["BMIN"]
            bpa = header["BPA"]
        except KeyError:

            def get_history(hdr):
                """Retrieve all history cards from a FITS header.

                Parameters
                ----------
                hdr : astropy.io.fits.Header
                    The FITS header object from which to extract history cards.

                Returns
                -------
                list[str]
                    A list of strings, where each string represents a
                    history card from the FITS header.

                """
                return hdr["HISTORY"]

            for hist_entry in get_history(header):
                results = beam_regex.search(hist_entry)
                if results:
                    bmaj, bmin, bpa = [
                        float(results.group(key))
                        for key in ("bmaj", "bmin", "bpa")
                    ]
                    break

        return bmaj, bmin, bpa

    def parse_times(self):
        """Attempt to do something sane with timestamps.

        Returns
        -------
        taustart_ts : datetime.datetime
            Timezone-naive (implicit UTC) datetime representing the
            start of the observation.
        tau_time : float
            Integration time in seconds.

        """
        try:
            start = self.parse_start_time()
        except KeyError:
            # If no start time specified, give up:
            logger.warning(
                (
                    "Timestamp not specified in FITS file:"
                    " using 'now' with dummy (zero-valued) integration "
                    "time."
                )
            )
            return datetime.datetime.now(), 0.0

        try:
            end = dateutil.parser.parse(self.header["end_utc"])
        except KeyError:
            msg = "End time not specified in {}, setting to start".format(
                self.url
            )
            logger.warning(msg)
            end = start

        delta = end - start
        tau_time = delta.total_seconds()

        # For simplicity, the database requires naive datetimes (implicit UTC)
        # So we convert to UTC and then drop the timezone:
        try:
            timezone = pytz.timezone(self.header["timesys"])
            start_w_tz = start.replace(tzinfo=timezone)
            start_utc = pytz.utc.normalize(start_w_tz.astimezone(pytz.utc))
            return start_utc.replace(tzinfo=None), tau_time
        except (pytz.UnknownTimeZoneError, KeyError):
            logger.debug("Timezone not specified in FITS file: assuming UTC.")
            return start, tau_time

    def parse_start_time(self):
        """Parse and return the start time of the observation, that
        yielded this FITS image, from its header.

        Returns
        -------
        datetime.datetime
            The start time of the observation as an instance of
            `datetime.datetime`.

        Raises
        ------
        KeyError
            If the timestamp in the FITS file is unreadable.
        Warning
            Logged if a non-standard date format is encountered in the FITS
            file.

        """
        header = self.header
        try:
            if ";" in header["date-obs"]:
                start = dateutil.parser.parse(
                    header["date-obs"].split(";")[0].split('"')[1]
                )
            else:
                start = dateutil.parser.parse(header["date-obs"])
        except AttributeError:
            # Maybe it's a float, Westerbork-style?
            if isinstance(header["date-obs"], float):
                logger.warning("Non-standard date specified in FITS file!")
                frac, year = numpy.modf(header["date-obs"])
                start = datetime.datetime(int(year), 1, 1)
                delta = datetime.timedelta(365.242199 * frac)
                start += delta
            else:
                raise KeyError("Timestamp in fits file unreadable")
        return start
