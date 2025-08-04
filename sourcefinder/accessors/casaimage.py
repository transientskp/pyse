import logging
import warnings
from math import degrees

from casacore.tables import table as casacore_table

from sourcefinder.utils import is_valid_beam_tuple
from sourcefinder.accessors.dataaccessor import DataAccessor
from sourcefinder.utility.coordinates import WCS
from sourcefinder.utility.coordinates import mjd2datetime

logger = logging.getLogger(__name__)


class CasaImage(DataAccessor):
    """
    Provides common functionality for pulling data from the CASA image format.

    (Technically known as the 'MeasurementSet' format.)

    Parameters
    ----------
    url : str
        The location of the CASA image file.
    plane : int, default: 0
        The data plane to extract from the image cube.
    beam : tuple, default: None
        Beam parameters (major axis, minor axis, position angle) in degrees.
        If not provided, these are extracted from the CASA table.

    Notes
    -----
    CasaImage does not provide tau_time or taustart_ts, as there was
    no clear standard for these metadata, so cannot be
    instantiated directly - subclass it and extract these attributes
    as appropriate to a given telescope.
    """

    def __init__(self, url, plane=0, beam=None):
        # super().__init__()
        self.url = url

        # we don't want the table as a property since it makes the accessor
        # not serializable
        table = casacore_table(self.url, ack=False)
        self.data = self.parse_data(table, plane)
        self.wcs = self.parse_coordinates(table)
        self.centre_ra, self.centre_decl = self.parse_phase_centre(table)
        self.freq_eff, self.freq_bw = self.parse_frequency(table)
        self.pixelsize = self.parse_pixelsize()

        if is_valid_beam_tuple(beam) or not is_valid_beam_tuple(self.beam):
            # An argument-supplied beam overrides a beam derived from
            # (bmaj, bmin, bpa) in a config.toml. Only if those two options
            # are not specified, we parse the beam from the header.
            bmaj, bmin, bpa = beam if is_valid_beam_tuple(beam) else (
                self.parse_beam(table))
            self.beam = self.degrees2pixels(
                bmaj, bmin, bpa, self.pixelsize[0], self.pixelsize[1]
            )

    def parse_data(self, table, plane=0):
        """
        Extract and squeeze data from a CASA table, select desired plane and
        transpose.
        Squeezing is done to remove any singleton dimensions.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which data is extracted.
        plane : int, default: 0
            The data plane to extract from the image cube.

        Returns
        -------
        data : numpy.ndarray
            The extracted and transposed data array. If the data cube has more
            than two dimensions - after removing singleton dimensions - the
            plane specified by the `plane` parameter is selected from the first
            axis.
        """
        data = table[0]['map'].squeeze()
        planes = len(data.shape)
        if planes != 2:
            msg = ("received datacube with %s planes, assuming Stokes I "
                   "and taking plane 0" % planes)
            logger.warning(msg)
            warnings.warn(msg)
            data = data[plane, :, :]
        data = data.transpose()
        return data

    def parse_coordinates(self, table):
        """
        Parse and return a WCS (World Coordinate System) object from the
        CASA table.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which the WCS information is extracted.

        Returns
        -------
        wcs : WCS
            A WCS object containing the coordinate system information extracted
            from the CASA table.
        """
        wcs = WCS()
        my_coordinates = table.getkeyword('coords')['direction0']
        wcs.crval = my_coordinates['crval']
        wcs.crpix = my_coordinates['crpix']
        wcs.cdelt = my_coordinates['cdelt']
        ctype = ['unknown', 'unknown']
        # What about other projections?!
        if my_coordinates['projection'] == "SIN":
            if my_coordinates['axes'][0] == "Right Ascension":
                ctype[0] = "RA---SIN"
            if my_coordinates['axes'][1] == "Declination":
                ctype[1] = "DEC--SIN"
        wcs.ctype = tuple(ctype)
        # Rotation, units? We better set a default
        wcs.crota = (0., 0.)
        wcs.cunit = table.getkeyword('coords')['direction0']['units']
        return wcs

    def parse_frequency(self, table):
        """
        Extract frequency-related information from the CASA table headers.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which frequency information is extracted.

        Returns
        -------
        freq_eff : float
            The effective frequency (rest frequency) in Hz.
        freq_bw : float
            The frequency bandwidth (channel width) in Hz.
        """
        freq_eff = table.getkeywords()['coords']['spectral2']['restfreq']
        freq_bw = table.getkeywords()['coords']['spectral2']['wcs']['cdelt']
        return freq_eff, freq_bw

    def parse_beam(self, table):
        """Extract beam parameters from the CASA table.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which the beam parameters are extracted.

        Returns
        -------
        bmaj : float
            The semi-major axis of the beam in degrees.
        bmin : float
            The semi-minor axis of the beam in degrees.
        bpa : float
            The angle of the major axis of the synthesized beam,
            measured east from local north, in degrees.

        """

        def ensure_degrees(quantity):
            """
            Convert a quantity to degrees.

            Parameters
            ----------
            quantity : dict
                A dictionary containing the value and unit of the quantity to
                be converted. Supported units are 'deg', 'arcsec', and 'rad'.

            Returns
            -------
            float
                The value of the quantity converted to degrees.

            Raises
            ------
            Exception
                If the unit of the quantity is not recognized.
            """
            if quantity['unit'] == 'deg':
                return quantity['value']
            elif quantity['unit'] == 'arcsec':
                return quantity['value'] / 3600
            elif quantity['unit'] == 'rad':
                return degrees(quantity['value'])
            else:
                raise Exception("Beam units (%s) unknown" % quantity['unit'])

        restoringbeam = table.getkeyword('imageinfo')['restoringbeam']
        bmaj = ensure_degrees(restoringbeam['major'])
        bmin = ensure_degrees(restoringbeam['minor'])
        bpa = ensure_degrees(restoringbeam['positionangle'])
        return bmaj, bmin, bpa

    @staticmethod
    def parse_phase_centre(table):
        """
        Assume the units for the pointing centre are in radians and return the
        RA modulo 360.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which the pointing centre is extracted.

        Notes
        -----
        The units for the pointing centre are not given in either the image
        cube itself nor in the image content description. Assume radians. The
        Right Ascension (RA) is returned modulo 360 to ensure it is always
        within the range 0 <= RA < 360.
        """
        centre_ra, centre_decl = table.getkeyword('coords')['pointingcenter'][
            'value']
        return degrees(centre_ra) % 360, degrees(centre_decl)

    @staticmethod
    def parse_taustartts(table):
        """
        Extract integration start-time from CASA table header.

        This applies to some CASA images (typically those created from uvFITS
        files) but not all, and so should be called for each sub-class.

        Parameters
        ----------
        table : casacore.tables.table
            The MAIN table of the CASA image.

        Returns
        -------
        datetime.datetime
            Time of observation start as an instance of `datetime.datetime`.
        """
        obsdate = table.getkeyword('coords')['obsdate']['m0']['value']
        return mjd2datetime(obsdate)

    @staticmethod
    def unique_column_values(table, column_name):
        """
        Find all the unique values in a particular column of a CASA table.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which the column values are extracted.
        column_name : str
            The name of the column to extract unique values from.

        Returns
        -------
        numpy.ndarray
            An array containing the unique values in the specified column.
        """
        return table.query(
            columns=column_name, sortlist="unique %s" % (column_name)
        ).getcol(column_name)
