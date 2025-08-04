"""
This module implements the CASA LOFAR data container format, described in this
document:

http://www.lofar.org/operations/lib/exe/fetch.php?media=:public:documents
:casa_image_for_lofar_0.03.00.pdf
"""
import datetime
import logging

import numpy
from casacore.tables import table as casacore_table

from sourcefinder.accessors.casaimage import CasaImage
from sourcefinder.accessors.lofaraccessor import LofarAccessor
from sourcefinder.utility.coordinates import julian2unix

logger = logging.getLogger(__name__)

subtable_names = (
    'LOFAR_FIELD',
    'LOFAR_ANTENNA',
    'LOFAR_HISTORY',
    'LOFAR_ORIGIN',
    'LOFAR_QUALITY',
    'LOFAR_STATION',
    'LOFAR_POINTING',
    'LOFAR_OBSERVATION'
)


class LofarCasaImage(CasaImage, LofarAccessor):  # type: ignore[misc]
    """
    Use casacore to pull image data out of a CASA table.

    This accessor assumes the casatable contains the values described in the
    CASA Image description for LOFAR. 0.03.00.

    Parameters
    ----------
    url : str
        Location of the CASA table.
    plane : int, default: 0
        If the data is a datacube, specifies which plane to use.
    beam : tuple, default: None
        Beam parameters in degrees, in the form (bmaj, bmin, bpa). If not
        supplied, the method will attempt to read these from the header.
    """

    def __init__(self, url, plane=0, beam=None):
        super().__init__(url, plane, beam)
        table = casacore_table(self.url, ack=False)
        subtables = self.open_subtables(table)
        self.taustart_ts = self.parse_taustartts(subtables)
        self.tau_time = self.parse_tautime(subtables)

        # Additional, LOFAR-specific metadata
        self.antenna_set = self.parse_antennaset(subtables)
        self.ncore, self.nremote, self.nintl = self.parse_stations(subtables)
        self.subbandwidth = self.parse_subbandwidth(subtables)
        self.subbands = self.parse_subbands(subtables)

    @staticmethod
    def open_subtables(table):
        """Open all subtables defined in the LOFAR format.

        Parameters
        ----------
        table : casacore.tables.table
            A casacore table handler to a LOFAR CASA table.

        Returns
        -------
        dict
            A dictionary containing all LOFAR CASA subtables

        """
        subtables = {}
        for subtable in subtable_names:
            subtable_location = table.getkeyword("ATTRGROUPS")[subtable]
            subtables[subtable] = casacore_table(subtable_location, ack=False)
        return subtables

    @staticmethod
    def parse_taustartts(subtables):
        """Extract the image start time from the CASA table header.
        
        Parameters
        ----------
        subtables : dict
            A dictionary containing all LOFAR CASA subtables.
        
        Returns
        -------
        datetime.datetime
            The earliest observation start time as a datetime object.
        
        Notes
        -----
        We sort the table in order of ascending start time then
        choose the first value to ensure we get the earliest possible
        starting time.

        """
        observation_table = subtables['LOFAR_OBSERVATION']
        julianstart = observation_table.query(
            sortlist="OBSERVATION_START", limit=1).getcell(
            "OBSERVATION_START", 0
        )
        unixstart = julian2unix(julianstart)
        taustart_ts = datetime.datetime.fromtimestamp(unixstart)
        return taustart_ts

    @staticmethod
    def non_overlapping_time(series):
        """Calculate the sum of total time ranges without overlap.

        Parameters
        ----------
        series : list of tuple
            A list of 2-item tuples representing time ranges, where each tuple
            contains the start and end time of a range.

        Returns
        -------
        float
            The total length of all time ranges without overlap, most likely in
            seconds.

        """
        series.sort()
        overlap = total = 0
        for n, (start, end) in enumerate(series):
            total += end - start
            for (nextstart, nextend) in series[n + 1:]:
                if nextstart >= end:
                    break
                overlapstart = max(nextstart, start)
                overlapend = min(nextend, end)
                overlap += overlapend - overlapstart
                start = overlapend
        return total - overlap

    @staticmethod
    def parse_tautime(subtables):
        """Returns the total on-sky time for this image.
        
        Parameters
        ----------
        subtables : dict
            A dictionary containing all LOFAR CASA subtables.
        
        Returns
        -------
        float
            The total on-sky time, most likely in seconds.

        """
        origin_table = subtables['LOFAR_ORIGIN']
        startcol = origin_table.col('START')
        endcol = origin_table.col('END')
        series = [(int(start), int(end)) for start, end in
                  zip(startcol, endcol)]
        tau_time = LofarCasaImage.non_overlapping_time(series)
        return tau_time

    @staticmethod
    def parse_antennaset(subtables):
        """Extract the antenna set used in the observation.

        This method retrieves the unique antenna set from the
        LOFAR_OBSERVATION subtable.

        Parameters
        ----------
        subtables : dict
            A dictionary containing all LOFAR CASA subtables.

        Returns
        -------
        numpy.ndarray
            An array containing the unique antenna set used in the observation.

        Raises
        ------
        Exception if multiple antenna sets are found.

        Notes
        -----
        The method uses the `unique_column_values` function from the
        `CasaImage` class to extract unique values from the "ANTENNA_SET"
        column.

        """
        observation_table = subtables['LOFAR_OBSERVATION']
        antennasets = CasaImage.unique_column_values(observation_table,
                                                     "ANTENNA_SET")
        if len(antennasets) == 1:
            return antennasets[0]
        else:
            raise Exception("Cannot handle multiple antenna sets in image")

    @staticmethod
    def parse_subbands(subtables):
        """Extract the number of subbands used in the observation.

        This method retrieves the unique number of subbands from the
        LOFAR_ORIGIN subtable. If multiple values are found, an exception is
        raised.

        Parameters
        ----------
        subtables : dict
            A dictionary containing all LOFAR CASA subtables.

        Returns
        -------
        int
            The number of subbands used in the observation.

        Raises
        ------
        Exception
            If varying numbers of channels are found in the subtable.

        Notes
        -----
        The method uses the `unique_column_values` function from the
        `CasaImage` class to extract unique values from the "NUM_CHAN" column.

        """
        origin_table = subtables['LOFAR_ORIGIN']
        num_chans = CasaImage.unique_column_values(origin_table, "NUM_CHAN")
        if len(num_chans) == 1:
            return num_chans[0]
        else:
            raise Exception(
                "Cannot handle varying numbers of channels in image")

    @staticmethod
    def parse_subbandwidth(subtables):
        """Calculate the subband width for the observation.

        This method determines the subband width based on the clock frequency
        retrieved from the LOFAR_OBSERVATION subtable.

        Parameters
        ----------
        subtables : dict
            A dictionary containing all LOFAR CASA subtables.

        Returns
        -------
        float
            The subband width in Hz.

        Raises
        ------
        Exception
            If multiple clock frequencies are found in the subtable.

        Notes
        -----
        The method uses the clock frequency and its unit to calculate the
        subband width. The clock frequency is divided by 1024 to determine
        the subband width. For more details, see:
        https://www.astron.nl/lofarwiki/doku.php?id=public:documents:
        raw_olap_data_formats&s[]=subband
        'base_subband_hz = clock_hz / 1024'

        """
        freq_units = {
            'Hz': 1,
            'kHz': 10 ** 3,
            'MHz': 10 ** 6,
            'GHz': 10 ** 9,
        }
        observation_table = subtables['LOFAR_OBSERVATION']
        clockcol = observation_table.col('CLOCK_FREQUENCY')
        clock_values = CasaImage.unique_column_values(observation_table,
                                                      "CLOCK_FREQUENCY")
        if len(clock_values) == 1:
            clock = clock_values[0]
            unit = clockcol.getkeyword('QuantumUnits')[0]
            trueclock = freq_units[unit] * clock
            subbandwidth = trueclock / 1024
            return subbandwidth
        else:
            raise Exception("Cannot handle varying clocks in image")

    @staticmethod
    def parse_stations(subtables):
        """Extract the number of specific LOFAR stations used in the
        observation.

        This method calculates the number of core, remote, and international
        LOFAR stations used based on the observation and antenna subtables.

        Parameters
        ----------
        subtables : dict
            A dictionary containing all LOFAR CASA subtables.

        Returns
        -------
        tuple
            A tuple containing the number of core stations, remote stations,
            and international stations in the format (ncore, nremote, nintl).

        Notes
        -----
        The method uses the "NVIS_USED" column from the LOFAR_OBSERVATION
        subtable and the "NAME" column from the LOFAR_ANTENNA subtable to
        determine which
        stations were used. Stations are categorized based on their names:

        - Core stations start with "CS".
        - Remote stations start with "RS".
        - International stations have other prefixes.

        """
        observation_table = subtables['LOFAR_OBSERVATION']
        antenna_table = subtables['LOFAR_ANTENNA']
        nvis_used = observation_table.getcol('NVIS_USED')
        names = numpy.array(antenna_table.getcol('NAME'))
        mask = numpy.sum(nvis_used, axis=2) > 0
        used = names[mask[0]]
        ncore = nremote = nintl = 0
        for station in used:
            if station.startswith('CS'):
                ncore += 1
            elif station.startswith('RS'):
                nremote += 1
            else:
                nintl += 1
        return ncore, nremote, nintl
