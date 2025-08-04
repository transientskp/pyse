import logging

from casacore.tables import table as casacore_table

from sourcefinder.accessors.casaimage import CasaImage

logger = logging.getLogger(__name__)


class AartfaacCasaImage(CasaImage):
    """A class to represent an AARTFAAC CASA image, extending the CasaImage class.

    Parameters
    ----------
    url : str
        The location of the CASA table.
    plane : int, default: 0
        If the data is a cube, specifies which plane to use.
    beam : tuple, default: None
        Beam parameters in degrees, in the form (bmaj, bmin, bpa).

    """
    def __init__(self, url, plane=0, beam=None):
        super().__init__(url, plane=0, beam=None)
        table = casacore_table(self.url, ack=False)
        self.taustart_ts = self.parse_taustartts(table)
        self.telescope = table.getkeyword('coords')['telescope']

        # TODO: header does't contain integration time
        # aartfaac imaginig pipeline issue #25
        self.tau_time = 1

    def parse_frequency(self, table):
        """Extract frequency-related information from the casacore
        table.

        This method overrides the implementation in the `CasaImage` class,
        which retrieves the entries from the 'spectral2' sub-table.

        Parameters
        ----------
        table : casacore.tables.table
            The CASA table from which frequency information is extracted.

        Returns
        -------
        freq_eff : float
            The effective frequency (rest frequency) in Hz extracted
            from the casacore table.
        freq_bw : float
            The frequency bandwidth in Hz, derived from the WCS
            'cdelt' value in the casacore table.

        """
        keywords = table.getkeywords()

        # due to some undocumented casacore feature, the 'spectral' keyword
        # changes from spectral1 to spectral2 when AARTFAAC imaging developers
        # changed some of the header information. For now we will try both
        # locations.
        if 'spectral1' in keywords['coords']:
            keyword = 'spectral1'

        if 'spectral2' in keywords['coords']:
            keyword = 'spectral2'

        freq_eff = keywords['coords'][keyword]['restfreq']
        freq_bw = keywords['coords'][keyword]['wcs']['cdelt']
        return freq_eff, freq_bw
