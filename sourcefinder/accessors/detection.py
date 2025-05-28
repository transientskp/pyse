import logging
import os.path
from collections import namedtuple

import astropy.io.fits as pyfits
from casacore.images import image as casacore_image
from casacore.tables import table as casacore_table

from sourcefinder.accessors.aartfaaccasaimage import AartfaacCasaImage
from sourcefinder.accessors.amicasaimage import AmiCasaImage
from sourcefinder.accessors.fitsimage import FitsImage
from sourcefinder.accessors.kat7casaimage import Kat7CasaImage
from sourcefinder.accessors.lofarcasaimage import LofarCasaImage
from sourcefinder.accessors.lofarfitsimage import LofarFitsImage
from sourcefinder.accessors.lofarhdf5image import LofarHdf5Image

logger = logging.getLogger(__name__)

# files that should be contained by a casa table
casafiles = ("table.dat", "table.f0", "table.f0_TSM0", "table.info",
             "table.lock")

# We will take the first accessor for which the test returns True.
FitsTest = namedtuple('FitsTest', ['accessor', 'test'])
fits_type_mapping = [
    FitsTest(
        accessor=LofarFitsImage,
        test=lambda hdr: 'TELESCOP' in hdr and 'ANTENNA' in hdr and hdr.get(
            'TELESCOP') == "LOFAR"
    )
]

casa_telescope_keyword_mapping = {
    'LOFAR': LofarCasaImage,
    'KAT-7': Kat7CasaImage,
    'AARTFAAC': AartfaacCasaImage,
    'AMI-LA': AmiCasaImage,
}


def isfits(filename):
    """
    Check if the given file is a FITS file.

    This function verifies whether the specified file exists, has a `.fits`
    extension, and can be opened using the `astropy.io.fits` module.

    Parameters
    ----------
    filename : str
        The path to the file to be checked.

    Returns
    -------
    bool
        True if the file is a valid FITS file, False otherwise.
    """
    if not os.path.isfile(filename):
        return False
    if filename[-4:].lower() != 'fits':
        return False
    try:
        with pyfits.open(filename):
            pass
    except IOError:
        return False
    return True


def iscasa(filename):
    """
    Determine if the given filename corresponds to a LOFAR CASA directory.

    This function checks if the specified directory exists, contains the
    expected files for a CASA table and can be opened using the
    `casacore.tables.table` module.

    Parameters
    ----------
    filename : str
        The path to the directory to be checked.

    Returns
    -------
    bool
        True if the directory is a valid LOFAR CASA directory, False otherwise.
    """
    if not os.path.isdir(filename):
        return False
    for file_ in casafiles:
        casafile = os.path.join(filename, file_)
        if not os.path.isfile(casafile):
            logger.debug("%s doesn't contain %s" % (filename, file_))
            return False
    try:
        table = casacore_table(filename, ack=False)
        table.close()
    except RuntimeError as e:
        logger.debug("directory looks casacore, but cannot open: %s" % str(e))
        return False
    return True


def islofarhdf5(filename):
    """
    Check if the given file is a LOFAR HDF5 container.

    This function verifies whether the specified file exists, has a `.h5`
    extension, and can be opened using the `casacore.images.image` module.

    Parameters
    ----------
    filename : str
        The path to the file to be checked.

    Returns
    -------
    bool
        True if the file is a valid LOFAR HDF5 container, False otherwise.
    """
    if not os.path.isfile(filename):
        return False
    if filename[-2:].lower() != 'h5':
        return False
    try:
        casacore_image(filename)
    except RuntimeError:
        return False
    return True


def fits_detect(filename):
    """
    Detect which telescope produced FITS data, return corresponding accessor.

    This function identifies the telescope that produced the FITS data by
    checking for known FITS image types with expected metadata. If the
    telescope cannot be determined, it defaults to using a regular FitsImage
    accessor.

    Parameters
    ----------
    filename : str
        The path to the FITS file to be analyzed.

    Returns
    -------
    FitsImage or subclass
        The accessor class corresponding to the detected telescope.
    """
    with pyfits.open(filename) as hdulist:
        hdr = hdulist[0].header
    for fits_test in fits_type_mapping:
        if fits_test.test(hdr):
            return fits_test.accessor
    return FitsImage


def casa_detect(filename):
    """
    Detect which telescope produced CASA data, return corresponding accessor.

    This function identifies the telescope that produced the CASA data by
    checking for known CASA table types with expected metadata. If the
    telescope cannot be determined, it returns `None`.

    Parameters
    ----------
    filename : str
        The path to the CASA table to be analyzed.

    Returns
    -------
    subclass of FitsImage or None
        The accessor class corresponding to the detected telescope, or `None`
        if the telescope is unknown.
    """
    table = casacore_table(filename, ack=False)
    telescope = table.getkeyword('coords')['telescope']
    return casa_telescope_keyword_mapping.get(telescope, None)


def detect(filename):
    """
    Determine the accessor class to process the given file.

    This function checks the format of the provided file and returns the
    appropriate accessor class to handle it. It supports FITS files, CASA
    directories, and LOFAR HDF5 containers. If the format is unsupported,
    an `OSError` is raised.

    Parameters
    ----------
    filename : str
        The path to the file or directory to be processed.

    Returns
    -------
    FitsImage or subclass
        The accessor class corresponding to the detected file format.

    Raises
    ------
    OSError
        If the file format is unsupported.
    """
    if isfits(filename):
        return fits_detect(filename)
    elif iscasa(filename):
        return casa_detect(filename)
    elif islofarhdf5(filename):
        return LofarHdf5Image
    else:
        raise OSError("unsupported format: %s" % filename)
