"""
Data accessors.

These can be used to populate ImageData objects based on some data source
(FITS file, array in memory... etc.).
"""

import os

import astropy.io.fits as pyfits
from astropy.io.fits.hdu.hdulist import HDUList

from sourcefinder.accessors import detection
from sourcefinder.accessors.fitsimageblob import FitsImageBlob
from sourcefinder.accessors.lofarfitsimage import LofarFitsImage
from sourcefinder.config import Conf, ImgConf, ExportSettings
from sourcefinder.image import ImageData


def sourcefinder_image_from_accessor(
    image,
    conf: Conf = Conf(image=ImgConf(), export=ExportSettings()),
):
    """Create a sourcefinder.image.ImageData object from an image
    'accessor'.

    This function initializes a `sourcefinder.image.ImageData` object using
    the data, beam, and WCS information provided by the given image accessor.

    Parameters
    ----------
    image : DataAccessor
        FITS/AIPS/HDF5 image available through an accessor.
    conf : Conf, default: Conf(image=ImgConf(), export=ExportSettings())
        Configuration options for source finding. This includes settings
        related to image processing (e.g., background and rms
        noise estimation, thresholds) as well as export options (e.g., source
        parameters and output maps).

    Returns
    -------
    ImageData
        A sourcefinder.image.ImageData object.

    """
    image = ImageData(image.data, image.beam, image.wcs, conf=conf)
    return image


def writefits(data, filename, header={}):
    """Dump a NumPy array to a FITS file.

    This function writes a given NumPy array to a FITS file, optionally
    including header information. The header can be provided as a dictionary
    containing key-value pairs to be added to the FITS file's metadata.

    Parameters
    ----------
    data : numpy.ndarray
        The NumPy array to be written to the FITS file.
    filename : Path or str
        The path to the output FITS file.
    header : dict, default: {}
        A dictionary containing key-value pairs for the FITS header.

    Raises
    ------
    OSError
        If the file cannot be written due to permission issues or other errors.

    Notes
    -----
    The data is transposed before writing to match the transpose from
    `fitsimage.FitsImage.read_data()`. This is necessary to ensure that the
    data is stored in the correct orientation in the FITS file.

    """
    if header.__class__.__name__ == "Header":
        pyfits.writeto(filename, data.transpose(), header)
    else:
        hdu = pyfits.PrimaryHDU(data.transpose())
        for key in header.keys():
            hdu.header.update(key, header[key])
        hdu.writeto(filename)


def open(path, *args, **kwargs):
    """Returns an accessor object (if available) for the file or
    directory 'path'.

    This function attempts to find an appropriate accessor for the given file
    or directory path. Accessors are tried in order from most specific to least
    specific. For example, an accessor providing `LofarAccessor` is preferred
    over one providing `DataAccessor`, but the latter will be used if no better
    match is found.

    Parameters
    ----------
    path : str or HDUList
        The file path or HDUList object to be processed.
    *args : tuple
        Additional positional arguments to pass to the accessor constructor.
    **kwargs : dict
        Additional keyword arguments to pass to the accessor constructor.

    Returns
    -------
    DataAccessor or subclass
        An accessor object for the given file or directory.

    Raises
    ------
    OSError
        If the file does not exist, cannot be read, or no matching accessor
        class is found.
    Exception
        If the `path` parameter is neither a string nor an `HDUList`.

    """

    if type(path) == HDUList:
        return FitsImageBlob(path, *args, **kwargs)
    elif type(path) == str:
        if not os.access(path, os.F_OK):
            raise OSError("%s does not exist!" % path)
        if not os.access(path, os.R_OK):
            raise OSError("Don't have permission to read %s!" % path)
        Accessor = detection.detect(path)
        if not Accessor:
            raise OSError("no accessor found for %s" % path)
        return Accessor(path, *args, **kwargs)
    else:
        raise Exception("image should be path or HDUlist, got " + str(path))
