"""
Some generic utility routines for number handling and
calculating (specific) variances
"""

import itertools
import logging

import numpy as np
from numba import guvectorize, float32, int32

import time

from sourcefinder import extract
from sourcefinder import stats
from sourcefinder import utils
from sourcefinder.utility import containers
from sourcefinder.utility.uncertain import Uncertain
from scipy.interpolate import interp1d
import psutil
from multiprocessing import Pool
from functools import cached_property
from functools import partial

try:
    import ndimage
except ImportError:
    from scipy import ndimage
from numba import guvectorize, float32, int32


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method._name_.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{0}  {1:2.2f} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed


def gather(*args):
    return list(args)


logger = logging.getLogger(__name__)

#
# Hard-coded configuration parameters; not user settable.
#
INTERPOLATE_ORDER = 1
MEDIAN_FILTER = 0  # If non-zero, apply a median filter of size
# MEDIAN_FILTER to the background and RMS grids prior
# to interpolating.
MF_THRESHOLD = 0  # If MEDIAN_FILTER is non-zero, only use the filtered
# grid when the (absolute) difference between the raw
# and filtered grids is larger than MF_THRESHOLD.
DEBLEND_MINCONT = 0.005  # Min. fraction of island flux in deblended subisland
STRUCTURING_ELEMENT = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # Island connectiivty
# Vectorized processing of source islands is much faster, but excludes Gaussian
# fits, therefore slightly less accurate.
VECTORIZED = False


class ImageData(object):
    """Encapsulates an image in terms of a numpy array + meta/headerdata.

    This is your primary contact point for interaction with images: it icludes
    facilities for source extraction and measurement, etc.
    """

    def __init__(self, data, beam, wcs, margin=0, radius=0, back_size_x=32,
                 back_size_y=32, residuals=False, islands=False, eps_ra =0,
                 eps_dec=0):
        """
        Initializes an ImageData object.

        Parameters
        ----------
        data : 2D np.ndarray
            Actual image data.
        beam : tuple
            Beam shape specification as (semimajor, semiminor, theta).
        wcs : utility.coordinates.wcs
            World coordinate system specification.
        margin : int, default: 0
            Margin value.
        radius : float, default: 0
            Radius value.
        back_size_x : int, default: 32
            Background size in the x-direction.
        back_size_y : int, default: 32
            Background size in the y-direction.
        residuals : bool, default: False
            Whether to save Gaussian residuals, at the pixels corresponding to
            the islands, as an image. Other pixel values will be zero.
        islands : bool, default: False
            Whether to save the Gaussian reconstructions, at the pixels
            corresponding to the islands, as an image. Other pixel values will
            be zero.
        eps_ra : float, default: 0
            Calibration uncertainty along Right Ascension in degrees.
            See equation 27a of NVSS paper.
        eps_dec : float, default: 0
            Calibration uncertainty along Declination in degrees.
            See equation 27b of NVSS paper.
        """

        # Do data, wcs and beam need deepcopy?
        # Probably not (memory overhead, in particular for data),
        # but then the user shouldn't change them outside ImageData in the
        # meantime
        # self.rawdata is a 2D numpy array, C-contiguous needed for sep.
        # single precision is good enough in all cases.
        self.rawdata = np.ascontiguousarray(data, dtype=np.float32)
        self.wcs = wcs  # a utility.coordinates.wcs instance
        self.beam = beam  # tuple of (semimaj, semimin, theta) in pixel coordinates.
        # These three quantities are only dependent on the beam, so should be calculated
        # once the beam is known and not for each source separately.
        self.fudge_max_pix_factor = utils.fudge_max_pix(beam[0], beam[1], beam[2])
        self.max_pix_variance_factor = utils.maximum_pixel_method_variance(
                           beam[0], beam[1], beam[2])
        self.beamsize = utils.calculate_beamsize(beam[0], beam[1])
        self.correlation_lengths = utils.calculate_correlation_lengths(beam[0], beam[1])
        self.clip = {}
        self.labels = {}
        self.freq_low = 1
        self.freq_high = 1

        self.back_size_x = back_size_x
        self.back_size_y = back_size_y
        self.margin = margin
        self.radius = radius
        self.residuals = residuals
        self.islands = islands
        self.eps_ra = eps_ra
        self.eps_dec = eps_dec

    ###########################################################################
    #                                                                         #
    # Properties and attributes.                                              #
    #                                                                         #
    # Properties are attributes managed by methods; rather than calling the   #
    # method directly, the attribute automatically invokes it. Result of the  #
    # function calls are cached, subsequent calls doesn't recompute.          #
    #                                                                         #
    # clearcache() clears all the cached data, which can get quite large.     #
    # It may be wise to call this, for example, in an exception handler       #
    # dealing with MemoryErrors.                                              #
    #                                                                         #
    ###########################################################################
    @cached_property
    def grids(self):
        """Gridded RMS and background data for interpolating"""
        return self.__grids()


    @cached_property
    def backmap(self):
        """Mean background map"""
        if not hasattr(self, "_user_backmap"):
            return self._interpolate(self.grids['bg'],
                                         self.grids['indices'])
        else:
            return self._user_backmap

    @cached_property
    def rmsmap(self):
        """root-mean-squares map, i.e. the standard deviation of the local
        background noise, interpolated across the image."""
        if not hasattr(self, "_user_noisemap"):
            return self._interpolate(self.grids['rms'],
                                         self.grids['indices'], roundup=True)
        else:
            return self._user_noisemap

    @cached_property
    def data(self):
        """Masked image data"""
        # We will ignore all the data which is masked for the rest of the
        # sourcefinding process. We build up the mask by stacking ("or-ing
        # together") a number of different effects:
        #
        # * A margin from the edge of the image;
        # * Any data outside a given radius from the centre of the image;
        # * Data which is "obviously" bad (equal to 0 or NaN).
        mask = np.zeros((self.xdim, self.ydim))
        if self.margin:
            margin_mask = np.ones((self.xdim, self.ydim))
            margin_mask[self.margin:-self.margin, self.margin:-self.margin] = 0
            mask = np.logical_or(mask, margin_mask)
        if self.radius:
            radius_mask = utils.circular_mask(self.xdim, self.ydim, self.radius)
            mask = np.logical_or(mask, radius_mask)
        mask = np.logical_or(mask, np.isnan(self.rawdata))
        return np.ma.array(self.rawdata, mask=mask)

    @cached_property
    def data_bgsubbed(self):
        """Background subtracted masked image data"""
        return (self.data - self.backmap).astype(np.float32, copy=False)

    @property
    def xdim(self):
        """X pixel dimension of (unmasked) data"""
        return self.rawdata.shape[0]

    @property
    def ydim(self):
        """Y pixel dimension of (unmasked) data"""
        return self.rawdata.shape[1]

    @property
    def pixmax(self):
        """Maximum pixel value (pre-background subtraction)"""
        return self.data.max()

    @property
    def pixmin(self):
        """Minimum pixel value (pre-background subtraction)"""
        return self.data.min()

    def clearcache(self):
        """Zap any calculated data stored in this object.

        Clear the background and rms maps, labels, clip, and any locally held
        data. All of these can be reconstructed from the data accessor.

        Note that this *must* be run to pick up any new settings.
        """
        try:
            self.labels.clear()
            self.clip.clear()
            del self.backmap
            del self.rmsmap
            del self.data
            del self.data_bgsubbed
            del self.grids
            del self.Gaussian_islands
            del self.Gaussian_residuals
            del self.residuals_from_deblending
        except AttributeError:
            pass

    ###########################################################################
    #                                                                         #
    # General purpose image handling.                                         #
    #                                                                         #
    # Routines for saving and trimming data, and calculating background/RMS   #
    # maps (in conjuntion with the properties above).                         #
    #                                                                         #
    ###########################################################################

    # Private "support" methods
    def __grids(self):
        """Calculate background and RMS grids of this image.

        These grids can be interpolated up to make maps of the original image
        dimensions: see _interpolate().

        This is called automatically when ImageData.backmap,
        ImageData.rmsmap or ImageData.fdrmap is first accessed.
        """

        # there's no point in working with the whole of the data array
        # if it's masked.
        useful_chunk = ndimage.find_objects(np.where(self.data.mask, 0, 1))
        assert (len(useful_chunk) == 1)
        x_dim, y_dim  = self.data[useful_chunk[0]].shape
        # We should divide up the image into subimages such that each grid
        # node is centered on a subimage. This is only possible if
        # self.back_size_x and self.back_size_y are divisors of xdim and ydim,
        # respectively. If not, we need to select a frame within useful_chunk
        # that does have the appropriate dimensions. At the same time, it should
        # be as large as possible and centered within useful_chunk.
        rem_row = np.mod(x_dim, self.back_size_x)
        rem_col = np.mod(y_dim, self.back_size_y)
        start_offset_row, rem_rem_row = divmod(rem_row, 2)
        start_offset_col, rem_rem_col = divmod(rem_col, 2)
        end_offset_row = start_offset_row + rem_rem_row
        end_offset_col = start_offset_col + rem_rem_col

        offsets = np.array([start_offset_row, -end_offset_row,
                            start_offset_col, -end_offset_col])

        useful_chunk_inds = ImageData.slices_to_indices(useful_chunk)[0]

        centred_inds = useful_chunk_inds + offsets

        # Before proceeding, check that our data has the size of at least
        # one subimage, for both dimensions.
        if (centred_inds[1] - centred_inds[0] > self.back_size_x and
            centred_inds[3] - centred_inds[2] > self.back_size_y):

            subimages, number_of_elements_for_each_subimage = \
                utils.make_subimages(self.data.data[centred_inds[0]:
                                                    centred_inds[1],
                                                    centred_inds[2]:
                                                    centred_inds[3]],
                                     self.data.mask[centred_inds[0]:
                                                    centred_inds[1],
                                                    centred_inds[2]:
                                                    centred_inds[3]],
                                     self.back_size_x, self.back_size_y)

            mean_grid = np.zeros(number_of_elements_for_each_subimage.shape,
                                 dtype=np.float32)
            rms_grid = np.zeros(number_of_elements_for_each_subimage.shape,
                                dtype=np.float32)

            stats.data_clipper_dynamic(subimages,
                                       number_of_elements_for_each_subimage,
                                       mean_grid, rms_grid)

            # Fill in the zeroes with nearest neighbours.
            # In this way we do not have to make a MaskedArray, which
            # scipy.interpolate.interp1d cannot handle adequately.
            mean_grid = utils.nearest_nonzero(mean_grid, rms_grid)
            rms_grid = utils.nearest_nonzero(rms_grid, rms_grid)
        else:
            # Return an empty grid if we don't have enough pixels along both
            # dimensions. In that case ImageData._interpolate will return
            # completely masked background maps, which is what we want, since
            # no sources will be extracted.
            mean_grid = np.empty((0, 0), dtype=np.float32)
            rms_grid = np.empty((0, 0), dtype=np.float32)

        return {'bg': mean_grid, 'rms': rms_grid, 'indices': centred_inds}


    def _interpolate(self, grid, inds, roundup=False):

        """
        Interpolate a grid to produce a map of the dimensions of the image.

        Parameters
        ----------
        grid : np.ma.MaskedArray
            The grid to be interpolated.

        roundup : bool, default: False
            If True, values of the resultant map which are lower than the input
            grid are trimmed. Default is False.

        Returns
        -------
        np.ma.MaskedArray
            The interpolated map.

        Notes
        -----
        This function is used to transform the RMS, background or FDR grids
        produced by :func:`_grids()` to a map we can compare with the image
        data.
        """
        # Use zeroes with the mask from the observational image as a starting
        # point for the mean background and rms background maps. Next, use
        # the interpolated values from the background grids, which were derived
        # using kappa * sigma clipping, to fill in all unmasked pixels.
        my_map = np.ma.MaskedArray(np.zeros(self.data.shape),
                                   mask=self.data.mask, dtype=np.float32)

        # Remove the MaskedArrayFutureWarning warning and keep old numpy < 1.11
        # behavior
        my_map.unshare_mask()

        # If the grid has size 0 there is no point in proceeding.
        if grid.size == 0:
            my_map.mask = True
            return my_map

        my_xdim, my_ydim = inds[1] - inds[0], inds[3] - inds[2]

        if MEDIAN_FILTER:
            f_grid = ndimage.median_filter(grid, MEDIAN_FILTER)
            if MF_THRESHOLD:
                grid = np.where(
                    np.fabs(f_grid - grid) > MF_THRESHOLD, f_grid, grid
                )
            else:
                grid = f_grid

        # Bicubic spline interpolation
        xratio = float(my_xdim) / self.back_size_x
        yratio = float(my_ydim) / self.back_size_y

        # Inspired by https://stackoverflow.com/questions/13242382/
        # resampling-a-numpy-array-representing-an-image
        # Should be much faster than scipy.ndimage.map_coordinates.
        # scipy.ndimage.zoom should also be an option for speedup, but zoom did
        # not reproduce the exact same output as map_coordinates.
        # Used fitsdiff to check that it gives the exact same output as the
        # original code up to and including --relative-tolerance=1e-15 for
        # INTERPOLATE_ORDER=1.
        # Achieving this resemblance was quite involved and the
        # fill_value is essential in interp1d. However, for some unit tests,
        # grid.shape=(1,1) and then it will break with "ValueError: x and y
        # arrays must have at least 2 entries". So in that case
        # map_coordinates should be used.

        if INTERPOLATE_ORDER == 1 and grid.shape[0] > 1 and grid.shape[1] > 1:
            x_initial = np.linspace(0., grid.shape[0]-1, grid.shape[0],
                                    endpoint=True, dtype=np.float32)
            y_initial = np.linspace(0., grid.shape[1]-1, grid.shape[1],
                                    endpoint=True, dtype=np.float32)
            x_sought = np.linspace(-0.5, -0.5 + xratio, my_xdim,
                                   endpoint=True, dtype=np.float32)
            y_sought = np.linspace(-0.5, -0.5 + yratio, my_ydim,
                                   endpoint=True, dtype=np.float32)

            primary_interpolation = interp1d(y_initial, grid, kind='slinear',
                                             assume_sorted=True, axis=1,
                                             copy=False, bounds_error=False,
                                             fill_value=(grid[:, 0],
                                                         grid[:, -1]))
            transposed = primary_interpolation(y_sought).T

            perpendicular_interpolation = interp1d(x_initial, transposed,
                                                   kind='slinear',
                                                   assume_sorted=True,
                                                   axis=1, copy=False,
                                                   bounds_error=False,
                                                   fill_value=(transposed[:, 0],
                                                               transposed[:, -1]))

            my_map[inds[0]:inds[1], inds[2]:inds[3]] = \
                perpendicular_interpolation(x_sought).T

        else:
            # This condition is there to make sure we actually have some
            # unmasked patch of the image to fill.
            slicex = slice(-0.5, -0.5 + xratio, 1j * my_xdim)
            slicey = slice(-0.5, -0.5 + yratio, 1j * my_ydim)

            my_map[inds[0]:inds[1], inds[2]:inds[3]] = (
                ndimage.map_coordinates(grid, np.mgrid[slicex, slicey],
                                        mode='nearest',
                                        order=INTERPOLATE_ORDER))

        # If the input grid was entirely masked, then the output map must
        # also be masked: there's no useful data here. We don't search for
        # sources on a masked background/RMS, so this data will be cleanly
        # skipped by the rest of the sourcefinder
        if np.ma.getmask(grid).all():
            my_map.mask = True
        elif roundup:
            # In some cases, the spline interpolation may produce values
            # lower than the minimum value in the map. If required, these
            # can be trimmed off. No point doing this if the map is already
            # fully masked, though.
            my_map = np.ma.MaskedArray(
                data=np.where(
                    my_map >= np.min(grid), my_map, np.min(grid)),
                mask=my_map.mask
            )
        return my_map

    ###########################################################################
    #                                                                         #
    # Source extraction.                                                      #
    #                                                                         #
    # Provides for both traditional (islands-above-RMS) and FDR source        #
    # extraction systems.                                                     #
    #                                                                         #
    ###########################################################################

    @timeit
    def extract(self, det, anl, noisemap=None, bgmap=None, labelled_data=None,
                labels=None, deblend_nthresh=0, force_beam=False):

        """
        Kick off conventional (ie, rms island finding) source extraction.

        Parameters
        ----------
        det : float
            Detection threshold, as a multiple of the rms noise. At
            least one pixel in a source must exceed this for it to be
            regarded as significant.
        anl : float
            Analysis threshold, as a multiple of the rms noise. All
            the pixels within the island that exceed this will be used
            when fitting the source.
        noisemap : np.ndarray, default: None
            Noise map, i.e. the standard deviation (rms) of the background
            noise across the observational image
        bgmap : np.ndarray, default: None
            Background map, i.e. the mean of the background noise across
            the observational image.
        labelled_data : np.ndarray, default: None
            The output of a connected component
            analysis of the image, with a unique label for each source. Should
            have the same shape as the observational image.
        labels : np.ndarray, default: None
            Labels array, i.e. a 1D integer array of labels for each source.
        deblend_nthresh : int, default: 0
            Number of subthresholds to use for deblending. Set to 0
            to disable.
        force_beam : bool, default: False
            Force all extractions to have major/minor axes and position angle
            equal to the restoring beam.

        Returns
        -------
        :class:`sourcefinder.utility.containers.ExtractionResults`
        """
        if anl > det:
            logger.warning(
                "Analysis threshold is higher than detection threshold"
            )

        # If the image data is flat we may as well crash out here with a
        # sensible error message, otherwise the RMS estimation code will
        # crash out with a confusing error later.
        if np.ma.max(self.data) == np.ma.min(self.data):
            raise RuntimeError("Bad data: Image data is flat")

        if (type(bgmap).__name__ == 'ndarray' or
                type(bgmap).__name__ == 'MaskedArray'):
            if bgmap.shape != self.backmap.shape:
                raise IndexError("Background map has wrong shape")
            else:
                self.backmap = bgmap

        if (type(noisemap).__name__ == 'ndarray' or
                                       type(noisemap).__name__ == 'MaskedArray'):
            if noisemap.shape != self.rmsmap.shape:
                raise IndexError("Noisemap has wrong shape")
            if noisemap.min() < 0:
                raise ValueError("RMS noise cannot be negative")
            else:
                self.rmsmap = noisemap

        if labelled_data is not None and labelled_data.shape != self.data.shape:
            raise ValueError("Labelled map is wrong shape")

        return self._pyse(
            det * self.rmsmap, anl * self.rmsmap, deblend_nthresh, force_beam,
            labelled_data=labelled_data, labels=labels
        )

    def reverse_se(self, det, anl):
        """Run source extraction on the negative of this image.

        This process can be used to estimate the false positive rate, as there
        should be no sources in the negative image.

        Parameters
        ----------
        det : float
            Detection threshold, as a multiple of the rms noise. At
            least one pixel in a source must exceed this for it to be
            regarded as significant.
        anl : float
            Analysis threshold, as a multiple of the rms noise. All
            the pixels within the island that exceed this will be used
            when fitting the source.

        Returns
        -------
        :class:`sourcefinder.utility.containers.ExtractionResults`

        To prevent interference with the normal extraction process, cached
        data (background map, clips, etc.) is cleared before and after
        running this method. If this method is used frequently, a separate
        cache may be implemented in the future.
        """
        self.labels.clear()
        self.clip.clear()
        self.data_bgsubbed *= -1
        results = self.extract(det=det, anl=anl)
        self.data_bgsubbed *= -1
        self.labels.clear()
        self.clip.clear()
        return results

    def fd_extract(self, alpha, anl=None, noisemap=None,
                   bgmap=None, deblend_nthresh=0, force_beam=False
                   ):
        """
        False Detection Rate based source extraction.

        The FDR procedure guarantees that the False Detection Rate (FDR) is less
        than alpha.

        Parameters
        ----------
        alpha : float
            Maximum allowed fraction of false positives. Must be between 0 and
            1, exclusive.
        anl : float, default: None
            Analysis threshold, as a multiple of the rms noise. All
            the pixels within the island that exceed this will be used
            when fitting the source.
        noisemap : np.ndarray, default: None
            Noise map, i.e. the standard deviation (rms) of the background
            noise across the observational image
        bgmap : np.ndarray, default: None
            Background map, i.e. the mean of the background noise across
            the observational image.
        deblend_nthresh : int, default: 0
            Number of subthresholds to use for deblending. Set to 0
            to disable.
        force_beam : bool, default: False
            Force all extractions to have major/minor axes and position angle
            equal to the restoring beam.

        Returns
        -------
        :class:`sourcefinder.utility.containers.ExtractionResults`

        Notes
        -----
        See Hopkins et al., AJ, 123, 1086 (2002) for more details.
        http://adsabs.harvard.edu/abs/2002AJ....123.1086H
        """
        # The correlation length in config.py is used not only for the
        # calculation of error bars with the Condon formulae, but also for
        # calculating the number of independent pixels.
        corlengthlong, corlengthshort = self.correlation_lengths

        C_n = (1.0 / np.arange(
            round(0.25 * np.pi * corlengthlong *
                  corlengthshort + 1))[1:]).sum()

        # Calculate the FDR threshold
        # Things will go terribly wrong in the line below if the interpolated
        # noise values get very close or below zero. Use INTERPOLATE_ORDER=1
        # or the roundup option.
        if (type(bgmap).__name__ == 'ndarray' or
                type(bgmap).__name__ == 'MaskedArray'):
            if bgmap.shape != self.backmap.shape:
                raise IndexError("Background map has wrong shape")
            else:
                self.backmap = bgmap
        if (type(noisemap).__name__ == 'ndarray' or
                    type(noisemap).__name__ == 'MaskedArray'):
            if noisemap.shape != self.rmsmap.shape:
                raise IndexError("Noisemap has wrong shape")
            if noisemap.min() < 0:
                raise ValueError("RMS noise cannot be negative")
            else:
                self.rmsmap = noisemap

        normalized_data = self.data_bgsubbed / self.rmsmap

        n1 = np.sqrt(2 * np.pi)
        prob = np.sort(
            np.ravel(np.exp(-0.5 * normalized_data ** 2) / n1))
        lengthprob = float(len(prob))
        compare = (alpha / C_n) * np.arange(lengthprob + 1)[1:] / lengthprob
        # Find the last undercrossing, see, e.g., fig. 9 in Miller et al., AJ
        # 122, 3492 (2001).  Searchsorted is not used because the array is not
        # sorted.
        try:
            index = (np.where(prob - compare < 0.)[0]).max()
        except ValueError:
            # Everything below threshold
            return containers.ExtractionResults()

        fdr_threshold = np.sqrt(-2.0 * np.log(n1 * prob[index]))
        # Default we require that all source pixels are above the threshold,
        # not only the peak pixel.  This gives a better guarantee that indeed
        # the fraction of false positives is less than fdr_alpha in config.py.
        # See, e.g., Hopkins et al., AJ 123, 1086 (2002).
        if not anl:
            anl = fdr_threshold
        return self._pyse(fdr_threshold * self.rmsmap, anl * self.rmsmap,
                          deblend_nthresh, force_beam)

    @staticmethod
    def box_slice_about_pixel(x, y, box_radius):
        """
        Returns a slice centred about (x,y), of width = 2 * int(box_radius) + 1.

        Parameters
        ----------
        x : int
            Desired row index.
        y : int
            Desired column index.
        box_radius : float
            Radius of the box in pixel coordinates.

        Returns
        -------
        tuple of slice
            Slice centred about (x,y) with width = 2*box_radius + 1.
        """
        ibr = int(box_radius)
        x = int(x)
        y = int(y)
        return (slice(x - ibr, x + ibr + 1),
                slice(y - ibr, y + ibr + 1))

    def fit_to_point(self, x, y, boxsize, threshold, fixed):
        """
        Fit an elliptical Gaussian to a specified point on the image.

        Parameters
        ----------
        x : int
            Pixel x-coordinate of the point to fit.
        y : int
            Pixel y-coordinate of the point to fit.
        boxsize : int
            Length of the square section of the image to use for the fit.
        threshold : float
            Threshold below which data is not used for fitting (in units of
            rmsmap).
        fixed : str
            If set to ``position``, the pixel coordinates are fixed in the
            fit.

        Returns
        -------
        Detection
            An instance of :class:`sourcefinder.extract.Detection` containing
            the fit results.
        """
        logger.debug("Force-fitting pixel location ({},{})".format(x, y))
        # First, check that x and y are actually valid semi-positive integers.
        # Otherwise,
        # If they are too high (positive), then indexing will fail
        # BUT, if they are negative, then we get wrap-around indexing
        # and the fit continues at the wrong position!
        if (x < 0 or x > self.xdim
                or y < 0 or y > self.ydim):
            logger.warning("Dropping forced fit at ({},{}), "
                           "pixel position outside image".format(x, y)
                           )
            return None

        # Next, check if any of the central pixels (in a 3x3 box about the
        # fitted pixel position) have been Masked
        # (e.g. if NaNs, or close to image edge) - reject if so.
        central_pixels_slice = ImageData.box_slice_about_pixel(x, y, 1)
        if self.data.mask[central_pixels_slice].any():
            logger.warning(
                "Dropping forced fit at ({},{}), "
                "Masked pixel in central fitting region".format(x, y))
            return None

        if ((
                    # Recent NumPy
                    hasattr(np.ma.core, "MaskedConstant") and
                    isinstance(self.rmsmap, np.ma.core.MaskedConstant)
            ) or (
                # Old NumPy
                np.ma.is_masked(self.rmsmap[int(x), int(y)])
        )):
            logger.error("Background is masked: cannot fit")
            return None

        chunk = ImageData.box_slice_about_pixel(x, y, boxsize / 2.0)
        if threshold is not None:
            # We'll mask out anything below threshold*self.rmsmap from the fit.
            labels, num = self.labels.setdefault(
                # Dictionary mapping threshold -> islands map
                threshold,
                ndimage.label(
                    self.clip.setdefault(
                        # Dictionary mapping threshold -> mask
                        threshold,
                        np.where(
                            self.data_bgsubbed > threshold * self.rmsmap, 1, 0
                        )
                    )
                )
            )

            mylabel = labels[int(x), int(y)]
            if mylabel == 0:  # 'Background'
                raise ValueError(
                    "Fit region is below specified threshold, fit aborted.")
            mask = np.where(labels[chunk] == mylabel, 0, 1)
            fitme = np.ma.array(self.data_bgsubbed[chunk], mask=mask)
            if len(fitme.compressed()) < 1:
                raise IndexError("Fit region too close to edge or too small")
        else:
            fitme = self.data_bgsubbed[chunk]
            if fitme.size < 1:
                raise IndexError("Fit region too close to edge or too small")

        if not len(fitme.compressed()):
            logger.error("All data is masked: cannot fit")
            return None

        # set argument for fixed parameters based on input string
        if fixed == 'position':
            fixed = {'xbar': boxsize / 2.0, 'ybar': boxsize / 2.0}
        elif fixed == 'position+shape':
            fixed = {'xbar': boxsize / 2.0, 'ybar': boxsize / 2.0,
                     'semimajor': self.beam[0],
                     'semiminor': self.beam[1],
                     'theta': self.beam[2]}
        elif fixed is None:
            fixed = {}
        else:
            raise TypeError("Unkown fixed parameter")

        if threshold is not None:
            threshold_at_pixel = threshold * self.rmsmap[int(x), int(y)]
        else:
            threshold_at_pixel = None

        try:
            measurement, _, _ = extract.source_profile_and_errors(
                fitme, threshold_at_pixel, self.rmsmap[int(x), int(y)],
                self.beam, self.fudge_max_pix_factor,
                self.max_pix_variance_factor,
                self.beamsize, self.correlation_lengths, fixed=fixed)
        except ValueError:
            # Fit failed to converge
            # Moments are not applicable when holding parameters fixed
            logger.error("Gaussian fit failed at %f, %f", x, y)
            return None

        try:
            assert (abs(measurement['xbar']) < boxsize)
            assert (abs(measurement['ybar']) < boxsize)
        except AssertionError:
            logger.warning('Fit falls outside of box.')

        measurement['xbar'] += x - boxsize / 2.0
        measurement['ybar'] += y - boxsize / 2.0
        measurement.sig = (fitme / self.rmsmap[chunk]).max()

        return extract.Detection(measurement, self)

    def fit_fixed_positions(self, positions, boxsize, threshold=None,
                            fixed='position+shape',
                            ids=None):
        """
        Convenience function to fit a list of sources at the given positions.

        This function wraps around :py:func:`fit_to_point`.

        Parameters
        ----------
        positions : tuple
            List of (RA, Dec) tuples. Positions to be fit, in decimal degrees.
        boxsize : int
            Length of the square section of the image to use for the fit.
        threshold : float, default: None
            Threshold below which data is not used for fitting.
        fixed : str, default: 'position+shape'
            If set to `position`, the pixel coordinates are fixed in the fit.
        ids : tuple, default: None
            List of identifiers. If not None, must match the length and order of
            the requested fits.

        Note
        ----
        boxsize is in pixel coordinates, not in sky coordinates.

        Returns
        -------
        tuple
            A list of successful fits. If ``ids`` is None, returns a single list
            of :class:`sourcefinder.extract.Detection` s. Otherwise, returns a
            tuple of two matched lists: ([detections], [matching_ids]).
        """
        if ids is not None:
            assert len(ids) == len(positions)

        successful_fits = []
        successful_ids = []
        for idx, posn in enumerate(positions):
            try:
                x, y, = self.wcs.s2p((posn[0], posn[1]))
            except RuntimeError as e:
                if (str(e).startswith("wcsp2s error: 8:") or
                        str(e).startswith("wcsp2s error: 9:")):
                    logger.warning("Input coordinates (%.2f, %.2f) invalid: ",
                                   posn[0], posn[1])
                else:
                    raise
            else:
                try:
                    fit_results = self.fit_to_point(x, y,
                                                    boxsize=boxsize,
                                                    threshold=threshold,
                                                    fixed=fixed)
                    if not fit_results:
                        # We were unable to get a good fit
                        continue
                    if (fit_results.ra.error == float('inf') or
                            fit_results.dec.error == float('inf')):
                        logging.warning("position errors extend outside image")
                    else:
                        successful_fits.append(fit_results)
                        if ids:
                            successful_ids.append(ids[idx])

                except IndexError as e:
                    logger.warning("Input pixel coordinates (%.2f, %.2f) "
                                   "could not be fit because: " + str(e),
                                   posn[0], posn[1])
        if ids:
            return successful_fits, successful_ids
        return successful_fits

    def label_islands(self, detectionthresholdmap, analysisthresholdmap,
                      deblend_nthresh):
        """
        Return a labelled array of pixels for fitting.

        Parameters
        ----------
        detectionthresholdmap : np.ma.MaskedArray
            Detection threshold map with shape (nrow, ncol), matching the shape
            of the observational image (self.rawdata). The values are of dtype
            np.float32.
        analysisthresholdmap : np.ma.MaskedArray
            Analysis threshold map with shape (nrow, ncol), matching the shape
            of the observational image (self.rawdata). The values are of dtype
            np.float32.
        deblend_nthresh : int
            Number of thresholds for deblending.

        Returns
        -------
        tuple
            A tuple containing:
            labels_above_det_thr : np.ndarray
                1D array of labels above detection threshold, with shape
                (num_islands_above_detection_threshold,) and dtype np.int64.
                Note that the length of this array may be smaller than the total
                number of islands above the analysis threshold, as some labels
                may have been filtered out due to a peak spectral brightness
                lower than the local detection threshold.
            labelled_data : np.ndarray
                Array of labelled pixels, where each pixel with a nonzero label
                corresponds to an island above the analysis threshold. The array
                has the same shape as the observational image (self.rawdata) and
                contains integer values corresponding to the labels of the
                islands. Pixels that do not belong to any island are assigned a
                label of 0. The number of islands above the analysis threshold
                is equal to the number of unique labels in this array, which is
                equal to or larger than num_islands_above_detection_threshold,
                i.e. the number of islands above the detection threshold.
                This array has dtype np.int32.
            num_islands_above_detection_threshold : int
                Number of islands above detection threshold.
            maxposs_above_det_thr : np.ndarray
                Array of indices of the maximum pixel values above detection
                threshold, with shape (num_islands_above_detection_threshold, 2)
                and dtype np.int32.
            maxis_above_det_thr : np.ndarray
                Array of maximum pixel values above detection threshold, with
                shape (num_islands_above_detection_threshold,) and dtype
                np.float32.
            npixs_above_det : np.ndarray
                1D array of pixel counts for each island with peak spectral
                brightness above the detection threshold, with shape
                (num_islands_above_detection_threshold,) and dtype np.int32.
            all_indices_above_det_thr : np.ndarray
                Array of indices of the islands above detection threshold, with
                shape (num_islands_above_detection_threshold, 4) and dtype
                np.int32.
            slices : list of slice
                List of slices encompassing all islands in labelled_data, i.e.
                encompassing all islands above the analysis threshold.
        """
        # If there is no usable data, we return an empty set of islands.
        if not len(self.rmsmap.compressed()):
            logging.warning("RMS map masked; sourcefinding skipped")
            return ([], np.zeros(self.data_bgsubbed.shape, dtype=np.int32),
                   None, None, None, None, None, None)

        # At this point, we select all the data which is eligible for
        # sourcefitting. We are actually using three separate filters, which
        # exclude:
        #
        # 1. Anything which has been masked before we reach this point;
        # 2. Any pixels which fall below the analysis threshold at that pixel
        #    position;
        # 3. Any pixels corresponding to a position where the RMS noise is
        #    less than RMS_FILTER (default 0.001) times the median RMS across
        #    the whole image.
        #
        # The third filter attempts to exclude those regions of the image
        # which contain no usable data; for example, the parts of the image
        # falling outside the circular region produced by awimager.
        RMS_FILTER = 0.001

        clipped_data = np.ma.where(
            (self.data_bgsubbed > analysisthresholdmap) &
            (self.rmsmap >= (RMS_FILTER * self.grids["rms"].mean())), 1, 0
        ).filled(fill_value=0)
        labelled_data, num_labels = ndimage.label(clipped_data,
                                                  STRUCTURING_ELEMENT)

        # Get a bounding box for each island:
        # NB Slices ordered by label value (1...N,)
        # 'None' returned for missing label indices.
        slices = ndimage.find_objects(labelled_data)

        # Derive all indices than correspond to the slices
        all_indices = ImageData.slices_to_indices(slices)

        # Derive maximum positions, maximum values and number of pixels
        # per island.
        maxposs = np.empty((num_labels, 2), dtype=np.int32)
        dummy = np.empty_like(maxposs)
        maxis = np.empty(num_labels, dtype=np.float32)
        npixs = np.empty(num_labels, dtype=np.int32)

        ImageData.extract_parms_image_slice(self.data_bgsubbed.data.astype(
                                            dtype=np.float32, copy=False),
                                            all_indices, labelled_data,
                                            np.arange(1, num_labels+1,
                                            dtype=np.int32),
                                            dummy, maxposs, maxis, npixs)

        # Here we remove the labels that correspond to islands below the
        # detection threshold.
        above_det_thr = maxis > detectionthresholdmap[maxposs[:, 0],
                                                      maxposs[:, 1]]

        num_islands_above_detection_threshold = above_det_thr.sum()

        # np.arange(1, num_labels + 1)) to discard the zero label
        # (background).
        # Will break in the pathological case that all the image pixels
        # are covered by sources, but we will take that risk.
        labels_above_det_thr = np.extract(above_det_thr, np.arange(1,
                                          num_labels + 1))

        maxposs_above_det_thr = np.compress(above_det_thr, maxposs, axis=0)
        maxis_above_det_thr = np.extract(above_det_thr, maxis)
        npixs_above_det = np.extract(above_det_thr, npixs)
        all_indices_above_det_thr = np.compress(above_det_thr, all_indices,
                                                axis=0)

        print(f"Number of sources = {num_islands_above_detection_threshold}")

        return (labels_above_det_thr, labelled_data,
                num_islands_above_detection_threshold, maxposs_above_det_thr,
                maxis_above_det_thr, npixs_above_det, all_indices_above_det_thr,
                slices)

    @staticmethod
    def fit_islands(fudge_max_pix_factor, max_pix_variance_factor, beamsize,
                    correlation_lengths, fixed, island):
        """This function was created to enable the use of 'partial' such that
        we can parallellize source measurements"""
        return island.fit(fudge_max_pix_factor, max_pix_variance_factor, beamsize,
                          correlation_lengths, fixed=fixed)

    # “The sliced_to_indices function has been accelerated by a factor 2.7
    # using ChatGPT 4.0.  Its AI-output has been verified for correctness,
    # accuracy and completeness, adapted where needed, and approved by the
    # author
    @staticmethod
    def slices_to_indices(slices):
        """Convert the list of tuples of slices generated by
        scipy.ndimage.find_objects into a 2D int32 array with number of rows
        equal to the number of islands and 4 columns, i.e 4 integers per island,
        containing the same information as the slices, but more suitable for
        compilation by Numba"""
        num_slices = len(slices)
        all_indices = np.empty((num_slices, 4), dtype=np.int32)

        # Extract start and stop indices for rows and columns
        all_indices[:, 0] = [s[0].start for s in slices]  # Row start
        all_indices[:, 1] = [s[0].stop for s in slices]   # Row stop
        all_indices[:, 2] = [s[1].start for s in slices]  # Column start
        all_indices[:, 3] = [s[1].stop for s in slices]   # Column stop

        return all_indices

    @staticmethod
    @guvectorize([(float32[:, :], int32[:], int32[:, :], int32[:], int32[:],
                   int32[:], float32[:], int32[:])], '(n, m), (l), (n, m), ' +
                 '(), (k) -> (k), (), ()')
    def extract_parms_image_slice(some_image, inds, labelled_data, label, dummy,
                                  maxpos, maxi, npix):
        """
        For an island, indicated by a group of pixels with the same label,
        find the highest pixel value and its position, first relative to the
        upper left corner of the rectangular slice encompassing the island,
        but finally relative to the upper left corner of the image, i.e. the
        [0, 0] position of the Numpy array with all the image pixel values.
        Also, derive the number of pixels of the island.
    
        Parameters
        ----------
        some_image : np.ndarray
            2D array with all the pixel values, typically 
            self.data_bgsubbed.data.
        inds : np.ndarray
            Array of four indices indicating the slice encompassing an island.
            Such a slice would typically be a pick from a list of slices from a
            call to scipy.ndimage.find_objects. Since we are attempting 
            vectorized processing here, the slice should have been replaced by 
            its four coordinates through a call to slices_to_indices.
        labelled_data : np.ndarray
            Array with the same shape as some_image, with labelled islands with
            integer values and zeroes for all background pixels.
        label : int
            The label (integer value) corresponding to the slice encompassing
            the island. Or actually it should be the other way round, since there
            can be multiple islands within one rectangular slice.
        dummy : np.ndarray
            Artefact of the implementation of guvectorize: Empty array with the
            same shape as maxpos. It is needed because of a missing feature in
            guvectorize: There is no other way to tell guvectorize what the shape
            of the output array will be. Therefore, we define an otherwise
            redundant input array with the same shape as the desired output
            array. Defined as int32, but could be any type.
        maxpos : np.ndarray
            Array of two integers indicating the indices of the highest pixel
            value of the island with label = label relative to the position of
            pixel [0, 0] of the image.
        maxi : np.float32
            Float32 equal to the highest pixel value of the island with label =
            label.
        npix : np.int32
            Integer indicating the number of pixels of the island.
    
        Returns
        -------
        None
            No return values, because of the use of the guvectorize decorator:
            'guvectorize() functions don’t return their result value: they take
            it as an array argument, which must be filled in by the function'. In
            this case maxpos, maxi and npix will be filled with values.
        """

        labelled_data_chunk = labelled_data[inds[0]:inds[1], inds[2]:inds[3]]
        image_chunk = some_image[inds[0]:inds[1], inds[2]:inds[3]]
        segmented_island = np.where(labelled_data_chunk == label[0], 1, 0)

        selected_data = segmented_island * image_chunk
        maxpos_flat = selected_data.argmax()
        maxpos[0] = np.floor_divide(maxpos_flat, selected_data.shape[1])
        maxpos[1] = np.mod(maxpos_flat, selected_data.shape[1])
        maxi[0] = selected_data[maxpos[0], maxpos[1]]
        maxpos[0] += inds[0]
        maxpos[1] += inds[2]
        npix[0] = int32(segmented_island.sum())

    def _pyse(
            self, detectionthresholdmap, analysisthresholdmap,
            deblend_nthresh, force_beam, labelled_data=None,
            labels=np.array([], dtype=np.int32)):
        """
        Run Python-based source extraction on this image.
    
        Parameters
        ----------
        detectionthresholdmap : np.ma.MaskedArray
            2D array of floats with the same shape as the observational image
            (self.rawdata). The detection threshold map imposes an extra
            threshold for source detection and is therefore higher than the
            analysis threshold map.
        analysisthresholdmap : np.ma.MaskedArray
            2D array of floats with the same shape as the observational image
            (self.rawdata). The analysis threshold map imposes the primary
            threshold for source detection. It is lower (or equal) than the
            detection threshold map, or else we would be left with too few
            pixels for proper source shape measurements, in some cases.
        deblend_nthresh : int
            Number of subthresholds for deblending. 0 disables.
        force_beam : bool
            Force all extractions to have major/minor axes and position angle
            equal to the restoring beam.
        labelled_data : np.ndarray, optional, default=None
            Labelled island map (output of np.ndimage.label()). Will be
            calculated automatically if not provided.
        labels : np.ndarray, optional, default=np.array([], dtype=np.int32)
            Array of integers representing the labels in the island map to use
            for fitting.
    
        Returns
        -------
        .utility.containers.ExtractionResults
            Results of the source extraction.
    
        Notes
        -----
        This is described in detail in the "LOFAR Transients Pipeline" article
        by John D. Swinbank et al., see
        https://doi.org/10.1016/j.ascom.2015.03.002
        """
        # Map our chunks onto a list of islands.

        if labelled_data is None:
            (labels, labelled_data, num_islands, maxposs, maxis, npixs,
             indices, slices) =\
                self.label_islands(detectionthresholdmap,
                                   analysisthresholdmap, deblend_nthresh
                                   )

        num_islands = len(labels)

        results = containers.ExtractionResults()

        if deblend_nthresh or force_beam or not VECTORIZED:
            island_list = []
            for label in labels:
                chunk = slices[label - 1]
                analysis_threshold = (analysisthresholdmap[chunk] /
                                      self.rmsmap[chunk]).max()
                # In selected_data only the pixels with the "correct"
                # (see above) labels are retained. Other pixel values are
                # set to -(bignum).
                # In this way, disconnected pixels within (rectangular)
                # slices around islands (particularly the large ones) do
                # not affect the source measurements.
                selected_data = np.ma.where(
                    labelled_data[chunk] == label,
                    self.data_bgsubbed[chunk].data, -extract.BIGNUM
                ).filled(fill_value=-extract.BIGNUM)

                island_list.append(
                    extract.Island(
                        selected_data,
                        self.rmsmap[chunk],
                        chunk,
                        analysis_threshold,
                        detectionthresholdmap[chunk],
                        self.beam,
                        deblend_nthresh,
                        DEBLEND_MINCONT,
                        STRUCTURING_ELEMENT
                    )
                )

            if self.islands:
                self.Gaussian_islands = np.zeros(self.data.shape)
            # If required, we can save the 'leftovers' from the deblending and
            # fitting processes for later analysis. This needs setting up here:
            if self.residuals:
                self.Gaussian_residuals = np.zeros(self.data.shape)
                self.residuals_from_deblending = np.zeros(self.data.shape)
                for island in island_list:
                    self.residuals_from_deblending[island.chunk] += (
                        island.data.filled(fill_value=0.))

            # Deblend each of the islands to its consituent parts, if necessary
            if deblend_nthresh:
                deblended_list = [x.deblend() for x in island_list]
                island_list = list(utils.flatten(deblended_list))

            # Set up the fixed fit parameters if 'force beam' is on:
            if force_beam:
                fixed = {'semimajor': self.beam[0],
                         'semiminor': self.beam[1],
                         'theta': self.beam[2]}
            else:
                fixed = None

            # Iterate over the list of islands and measure the source in each,
            # appending it to the results list.
            with Pool(psutil.cpu_count()) as p:
                fit_islands_partial = partial(ImageData.fit_islands,
                                              self.fudge_max_pix_factor,
                                              self.max_pix_variance_factor,
                                              self.beamsize,
                                              self.correlation_lengths, fixed)
                fit_results = p.map(fit_islands_partial, island_list)

            for island, fit_result in zip(island_list, fit_results):
                if fit_result:
                    measurement, Gauss_island, Gauss_residual = fit_result
                else:
                    # Failed to fit; drop this island and go to the next.
                    continue
                try:
                    det = extract.Detection(measurement, self,
                                            chunk=island.chunk,
                                            eps_ra=self.eps_ra,
                                            eps_dec=self.eps_dec)
                    if (det.ra.error == float('inf') or
                            det.dec.error == float('inf')):
                        logger.warning('Bad fit from blind extraction at pixel coords:'
                                       '%f %f - measurement discarded'
                                       '(increase fitting margin?)', det.x, det.y)
                    else:
                        results.append(det)
                except RuntimeError as e:
                    logger.error("Island not processed; unphysical?")

                if self.islands:
                    self.Gaussian_islands[island.chunk] += Gauss_island

                if self.residuals:
                    self.residuals_from_deblending[island.chunk] -= (
                        island.data.filled(fill_value=0.))
                    self.Gaussian_residuals[island.chunk] += Gauss_residual

        elif num_islands > 0:

            (moments_of_sources, sky_barycenters, ra_errors, dec_errors,
             error_radii, smaj_asec, errsmaj_asec, smin_asec, errsmin_asec,
             theta_celes_values, theta_celes_errors, theta_dc_celes_values,
             theta_dc_celes_errors, Gaussian_islands, Gaussian_residuals,
             sig, chisq, reduced_chisq) = \
                extract.source_measurements_pixels_and_celestial_vectorised(
                    num_islands, npixs, maxposs, maxis, self.data_bgsubbed.data,
                    self.rmsmap.data, analysisthresholdmap.data, indices,
                    labelled_data, labels, self.wcs, self.fudge_max_pix_factor,
                    self.max_pix_variance_factor,  self.beam, self.beamsize,
                    self.correlation_lengths, self.eps_ra, self.eps_dec)

            if self.islands:
                self.Gaussian_islands = Gaussian_islands

            if self.residuals:
                self.Gaussian_residuals = Gaussian_residuals

            for count, label in enumerate(labels):
                chunk = slices[label - 1]

                param = extract.ParamSet()
                param["sig"] = maxis[count] / self.rmsmap.data[tuple(maxposs[count])]

                param["peak"] = Uncertain(moments_of_sources[count, 0, 0],
                                          moments_of_sources[count, 1, 0])
                param["flux"] = Uncertain(moments_of_sources[count, 0, 1],
                                          moments_of_sources[count, 1, 1])
                param["xbar"] = Uncertain(moments_of_sources[count, 0, 2],
                                          moments_of_sources[count, 1, 2])
                param["ybar"] = Uncertain(moments_of_sources[count, 0, 3],
                                          moments_of_sources[count, 1, 3])
                param["semimajor"] = Uncertain(moments_of_sources[count, 0, 4],
                                               moments_of_sources[count, 1, 4])
                param["semiminor"] = Uncertain(moments_of_sources[count, 0, 5],
                                               moments_of_sources[count, 1, 5])
                param["theta"] = Uncertain(moments_of_sources[count, 0, 6],
                                           moments_of_sources[count, 1, 6])
                param["semimaj_deconv"] = Uncertain(moments_of_sources[count, 0, 7],
                                                    moments_of_sources[count, 1, 7])
                param["semimin_deconv"] = Uncertain(moments_of_sources[count, 0, 8],
                                                    moments_of_sources[count, 1, 8])
                param["theta_deconv"] = Uncertain(moments_of_sources[count, 0, 9],
                                                  moments_of_sources[count, 1, 9])

                det = extract.Detection(param, self, chunk=chunk)
                results.append(det)

        def is_usable(det):
            """
            Check that both ends of each axis are usable; that is, that they
            fall within an unmasked part of the image. The axis will not likely
            fall exactly on a pixel number, so check all the surroundings.
            """
            def check_point(x, y):
                x = (int(x), int(np.ceil(x)))
                y = (int(y), int(np.ceil(y)))
                for position in itertools.product(x, y):
                    try:
                        if self.data.mask[position[0], position[1]]:
                            # Point falls in mask
                            return False
                    except IndexError:
                        # Point falls completely outside image
                        return False
                # Point is ok
                return True

            for point in (
                    (det.start_smaj_x, det.start_smaj_y),
                    (det.start_smin_x, det.start_smin_y),
                    (det.end_smaj_x, det.end_smaj_y),
                    (det.end_smin_x, det.end_smin_y)
            ):
                if not check_point(*point):
                    logger.debug("Unphysical source at pixel %f, %f" % (
                        det.x.value, det.y.value))
                    return False
            return True
        # Filter will return a list; ensure we return an ExtractionResults.
        return containers.ExtractionResults(list(filter(is_usable, results)))
