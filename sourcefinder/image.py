"""
Some generic utility routines for number handling and
calculating (specific) variances
"""

import itertools
import logging

import numpy
import time

from sourcefinder import extract
from sourcefinder import stats
from sourcefinder import utils
from sourcefinder.utility import containers
from sourcefinder.utility import coordinates
from sourcefinder.utility.uncertain import Uncertain
import dask.array as da
from scipy.interpolate import interp1d
import psutil
from multiprocessing import Pool
from functools import cached_property
from functools import partial
import sep

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

# Let's retain the option of calculating background maps in the original way,
# i.e. without using sep. This is slower, but more accurate. Ultimately, the
# switch for choosing either should be propagated through an argument of the 
# ImageData class instantiation, but we will implement that later. For now,
# we will not use SEP as default.
SEP = False
# Vectorized processing of source islands is much faster, but excludes Gaussian
# fits, therefore slightly less accurate.
VECTORIZED = False


class ImageData(object):
    """Encapsulates an image in terms of a numpy array + meta/headerdata.

    This is your primary contact point for interaction with images: it icludes
    facilities for source extraction and measurement, etc.
    """

    def __init__(self, data, beam, wcs, margin=0, radius=0, back_size_x=32,
                 back_size_y=32, residuals=False, islands=False):
        """Sets up an ImageData object.

        *Args:*
          - data (2D numpy.ndarray): actual image data
          - wcs (utility.coordinates.wcs): world coordinate system
            specification
          - beam (3-tuple): beam shape specification as
            (semimajor, semiminor, theta)

        """

        # Do data, wcs and beam need deepcopy?
        # Probably not (memory overhead, in particular for data),
        # but then the user shouldn't change them outside ImageData in the
        # meantime
        # self.rawdata is a 2D numpy array, C-contiguous needed for sep.
        # single precision is good enough in all cases.
        self.rawdata = numpy.ascontiguousarray(data, dtype=numpy.float32)
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
    def background(self):
        """"Returns background object from sep"""
        return sep.Background(self.data.data, mask = self.data.mask,
                              bw=self.back_size_x, bh=self.back_size_y, fw=0, fh=0)

    @cached_property
    @timeit
    def backmap(self):
        """Background map"""
        if not hasattr(self, "_user_backmap"):
            if SEP:
                return numpy.ma.array(self.background.back(), mask=self.data.mask)
            else:
                return self._interpolate(self.grids['bg'])
        else:
            return self._user_backmap

    @cached_property
    @timeit
    def rmsmap(self):
        """RMS map"""
        if not hasattr(self, "_user_noisemap"):
            if SEP:
                return numpy.ma.array(self.background.rms(), mask=self.data.mask)
            else:
                return self._interpolate(self.grids['rms'], roundup=True)
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
        mask = numpy.zeros((self.xdim, self.ydim))
        if self.margin:
            margin_mask = numpy.ones((self.xdim, self.ydim))
            margin_mask[self.margin:-self.margin, self.margin:-self.margin] = 0
            mask = numpy.logical_or(mask, margin_mask)
        if self.radius:
            radius_mask = utils.circular_mask(self.xdim, self.ydim, self.radius)
            mask = numpy.logical_or(mask, radius_mask)
        mask = numpy.logical_or(mask, numpy.isnan(self.rawdata))
        return numpy.ma.array(self.rawdata, mask=mask)

    @cached_property
    def data_bgsubbed(self):
        """Background subtracted masked image data"""
        return self.data - self.backmap

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
            del self.background
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
        useful_chunk = ndimage.find_objects(numpy.where(self.data.mask, 0, 1))
        assert (len(useful_chunk) == 1)
        y_dim = self.data[useful_chunk[0]].data.shape[1]
        useful_data = da.from_array(self.data[useful_chunk[0]].data, chunks=(self.back_size_x, y_dim))

        mode_and_rms = useful_data.map_blocks(ImageData.compute_mode_and_rms_of_row_of_subimages,
                                              y_dim,  self.back_size_y,
                                              dtype=numpy.complex64,
                                              chunks=(1, 1)).compute()

        # See also similar comment below. This solution was chosen because map_blocks does not seem to be able to
        # output multiple arrays. One can however output to a complex array and take real and imaginary
        # parts afterward. Not a very clean solution, I admit.
        mode_grid = mode_and_rms.real
        rms_grid = mode_and_rms.imag

        rms_grid = numpy.ma.array(
            rms_grid, mask=numpy.where(rms_grid == 0, 1, 0))
        # A rms of zero is not physical, since any instrument has system noise, so I use that as criterion
        # to mask values. A zero background mode is physically possible, but also highly unlikely, given the way
        # we determine it.
        mode_grid = numpy.ma.array(
            mode_grid, mask=numpy.where(rms_grid == 0, 1, 0))

        return { 'bg': mode_grid, 'rms': rms_grid,}

    @staticmethod
    def compute_mode_and_rms_of_row_of_subimages(row_of_subimages, y_dim, back_size_y):

        # We set up a dedicated logging subchannel, as the sigmaclip loop
        # logging is very chatty:
        sigmaclip_logger = logging.getLogger(__name__ + '.sigmaclip')
        row_of_complex_values = numpy.empty(0, numpy.complex64)

        for starty in range(0, y_dim, back_size_y):
            chunk = row_of_subimages[:, starty:starty+back_size_y]
            if not chunk.any():
                # In the original code we had rmsrow.append(False), but now we work with an array instead of a list,
                # so I'll set these values to zero instead and use these zeroes to create the mask.
                rms = 0
                mode = 0
            else:
                chunk, sigma, median, num_clip_its = stats.sigma_clip(
                    chunk.ravel())
                if len(chunk) == 0 or not chunk.any():
                    rms = 0
                    mode = 0
                else:
                    mean = numpy.mean(chunk)
                    rms = sigma
                    # In the case of a crowded field, the distribution will be
                    # skewed and we take the median as the background level.
                    # Otherwise, we take 2.5 * median - 1.5 * mean. This is the
                    # same as SExtractor: see discussion at
                    # <http://terapix.iap.fr/forum/showthread.php?tid=267>.
                    # (mean - median) / sigma is a quick n' dirty skewness
                    # estimator devised by Karl Pearson.
                    if numpy.fabs(mean - median) / sigma >= 0.3:
                        sigmaclip_logger.debug(
                            'bg skewed, %f clipping iterations', num_clip_its)
                        mode=median
                    else:
                        sigmaclip_logger.debug(
                            'bg not skewed, %f clipping iterations',
                            num_clip_its)
                        mode=2.5 * median - 1.5 * mean
            row_of_complex_values = numpy.append(row_of_complex_values,  numpy.array(mode + 1j*rms))[None]
        # This solution is a bit dirty. I would like dask.array.map_blocks to output two arrays,
        # but presently that module does not seem to provide for that. But I can, however, output to a
        # complex array and later take the real part of that for the mode and the imaginary part
        # for the rms.
        return row_of_complex_values

    def _interpolate(self, grid, roundup=False):
        """
        Interpolate a grid to produce a map of the dimensions of the image.

        Args:

            grid (numpy.ma.MaskedArray)

        Kwargs:

            roundup (bool)

        Returns:

            (numpy.ma.MaskedArray)

        Used to transform the RMS, background or FDR grids produced by
        L{_grids()} to a map we can compare with the image data.

        If roundup is true, values of the resultant map which are lower than
        the input grid are trimmed.
        """
        # there's no point in working with the whole of the data array if it's
        # masked.
        useful_chunk = ndimage.find_objects(numpy.where(self.data.mask, 0, 1))
        assert (len(useful_chunk) == 1)
        my_xdim, my_ydim = self.data[useful_chunk[0]].shape

        if MEDIAN_FILTER:
            f_grid = ndimage.median_filter(grid, MEDIAN_FILTER)
            if MF_THRESHOLD:
                grid = numpy.where(
                    numpy.fabs(f_grid - grid) > MF_THRESHOLD, f_grid, grid
                )
            else:
                grid = f_grid

        # Bicubic spline interpolation
        xratio = float(my_xdim) / self.back_size_x
        yratio = float(my_ydim) / self.back_size_y

        my_map = numpy.ma.MaskedArray(numpy.zeros(self.data.shape),
                                      mask=self.data.mask)

        # Remove the MaskedArrayFutureWarning warning and keep old numpy < 1.11
        # behavior
        my_map.unshare_mask()

        # Inspired by https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
        # Should be much faster than scipy.ndimage.map_coordinates.
        # scipy.ndimage.zoom should also be an option for speedup, but zoom dit not let me produce the exact
        # same output as map_coordinates. My bad.
        # I checked, using fitsdiff, that it gives the exact same output as the original code
        # up to and including --relative-tolerance=1e-15 for INTERPOLATE_ORDER=1.
        # It was actually quite a hassle to get the same output and the fill_value is essential
        # in interp1d. However, for some unit tests, grid.shape=(1,1) and then it will break
        # with "ValueError: x and y arrays must have at least 2 entries". So in that case
        # map_coordinates should be used.

        if INTERPOLATE_ORDER==1 and grid.shape[0]>1 and grid.shape[1]>1:
            x_initial = numpy.linspace(0., grid.shape[0]-1, grid.shape[0], endpoint=True)
            y_initial = numpy.linspace(0., grid.shape[1]-1, grid.shape[1], endpoint=True)
            x_sought = numpy.linspace(-0.5, -0.5 + xratio, my_xdim, endpoint=True)
            y_sought = numpy.linspace(-0.5, -0.5 + yratio, my_ydim, endpoint=True)

            primary_interpolation = interp1d(y_initial, grid, kind='slinear', assume_sorted=True,
                                             axis=1, copy=False, bounds_error=False,
                                             fill_value=(grid[:, 0], grid[:, -1]))
            transposed = primary_interpolation(y_sought).T

            perpendicular_interpolation = interp1d(x_initial, transposed, kind='slinear', assume_sorted=True,
                                                   axis=1, copy=False, bounds_error=False,
                                                   fill_value=(transposed[:, 0], transposed[:, -1]))
            my_map[useful_chunk[0]] = perpendicular_interpolation(x_sought).T
        else:
            slicex = slice(-0.5, -0.5 + xratio, 1j * my_xdim)
            slicey = slice(-0.5, -0.5 + yratio, 1j * my_ydim)
            my_map[useful_chunk[0]] = ndimage.map_coordinates(
               grid, numpy.mgrid[slicex, slicey],
               mode='nearest', order=INTERPOLATE_ORDER)

        # If the input grid was entirely masked, then the output map must
        # also be masked: there's no useful data here. We don't search for
        # sources on a masked background/RMS, so this data will be cleanly
        # skipped by the rest of the sourcefinder
        if numpy.ma.getmask(grid).all():
            my_map.mask = True
        elif roundup:
            # In some cases, the spline interpolation may produce values
            # lower than the minimum value in the map. If required, these
            # can be trimmed off. No point doing this if the map is already
            # fully masked, though.
            my_map = numpy.ma.MaskedArray(
                data=numpy.where(
                    my_map >= numpy.min(grid), my_map, numpy.min(grid)),
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

    def extract(self, det, anl, noisemap=None, bgmap=None, labelled_data=None,
                labels=None, deblend_nthresh=0, force_beam=False):

        """
        Kick off conventional (ie, RMS island finding) source extraction.

        Kwargs:

            det (float): detection threshold, as a multiple of the RMS
                noise. At least one pixel in a source must exceed this
                for it to be regarded as significant.

            anl (float): analysis threshold, as a multiple of the RMS
                noise. All the pixels within the island that exceed
                this will be used when fitting the source.

            noisemap (numpy.ndarray):

            bgmap (numpy.ndarray):

            deblend_nthresh (int): number of subthresholds to use for
                deblending. Set to 0 to disable.

            force_beam (bool): force all extractions to have major/minor axes
                equal to the restoring beam

        Returns:
             :class:`sourcefinder.utility.containers.ExtractionResults`
        """

        if anl > det:
            logger.warning(
                "Analysis threshold is higher than detection threshold"
            )

        # If the image data is flat we may as well crash out here with a
        # sensible error message, otherwise the RMS estimation code will
        # crash out with a confusing error later.
        if numpy.ma.max(self.data) == numpy.ma.min(self.data):
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

        Obviously, there should be no sources in the negative image, so this
        tells you about the false positive rate.

        We need to clear cached data -- backgroung map, cached clips, etc --
        before & after doing this, as they'll interfere with the normal
        extraction process. If this is regularly used, we'll want to
        implement a separate cache.
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
        """False Detection Rate based source extraction.
        The FDR procedure guarantees that <FDR> < alpha.

        See `Hopkins et al., AJ, 123, 1086 (2002)
        <http://adsabs.harvard.edu/abs/2002AJ....123.1086H>`_.
        """

        # The correlation length in config.py is used not only for the
        # calculation of error bars with the Condon formulae, but also for
        # calculating the number of independent pixels.
        corlengthlong, corlengthshort = self.correlation_lengths

        C_n = (1.0 / numpy.arange(
            round(0.25 * numpy.pi * corlengthlong *
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

        n1 = numpy.sqrt(2 * numpy.pi)
        prob = numpy.sort(
            numpy.ravel(numpy.exp(-0.5 * normalized_data ** 2) / n1))
        lengthprob = float(len(prob))
        compare = (alpha / C_n) * numpy.arange(lengthprob + 1)[1:] / lengthprob
        # Find the last undercrossing, see, e.g., fig. 9 in Miller et al., AJ
        # 122, 3492 (2001).  Searchsorted is not used because the array is not
        # sorted.
        try:
            index = (numpy.where(prob - compare < 0.)[0]).max()
        except ValueError:
            # Everything below threshold
            return containers.ExtractionResults()

        fdr_threshold = numpy.sqrt(-2.0 * numpy.log(n1 * prob[index]))
        # Default we require that all source pixels are above the threshold,
        # not only the peak pixel.  This gives a better guarantee that indeed
        # the fraction of false positives is less than fdr_alpha in config.py.
        # See, e.g., Hopkins et al., AJ 123, 1086 (2002).
        if not anl:
            anl = fdr_threshold
        return self._pyse(fdr_threshold * self.rmsmap, anl * self.rmsmap,
                          deblend_nthresh, force_beam)

    def flux_at_pixel(self, x, y, numpix=1):
        """Return the background-subtracted flux at a certain position
        in the map"""

        # numpix is the number of pixels to look around the target.
        # e.g. numpix = 1 means a total of 9 pixels, 1 in each direction.
        return self.data_bgsubbed[y - numpix:y + numpix + 1,
               x - numpix:x + numpix + 1].max()

    @staticmethod
    def box_slice_about_pixel(x, y, box_radius):
        """
        Returns a slice centred about (x,y), of width = 2*int(box_radius) + 1
        """
        ibr = int(box_radius)
        x = int(x)
        y = int(y)
        return (slice(x - ibr, x + ibr + 1),
                slice(y - ibr, y + ibr + 1))

    def fit_to_point(self, x, y, boxsize, threshold, fixed):
        """Fit an elliptical Gaussian to a specified point on the image.

        The fit is carried on a square section of the image, of length
        *boxsize* & centred at pixel coordinates *x*, *y*. Any data
        below *threshold* * rmsmap is not used for fitting. If *fixed*
        is set to ``position``, then the pixel coordinates are fixed
        in the fit.

        Returns an instance of :class:`sourcefinder.extract.Detection`.
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
                    hasattr(numpy.ma.core, "MaskedConstant") and
                    isinstance(self.rmsmap, numpy.ma.core.MaskedConstant)
            ) or (
                # Old NumPy
                numpy.ma.is_masked(self.rmsmap[int(x), int(y)])
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
                        numpy.where(
                            self.data_bgsubbed > threshold * self.rmsmap, 1, 0
                        )
                    )
                )
            )

            mylabel = labels[int(x), int(y)]
            if mylabel == 0:  # 'Background'
                raise ValueError(
                    "Fit region is below specified threshold, fit aborted.")
            mask = numpy.where(labels[chunk] == mylabel, 0, 1)
            fitme = numpy.ma.array(self.data_bgsubbed[chunk], mask=mask)
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
        Convenience function to fit a list of sources at the given positions

        This function wraps around fit_to_point().

        Args:
            positions (tuple): list of (RA, Dec) tuples. Positions to be fit,
                in decimal degrees.
            boxsize: See :py:func:`fit_to_point`
            threshold: as above.
            fixed: as above.
            ids (tuple): A list of identifiers. If not None, then must match
                the length and order of the ``requested_fits``. Any
                successfully fit positions will be returned in a tuple
                along with the matching id. As these are simply passed back to
                calling code they can be a string, tuple or whatever.

        In particular, boxsize is in pixel coordinates as in
        fit_to_point, not in sky coordinates.

        Returns:
            tuple: A list of successful fits.
                If ``ids`` is None, returns a single list of
                :class:`sourcefinder.extract.Detection` s.
                Otherwise, returns a tuple of two matched lists:
                ([detections], [matching_ids]).
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

    def label_islands(self, detectionthresholdmap, analysisthresholdmap, deblend_nthresh):
        """
        Return a lablled array of pixels for fitting.

        Args:

            detectionthresholdmap (numpy.ndarray):

            analysisthresholdmap (numpy.ndarray):

            deblend_nthresh: number of thresholds for deblending (integer)

        Returns:

            list of valid islands (list of int)

            labelled islands (numpy.ndarray)
        """
        # If there is no usable data, we return an empty set of islands.
        if not len(self.rmsmap.compressed()):
            logging.warning("RMS map masked; sourcefinding skipped")
            return [], numpy.zeros(self.data_bgsubbed.shape, dtype=numpy.int)

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
        # combined_mask = numpy.logical_or(self.rmsmap.data < RMS_FILTER * self.background.globalrms,
        #                                 self.data.mask)

        clipped_data = numpy.ma.where(
            (self.data_bgsubbed > analysisthresholdmap) &
            (self.rmsmap >= (RMS_FILTER * self.background.globalrms)),
            1, 0
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
        maxposs = numpy.empty((num_labels, 2), dtype=numpy.int32)
        dummy = numpy.empty_like(maxposs)
        maxis = numpy.empty(num_labels, dtype=numpy.float32)
        npixs = numpy.empty(num_labels, dtype=numpy.int32)

        ImageData.extract_parms_image_slice(self.data_bgsubbed.data.astype(
                                            dtype=numpy.float32, copy=False),
                                            all_indices, labelled_data,
                                            numpy.arange(1, num_labels+1,
                                            dtype=numpy.int32),
                                            dummy, maxposs, maxis, npixs)

        # Here we remove the labels that correspond to islands below the
        # detection threshold.
        above_det_thr = maxis > detectionthresholdmap[maxposs[:, 0],
                                                      maxposs[:, 1]]

        num_islands_above_detection_threshold = above_det_thr.sum()

        # numpy.arange(1, num_labels + 1)) to discard the zero label
        # (background).
        # Will break in the pathological case that all the image pixels
        # are covered by sources, but we will take that risk.
        labels_above_det_thr = numpy.extract(above_det_thr, numpy.arange(1,
                                             num_labels + 1))

        maxposs_above_det_thr = numpy.compress(above_det_thr, maxposs, axis=0)
        maxis_above_det_thr = numpy.extract(above_det_thr, maxis)
        npixs_above_det = numpy.extract(above_det_thr, npixs)
        all_indices_above_det_thr = numpy.compress(above_det_thr, all_indices,
                                                   axis=0)

        print(f"Number of sources = {num_islands_above_detection_threshold}")

        return (labels_above_det_thr, labelled_data,
                num_islands_above_detection_threshold, maxposs_above_det_thr,
                maxis_above_det_thr, npixs_above_det, all_indices_above_det_thr,
                slices)

    @staticmethod
    def fit_islands(fudge_max_pix_factor, max_pix_variance_factor, beamsize,
                    correlation_lengths, fixed, island):
        return island.fit(fudge_max_pix_factor, max_pix_variance_factor, beamsize,
                          correlation_lengths, fixed=fixed)

    @staticmethod
    def slices_to_indices(slices):
        all_indices = numpy.empty((len(slices), 4), dtype=numpy.int32)
        for i in range(len(slices)):
            some_slice = slices[i]
            all_indices[i, :] = numpy.array([some_slice[0].start,
                                             some_slice[0].stop,
                                             some_slice[1].start,
                                             some_slice[1].stop])
        return all_indices

    @staticmethod
    @guvectorize([(float32[:, :], int32[:], int32[:, :], int32[:], int32[:],
                 int32[:], float32[:], int32[:])], '(n, m), (l), (n, m), ' +
                 '(), (k) -> (k), (), ()')
    def extract_parms_image_slice(some_image, inds, labelled_data, label,
                                  dummy, maxpos, maxi, npix):
        """
        For an island, indicated by a group of pixels with the same label,
        find the highest pixel value and its position, first relative to the
        upper left corner of the rectangular slice encompassing the island,
        but finally relative to the upper left corner of the image, i.e. the
        [0, 0] position of the Numpy array with all the image pixel values.
        Also, derive the number of pixels of the island.

        :param some_image: The 2D ndarray with all the pixel values, typically
                           self.data_bgsubbed.data.

        :param inds: A ndarray of four indices indicating the slice encompassing
                     an island. Such a slice would typically be a pick from a
                     list of slices from a call to scipy.ndimage.find_objects.
                     Since we are attempting vectorized processing here, the
                     slice should have been replaced by its four coordinates
                     through a call to slices_to_indices.

        :param labelled_data: A ndarray with the same shape as some_image, with
                              labelled islands with integer values and zeroes
                              for all background pixels.

        :param label: The label (integer value) corresponding to the slice
                      encompassing the island. Or actually it should be the
                      other way round, since there can be multiple islands
                      within one rectangular slice.

        :param dummy: Artefact of the implementation of guvectorize:
                      Empty array with the same shape as maxpos. It is needed
                      because of a missing feature in guvectorize: There is no
                      other way to tell guvectorize what the shape of the
                      output array will be. Therefore, we define an otherwise
                      redundant input array with the same shape as the desired
                      output array. Defined as int32, but could be any type.

        :param maxpos: Ndarray of two integers indicating the indices of the
                       highest pixel value of the island with label = label
                       relative to the position of pixel [0, 0] of the image.

        :param maxi:  Float32 equal to the highest pixel value
                      of the island with label = label.

        :param npix:  Integer indicating the number of pixels of the island.

        :return:      No return values, because of the use of the guvectorize
                      decorator: 'guvectorize() functions don’t return their
                      result value: they take it as an array argument,
                      which must be filled in by the function'. In this case
                      maxpos, maxi and npix will be filled with values.
        """

        labelled_data_chunk = labelled_data[inds[0]:inds[1], inds[2]:inds[3]]
        image_chunk = some_image[inds[0]:inds[1], inds[2]:inds[3]]
        segmented_island = numpy.where(labelled_data_chunk == label[0], 1, 0)

        selected_data = segmented_island * image_chunk
        maxpos_flat = selected_data.argmax()
        maxpos[0] = numpy.floor_divide(maxpos_flat, selected_data.shape[1])
        maxpos[1] = numpy.mod(maxpos_flat, selected_data.shape[1])
        maxi[0] = selected_data[maxpos[0], maxpos[1]]
        maxpos[0] += inds[0]
        maxpos[1] += inds[2]
        npix[0] = int32(segmented_island.sum())

    @timeit
    def _pyse(
            self, detectionthresholdmap, analysisthresholdmap,
            deblend_nthresh, force_beam, labelled_data=None, labels=[],
            eps_ra=0, eps_dec=0
    ):
        """
        Run Python-based source extraction on this image.

        Args:

            detectionthresholdmap (numpy.ndarray):

            analysisthresholdmap (numpy.ndarray):

            deblend_nthresh (int): number of subthresholds for deblending. 0
                disables.

            force_beam (bool): force all extractions to have major/minor axes
                equal to the restoring beam

            labelled_data (numpy.ndarray): labelled island map (output of
            numpy.ndimage.label()). Will be calculated automatically if not
            provided.

            labels (tuple): list of labels in the island map to use for
            fitting.

        Returns:

            (.utility.containers.ExtractionResults):

        This is described in detail in the "Source Extraction System" document
        by John Swinbank, available from TKP svn.
        """
        # Map our chunks onto a list of islands.

        if labelled_data is None:
            (labels, labelled_data, num_islands, maxposs, maxis, npixs,
             indices, slices) =\
                self.label_islands(detectionthresholdmap,
                                   analysisthresholdmap, deblend_nthresh
                                   )

        start_post_labelling = time.time()

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
                selected_data = numpy.ma.where(
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
                self.Gaussian_islands = numpy.zeros(self.data.shape)
            # If required, we can save the 'leftovers' from the deblending and
            # fitting processes for later analysis. This needs setting up here:
            if self.residuals:
                self.Gaussian_residuals = numpy.zeros(self.data.shape)
                self.residuals_from_deblending = numpy.zeros(self.data.shape)
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
                                            chunk=island.chunk, eps_ra=eps_ra,
                                            eps_dec=eps_dec)
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

            (moments_of_sources, sky_barycenters, xpositions, ypositions,
             ra_errors, dec_errors, error_radii, smaj_asec, errsmaj_asec,
             smin_asec, errsmin_asec, theta_celes_values, theta_celes_errors,
             theta_dc_celes_values, theta_dc_celes_errors) = \
                extract.source_measurements_pixels_and_celestial_vectorised(
                    num_islands, npixs, maxposs, maxis, self.data_bgsubbed.data,
                    self.rmsmap.data, analysisthresholdmap.data, indices,
                    labelled_data, labels, self.wcs, self.fudge_max_pix_factor,
                    self.max_pix_variance_factor,  self.beam, self.beamsize,
                    self.correlation_lengths, eps_ra, eps_dec)

            if self.islands or self.residuals:
                self.Gaussian_islands = numpy.zeros_like(self.data.data)

                # Select the relevant elements of moments_sources, include the
                # peak spectral brightness, but exclude the flux density.
                relevant_moments = (
                    numpy.take(moments_of_sources[:, 0, :],
                               [0, 2, 3, 4, 5, 6], axis=1))

                extract.calculate_Gaussian_islands(indices[:, 0],
                                                   indices[:, 2],
                                                   xpositions, ypositions,
                                                   npixs,
                                                   relevant_moments,
                                                   self.Gaussian_islands)

                if self.residuals:
                    # It only makes sense to compute residuals where we have
                    # reconstructed Gaussian islands, i.e. above the analysis
                    # threshold.
                    # Some parts of self.data_bgsubbed may be masked, but no
                    # sources will have been detected in those masked patches
                    # of the sky, so no need to apply that mask here.
                    self.Gaussian_residuals = \
                        numpy.where(self.Gaussian_islands != 0,
                                    self.data_bgsubbed.data -
                                    self.Gaussian_islands,
                                    0).astype(numpy.float32)

            for count, label in enumerate(labels):
                chunk = slices[label - 1]

                param = extract.ParamSet()
                param.sig = maxis[count] / self.rmsmap.data[tuple(maxposs[count])]

                param["peak"] = Uncertain(moments_of_sources[count, 0, 0], moments_of_sources[count, 1, 0])
                param["flux"] = Uncertain(moments_of_sources[count, 0, 1], moments_of_sources[count, 1, 1])
                param["xbar"] = Uncertain(moments_of_sources[count, 0, 2], moments_of_sources[count, 1, 2])
                param["ybar"] = Uncertain(moments_of_sources[count, 0, 3], moments_of_sources[count, 1, 3])
                param["semimajor"] = Uncertain(moments_of_sources[count, 0, 4], moments_of_sources[count, 1, 4])
                param["semiminor"] = Uncertain(moments_of_sources[count, 0, 5], moments_of_sources[count, 1, 5])
                param["theta"] = Uncertain(moments_of_sources[count, 0, 6], moments_of_sources[count, 1, 6])
                param["semimaj_deconv"] = Uncertain(moments_of_sources[count, 0, 7], moments_of_sources[count, 1, 7])
                param["semimin_deconv"] = Uncertain(moments_of_sources[count, 0, 8], moments_of_sources[count, 1, 8])
                param["theta_deconv"] = Uncertain(moments_of_sources[count, 0, 9], moments_of_sources[count, 1, 9])

                det = extract.Detection(param, self, chunk=chunk)
                results.append(det)

        end_post_labelling = time.time()
        print("Post labelling took {:7.2f} seconds.".format(end_post_labelling-start_post_labelling))

        def is_usable(det):
            # Check that both ends of each axis are usable; that is, that they
            # fall within an unmasked part of the image.
            # The axis will not likely fall exactly on a pixel number, so
            # check all the surroundings.
            def check_point(x, y):
                x = (int(x), int(numpy.ceil(x)))
                y = (int(y), int(numpy.ceil(y)))
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
