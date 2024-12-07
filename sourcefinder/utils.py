"""
This module contain utilities for the source finding routines
"""

import math
import numpy as np
import scipy.integrate
from scipy.ndimage import distance_transform_edt

from sourcefinder.gaussian import gaussian
from sourcefinder.utility import coordinates

from numba import njit, prange


def generate_subthresholds(min_value, max_value, num_thresholds):
    """
    Generate a series of ``num_thresholds`` logarithmically spaced values
    in the range (min_value, max_value) (both exclusive).
    """
    # First, we calculate a logarithmically spaced sequence between exp(0.0)
    # and (max - min + 1). That is, the total range is between 1 and one
    # greater than the difference between max and min.
    # We subtract 1 from this to get the range between 0 and (max-min).
    # We add min to that to get the range between min and max.
    subthrrange = np.logspace(
        0.0,
        np.log(max_value + 1 - min_value),
        num=num_thresholds + 1,  # first value == min_value
        base=np.e,
        endpoint=False  # do not include max_value
    )[1:]
    subthrrange += (min_value - 1)
    return subthrrange


def get_error_radius(wcs, x_value, x_error, y_value, y_error):
    """
    Estimate an absolute angular error on the position (x_value, y_value)
    with the given errors.

    This is a pessimistic estimate, because we take sum of the error
    along the X and Y axes. Better might be to project them both back on
    to the major/minor axes of the elliptical fit, but this should do for
    now.
    """
    error_radius = 0.0
    try:
        centre_ra, centre_dec = wcs.p2s([x_value, y_value])
        # We check all possible combinations in case we have a nonlinear
        # WCS.
        for pixpos in [
            (x_value + x_error, y_value + y_error),
            (x_value - x_error, y_value + y_error),
            (x_value + x_error, y_value - y_error),
            (x_value - x_error, y_value - y_error)
        ]:
            error_ra, error_dec = wcs.p2s(pixpos)
            error_radius = max(
                error_radius,
                coordinates.angsep(centre_ra, centre_dec, error_ra, error_dec)
            )
    except RuntimeError:
        # We get a runtime error from wcs.p2s if the errors place the
        # limits outside the image, in which case we set the angular
        # uncertainty to infinity.
        error_radius = float('inf')
    return error_radius


def circular_mask(xdim, ydim, radius):
    """
    Returns a numpy array of shape (xdim, ydim). All points with radius of
    the centre are set to 0; outside that region, they are set to 1.
    """
    centre_x, centre_y = (xdim - 1) / 2.0, (ydim - 1) / 2.0
    x, y = np.ogrid[-centre_x:xdim - centre_x, -centre_y:ydim - centre_y]
    return x * x + y * y >= radius * radius


def generate_result_maps(data, sourcelist):
    """Return a source and residual image

    Given a data array (image) and list of sources, return two images, one
    showing the sources themselves and the other the residual after the
    sources have been removed from the input data.
    """
    residual_map = np.array(data)  # array constructor copies by default
    gaussian_map = np.zeros(residual_map.shape)
    for src in sourcelist:
        # Include everything with 6 times the std deviation along the major
        # axis. Should be very very close to 100% of the flux.
        box_size = 6 * src.smaj.value / math.sqrt(2 * math.log(2))

        lower_bound_x = max(0, int(src.x.value - 1 - box_size))
        upper_bound_x = min(residual_map.shape[0],
                            int(src.x.value - 1 + box_size))
        lower_bound_y = max(0, int(src.y.value - 1 - box_size))
        upper_bound_y = min(residual_map.shape[1],
                            int(src.y.value - 1 + box_size))

        local_gaussian = gaussian(
            src.peak.value,
            src.x.value,
            src.y.value,
            src.smaj.value,
            src.smin.value,
            src.theta.value
        )(
            np.indices(residual_map.shape)[0, lower_bound_x:upper_bound_x,
                                              lower_bound_y:upper_bound_y],
            np.indices(residual_map.shape)[1, lower_bound_x:upper_bound_x,
                                              lower_bound_y:upper_bound_y]
        )

        gaussian_map[lower_bound_x:upper_bound_x,
                     lower_bound_y:upper_bound_y] += local_gaussian
        residual_map[lower_bound_x:upper_bound_x,
                     lower_bound_y:upper_bound_y] -= local_gaussian

    return gaussian_map, residual_map


def calculate_correlation_lengths(semimajor, semiminor):
    """Calculate the Condon correlation length

    In order to derive the error bars from Gauss fitting from the
    Condon (1997, PASP 109, 116C) formulae, one needs the so called
    correlation length. The Condon formulae assumes a circular area
    with diameter theta_N (in pixels) for the correlation. This was
    later generalized by Hopkins et al. (2003, AJ 125, 465) for
    correlation areas which are not axisymmetric.

    Basically one has theta_N^2 = theta_B*theta_b.

    Good estimates in general are:

    + theta_B = 2.0 * semimajar

    + theta_b = 2.0 * semiminor
    """

    return 2.0 * semimajor, 2.0 * semiminor


def calculate_beamsize(semimajor, semiminor):
    """Calculate the beamsize based on the semi major and minor axes"""

    return np.pi * semimajor * semiminor


def fudge_max_pix(semimajor, semiminor, theta):
    """Estimate peak flux correction at pixel of maximum flux

    Previously, we adopted Rengelink's correction for the
    underestimate of the peak of the Gaussian by the maximum pixel
    method: fudge_max_pix = 1.06. See the WENSS paper
    (1997A&AS..124..259R) or his thesis.  (The peak of the Gaussian
    is, of course, never at the exact center of the pixel, that's why
    the maximum pixel method will always underestimate it.)

    But, instead of just taking 1.06 one can make an estimate of the
    overall correction by assuming that the true peak is at a random
    position on the peak pixel and averaging over all possible
    corrections.  This overall correction makes use of the beamshape,
    so strictly speaking only accurate for unresolved sources.
    After some investigation, it turns out that this method requires not only
    unresolved sources, but also a circular beam shape. With the general
    elliptical beam, the peak source pixel may be located more than half a
    pixel away from the true peak. In that case the integral below will not
    suffice as a correction. Calculating an overall correction for an ensemble
    of sources will, in the case of an elliptical beam shape, become much
    more involved.
    """

    # scipy.integrate.dblquad: Computes a double integral
    # from the scipy docs:
    #   Return the double (definite) integral of f1(y,x) from x=a..b
    #   and y=f2(x)..f3(x).

    log20 = np.log(2.0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    def landscape(y, x):
        up = math.pow(((cos_theta * x + sin_theta * y) / semiminor), 2)
        down = math.pow(((cos_theta * y - sin_theta * x) / semimajor), 2)
        return np.exp(log20 * (up + down))

    (correction, abserr) = scipy.integrate.dblquad(landscape, -0.5, 0.5,
                                                   lambda ymin: -0.5,
                                                   lambda ymax: 0.5)

    return correction


def maximum_pixel_method_variance(semimajor, semiminor, theta):
    """Estimate variance for peak flux at pixel position of maximum

    When we use the maximum pixel method, with a correction
    fudge_max_pix, there should be no bias, unless the peaks of the
    Gaussians are not randomly distributed, but relatively close to
    the centres of the pixels due to selection effects from detection
    thresholds.

    Disregarding the latter effect and noise, we can compute the
    variance of the maximum pixel method by integrating (the true
    flux-the average true flux)^2 = (the true flux-fudge_max_pix)^2
    over the pixel area and dividing by the pixel area ( = 1).  This
    is just equal to integral of the true flux^2 over the pixel area
    - fudge_max_pix^2.
    """

    # scipy.integrate.dblquad: Computes a double integral
    # from the scipy docs:
    #   Return the double (definite) integral of f1(y,x) from x=a..b
    #   and y=f2(x)..f3(x).

    log20 = np.log(2.0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    def landscape(y, x):
        return np.exp(2.0 * log20 * (
                      math.pow(((cos_theta * x + sin_theta * y) / semiminor),
                                  2) +
                      math.pow(((cos_theta * y - sin_theta * x) / semimajor),
                                  2)))

    (result, abserr) = scipy.integrate.dblquad(landscape, -0.5, 0.5,
                                               lambda ymin: -0.5,
                                               lambda ymax: 0.5)
    variance = result - math.pow(fudge_max_pix(semimajor, semiminor, theta), 2)

    return variance


def flatten(nested_list):
    """Flatten a nested list

    Nested lists are made in the deblending algorithm. They're
    awful. This is a piece of code I grabbed from
    http://www.daniweb.com/code/snippet216879.html.

    The output from this method is a generator, so make sure to turn
    it into a list, like this::

        flattened = list(flatten(nested)).
    """
    for elem in nested_list:
        if isinstance(elem, (tuple, list, np.ndarray)):
            for i in flatten(elem):
                yield i
        else:
            yield elem


# “The nearest_nonzero function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
def nearest_nonzero(some_arr, rms):
    """
    Replace values in some_arr based on the nearest non-zero values in rms.

    Parameters
    ----------
    some_arr : np.ndarray
        A 2D array whose values will be replaced where rms == 0.
    rms : np.ndarray
        A 2D array of the same shape as some_arr. Nearest non-zero neighbors
        in rms determine the replacement indices for some_arr.

    Returns
    -------
    np.ndarray
        A copy of some_arr with values replaced based on nearest non-zero
        neighbors in rms.
    """
    if some_arr.shape != rms.shape:
        raise ValueError("some_arr and rms must have the same shape.")
    
    # Handle empty array.
    if some_arr.size == 0:
        return some_arr

    # Handle single-element array.
    if some_arr.size == 1:
        if rms[0, 0] == 0:
            # Return an empty array if the single value in rms is 0.
            # A rms of 0, means a standard deviation of zero, which means we
            # cannot do any form of thresholding for source detection. If we
            # return an empty background grid, ImageData._interpolate will
            # return a completely masked background map, such that no sources
            # will be detected.
            return np.empty((0, 0))
        else:
            return some_arr  # No replacement needed if rms[0, 0] is non-zero.

    # Create a mask for zero values in rms
    zero_mask = rms == 0

    # Calculate the distance transform and nearest non-zero indices
    distances, nearest_indices = distance_transform_edt(zero_mask,
                                                        return_indices=True)

    nearest_values = some_arr[nearest_indices[0], nearest_indices[1]]
    # Use nearest indices from rms to update some_arr
    result = some_arr.copy()
    result[zero_mask] = nearest_values[zero_mask]
    
    return result


# “The make_subimages function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit(parallel=True, cache=True)
def make_subimages(a_data, a_mask, back_size_x, back_size_y):
    """
    Reshape the image data such that it is suitable for guvectorized
    kappa * sigma clipping. The idea is that we have designed a function
    that will perform kappa * sigma clipping on a single flattened subimage,
    i.e. on a 1D ndarray. If we decorate that function with Numba's guvectorize
    decorator, we can use a 2D grid of subimages as input instead of a single
    flattened subimage. This means that the input to that guvectorized
    kappa * sigma clipper should be 3D. This function makes that 3D input,
    which is essentially a reshape using Numba with parallelization.

    Parameters:
    a_data (np.ndarray): The data of the masked array (without the mask).
    a_mask (np.ndarray): The mask of the masked array (True means the value
                         is masked).
    back_size_x (int): The size of the subimage along the row indices.
    back_size_y (int): The size of the subimages along the column indices.

    Returns:
    b (np.ndarray): 3D array where each subimage of size (back_size_x *
                    back_size_y) is flattened and padded with NaN.
    c (np.ndarray): 2D array where each element indicates the number of
                    unmasked values in the corresponding subimage.
    """
    subimage_size = back_size_x * back_size_y
    k, l = a_data.shape
    p = k // back_size_x
    r = l // back_size_y

    # Initialize b with NaNs and c with zeros
    b = np.full((p, r, subimage_size), np.nan, dtype=np.float32)
    c = np.zeros((p, r), dtype=np.int32)

    # Iterate over the subimages in parallel
    for i in prange(p):
        for j in range(r):
            # Extract the subimage (data and mask)
            subimage_data = a_data[i * back_size_x:(i + 1) * back_size_x,
                            j * back_size_y:(j + 1) * back_size_y]
            subimage_mask = a_mask[i * back_size_x:(i + 1) * back_size_x,
                            j * back_size_y:(j + 1) * back_size_y]

            # Preallocate an array for unmasked values (max size d*d)
            unmasked_values = np.empty(subimage_size, dtype=a_data.dtype)
            count = 0

            # Collect unmasked values manually
            for m in range(back_size_x):
                for n in range(back_size_y):
                    if not subimage_mask[m, n]:  # If not masked
                        unmasked_values[count] = subimage_data[m, n]
                        count += 1

            # Store the count of unmasked values in c
            c[i, j] = count

            # Pad the unmasked values with NaNs and store in b
            b[i, j, :count] = unmasked_values[:count]

    return b, c
