"""
This module contain utilities for the source finding routines
"""

import math
import numpy as np
from numbers import Real
import scipy.integrate
from scipy.ndimage import distance_transform_edt

from sourcefinder.gaussian import gaussian
from sourcefinder.utility import coordinates

from numba import njit, prange, guvectorize


def generate_subthresholds(min_value, max_value, num_thresholds):
    r"""Generate a series of ``num_thresholds`` logarithmically spaced values
    in the range (min_value, max_value) (both exclusive).

    First, we calculate a logarithmically spaced sequence between exp(0.0)
    and (max_value - min_value + 1). That is, the total range is between 1 and
    one greater than the difference between max_value and min_value.
    We subtract 1 from this to get the range between 0 and
    (max_value - min_value).
    We add min to that to get the range between min and max.

    This formula sets the subthreshold levels:

    .. math::

        t_i = \exp\left(
            \frac{i \cdot \log(\text{max\_value} + 1 - \text{min\_value})}
                 {\text{num\_thresholds} + 1}
        \right) + \text{min\_value} - 1

    for :math:`i = 1, 2, \ldots, \text{num\_thresholds}`.

    Parameters
    ----------
    min_value : float
        The minimum value of the range (not included in the output).
    max_value : float
        The maximum value of the range (not included in the output).
    num_thresholds : int
        The number of threshold values to generate.

    Returns
    -------
    np.ndarray
        An array of logarithmically spaced values between min_value and
        max_value (exclusive).

    """
    subthrrange = np.logspace(
        0.0,
        np.log(max_value + 1 - min_value),
        num=num_thresholds + 1,  # first value == min_value
        base=np.e,
        endpoint=False,  # do not include max_value
    )[1:]
    subthrrange += min_value - 1
    return subthrrange


def get_error_radius(wcs, x_value, x_error, y_value, y_error):
    """Estimate an absolute angular error on the position (x_value,
    y_value) with the given errors.

    Parameters
    ----------
    wcs : object
        The WCS (World Coordinate System) object used for transforming pixel
        coordinates to sky coordinates.
    x_value : float
        Position along first pixel coordinate (row index of ndarray with image
        data).
    x_error : float
        The 1-sigma error in x-value.
    y_value : float
        Position along second pixel coordinate (column index of ndarray with
        image data).
    y_error : float
        The 1-sigma error in y-value.

    Returns
    -------
    float
        The estimated absolute angular error in arcseconds.

    Notes
    -----
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
            (x_value - x_error, y_value - y_error),
        ]:
            error_ra, error_dec = wcs.p2s(pixpos)
            error_radius = max(
                error_radius,
                coordinates.angsep(centre_ra, centre_dec, error_ra, error_dec),
            )
    except RuntimeError:
        # We get a runtime error from wcs.p2s if the errors place the
        # limits outside the image, in which case we set the angular
        # uncertainty to infinity.
        error_radius = float("inf")
    return error_radius


def circular_mask(xdim, ydim, radius):
    """Returns a numpy array of shape (xdim, ydim). All points within
    radius from the centre are set to 0; outside that region, they are
    set to 1.

    Parameters
    ----------
    xdim : int
        The dimension of the array along the x-axis.
    ydim : int
        The dimension of the array along the y-axis.
    radius : float
        The radius from the center within which points are set to 0.

    Returns
    -------
    np.ndarray
        A 2D ndarray with points within the radius set to 0 and outside set to
        1.

    """
    centre_x, centre_y = (xdim - 1) / 2.0, (ydim - 1) / 2.0
    x, y = np.ogrid[-centre_x : xdim - centre_x, -centre_y : ydim - centre_y]
    return x * x + y * y >= radius * radius


def generate_result_maps(data, sourcelist):
    """Return an image with Gaussian reconstructions of the sources and the
    corresponding residual image.

    Given a data array (image) and list of sources, return two images, one
    showing the sources themselves and the other the residual after the
    sources have been removed from the input data.

    Parameters
    ----------
    data : np.ndarray
        The input data array (image).
    sourcelist : list
        A list of sources to be removed from the input data.

    Returns
    -------
    gaussian_map : np.ndarray (2D)
        Shows the Gaussian reconstructions of the sources.
    residual_map : np.ndarray (2D)
        Shows the residuals from the subtractions of these
        reconstructions from the image data.

    """
    residual_map = np.array(data)  # array constructor copies by default
    gaussian_map = np.zeros(residual_map.shape)
    for src in sourcelist:
        # Include everything with 6 times the std deviation along the major
        # axis. Should be very very close to 100% of the flux.
        box_size = 6 * src.smaj.value / math.sqrt(2 * math.log(2))

        lower_bound_x = max(0, int(src.x.value - 1 - box_size))
        upper_bound_x = min(
            residual_map.shape[0], int(src.x.value - 1 + box_size)
        )
        lower_bound_y = max(0, int(src.y.value - 1 - box_size))
        upper_bound_y = min(
            residual_map.shape[1], int(src.y.value - 1 + box_size)
        )

        local_gaussian = gaussian(
            src.peak.value,
            src.x.value,
            src.y.value,
            src.smaj.value,
            src.smin.value,
            src.theta.value,
        )(
            np.indices(residual_map.shape)[
                0, lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y
            ],
            np.indices(residual_map.shape)[
                1, lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y
            ],
        )

        gaussian_map[
            lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y
        ] += local_gaussian
        residual_map[
            lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y
        ] -= local_gaussian

    return gaussian_map, residual_map


def is_valid_beam_tuple(b) -> bool:
    return (
        isinstance(b, tuple)
        and len(b) == 3
        and all(isinstance(x, Real) and x is not None for x in b)
    )


def calculate_correlation_lengths(semimajor, semiminor):
    """Calculate the Condon correlation lengths.

    In order to derive the error bars for Gaussian fits from the
    Condon (1997, PASP 109, 116C) formulae, one needs a quantity called the
    correlation length. The Condon formulae assume a circular area
    with diameter theta_N (in pixels) for the correlation: i.e. all noise
    within that area is assumed completely correlated while outside that area
    all noise is assumed completely uncorrelated. This was later generalized
    by Hopkins et al. (2003, AJ 125, 465, paragraph 3) for correlation areas
    which are not circular, i.e. for anisotropic restoring beams.

    Basically one has theta_N^2 = theta_B*theta_b.

    Good estimates in general are:

    + theta_B = 2.0 * semimajor

    + theta_b = 2.0 * semiminor

    but we have included this function to provide a convenient way of altering
    these dependencies.

    Parameters
    ----------
    semimajor : float
        The semi-major axis length in pixels.
    semiminor : float
        The semi-minor axis length in pixels.

    Returns
    -------
    tuple[float,float]
        A tuple containing the correlation lengths (theta_B, theta_b), in
        pixels.

    """

    return 2.0 * semimajor, 2.0 * semiminor


def calculate_beamsize(semimajor, semiminor):
    """Calculate the beamsize based on the semi-major and minor axes.

    Parameters
    ----------
    semimajor : float
        The semi-major axis length in pixels.
    semiminor : float
        The semi-minor axis length in pixels.

    Returns
    -------
    float
        The calculated beamsize.

    """
    return np.pi * semimajor * semiminor


def fudge_max_pix(semimajor, semiminor, theta):
    """Estimate peak spectral brightness correction at pixel of maximum spectral
    brightness.

    Previously, we adopted Rengelink's correction for the
    underestimate of the peak of the Gaussian by the maximum pixel
    method: fudge_max_pix = 1.06. See the WENSS paper
    (1997A&AS..124..259R) or his thesis.  The peak of the Gaussian
    is, of course, never at the exact center of the pixel, that's why
    the maximum pixel method will underestimate it, when averaged over an
    ensemble. This effect is smaller when the clean beam is more densely
    sampled.

    But, instead of just taking 1.06 one can make an estimate of the
    overall correction by assuming that the true peak is at a random
    position on the peak pixel and averaging over all possible
    corrections.  This overall correction makes use of the beamshape,
    so strictly speaking only accurate for unresolved sources.
    After some investigation, it turns out that this method requires not only
    unresolved sources, but also a circular beam shape. With the general
    elliptical beam, the peak pixel may be located more than half a
    pixel away from the true peak. In that case the integral below will not
    suffice as a correction. Calculating an overall correction for an ensemble
    of sources will, in the case of an elliptical beam shape, become much
    more involved.

    Parameters
    ----------
    semimajor : float
        The semi-major axis length in pixels.
    semiminor : float
        The semi-minor axis length in pixels.
    theta : float
        The position angle of the major axis in radians.

    Returns
    -------
    correction : float
        The estimated peak spectral brightness correction.

    """

    log20 = np.log(2.0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    def landscape(y, x):
        up = math.pow(((cos_theta * x + sin_theta * y) / semiminor), 2)
        down = math.pow(((cos_theta * y - sin_theta * x) / semimajor), 2)
        return np.exp(log20 * (up + down))

    (correction, abserr) = scipy.integrate.dblquad(
        landscape, -0.5, 0.5, lambda ymin: -0.5, lambda ymax: 0.5
    )

    return correction


def flatten(nested_list):
    """Flatten a nested list

    Nested lists are made in the deblending algorithm. They're
    awful. This is a piece of code I grabbed from
    http://www.daniweb.com/code/snippet216879.html.

    The output from this method is a generator, so make sure to turn
    it into a list, like this::

        flattened = list(flatten(nested)).

    Parameters
    ----------
    nested_list : list
        A list that may contain other lists or tuples.

    Yields
    ------
    element
        The next element in the flattened list.

    Notes
    -----
        The keyword "yield" is used; i.e. a generator object is returned.

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
    """Replace values in some_arr based on the nearest non-zero
    values in rms.

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
    distances, nearest_indices = distance_transform_edt(
        zero_mask, return_indices=True
    )

    nearest_values = some_arr[nearest_indices[0], nearest_indices[1]]
    # Use nearest indices from rms to update some_arr
    result = some_arr.copy()
    result[zero_mask] = nearest_values[zero_mask]

    return result


# “The make_subimages function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit(parallel=True)
def make_subimages(a_data, a_mask, back_size_x, back_size_y):
    """Make subimages.

    Reshape the image data such that it is suitable for guvectorized
    kappa * sigma clipping. The idea is that we have designed a function
    that will perform kappa * sigma clipping on a single flattened subimage,
    i.e. on a 1D ndarray. If we decorate that function with Numba's guvectorize
    decorator, we can use a 2D grid of subimages as input instead of a single
    flattened subimage. This means that the input to that guvectorized
    kappa * sigma clipper should be 3D. This function makes that 3D input,
    which is essentially a reshape using Numba with parallelization.

    Parameters
    ----------
    a_data: np.ndarray
        The data of the masked array (without the mask).
    a_mask: np.ndarray
        The mask of the masked array (True means the value is masked).
    back_size_x: int
        The size of the subimage along the row indices.
    back_size_y: int
        The size of the subimages along the column indices.

    Returns
    -------
    b: np.ndarray
        3D array where each subimage of size (back_size_x *
        back_size_y) is flattened and padded with NaN.
    c: np.ndarray
        2D array where each element indicates the number of unmasked
        values in the corresponding subimage.

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
            subimage_data = a_data[
                i * back_size_x : (i + 1) * back_size_x,
                j * back_size_y : (j + 1) * back_size_y,
            ]
            subimage_mask = a_mask[
                i * back_size_x : (i + 1) * back_size_x,
                j * back_size_y : (j + 1) * back_size_y,
            ]

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


# “The interp_per_row function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
# Define the row-wise interpolation function with guvectorize.
@guvectorize(
    ["void(float32[:], float32[:], float32[:], float32[:])"],
    "(n),(n),(k)->(k)",
    target="parallel",
    nopython=True,
    cache=True,
)
def interp_per_row(grid_row, y_initial, y_sought, interp_row):
    """Interpolate one row of the grid along the second dimension
    (y-axis).

    Parameters
    ----------
    grid_row
        1D array representing a single row of the grid.
    y_initial
        Original grid coordinates along the y-axis.
    y_sought
        Target coordinates for interpolation along the y-axis.
    interp_row
        Output array to store the interpolated row.

    """
    interp_row[:] = np.interp(y_sought, y_initial, grid_row)


# “The two_step_interp function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
def two_step_interp(grid, new_xdim, new_ydim):
    """Perform two-step interpolation.

    It is done on a grid to upsample it to new dimensions.  This
    function proivdes fast pieceswise bilinear interpolation in two
    steps:

    1. Interpolation across columns, i.e. per row of the input
       grid. Each row of the input grid is handled independently.
    2. Transpose the result.
    3. Again, interpolate across columns.

    This method was inspired by a comment from "tiago" in this SO
    discussion: https://stackoverflow.com/q/14530556

    Parameters
    ----------
    grid : numpy.ndarray
        The input 2D array to be upsampled.
    new_xdim : int
        The desired number of rows in the upsampled grid.
    new_ydim : int
        The desired number of columns in the upsampled grid.

    Returns
    -------
    numpy.ndarray
        The upsampled grid with dimensions (new_xdim, new_ydim).

    """
    # Define the main function for upsampling
    # Original grid coordinates
    x_initial = np.linspace(
        0, grid.shape[0] - 1, grid.shape[0], dtype=np.float32
    )
    y_initial = np.linspace(
        0, grid.shape[1] - 1, grid.shape[1], dtype=np.float32
    )

    # Target grid coordinates
    x_sought = np.linspace(
        -0.5, grid.shape[0] - 0.5, new_xdim, dtype=np.float32
    )
    y_sought = np.linspace(
        -0.5, grid.shape[1] - 0.5, new_ydim, dtype=np.float32
    )

    # Step 1: Interpolation per row.
    interp_rows = np.empty((grid.shape[0], new_ydim), dtype=np.float32)
    interp_per_row(grid, y_initial, y_sought, interp_rows)

    # Step 2: Interpolation along columns (reuse interpolate_rows)
    interp_cols = np.empty((new_xdim, new_ydim), dtype=np.float32)
    interp_per_row(interp_rows.T, x_initial, x_sought, interp_cols.T)

    return interp_cols


# “This 'newton_raphson_root_finder' function has been generated using
# ChatGPT 4.0. Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit
def newton_raphson_root_finder(
    f, sigma0, min_sigma, max_sigma, tol=1e-8, max_iter=100, *args
):
    """Solve the transcendental equation for sigma using Newton's
    method with interval safeguards.

    Parameters
    ----------
    f : function
        The function to find the root of. It should take three parameters:
        sigma, sigma_meas, and D.
    sigma0 : float
        Initial guess for the value of sigma.
    min_sigma : float
        Minimum bound for sigma.
    max_sigma : float
        Maximum bound for sigma.
    tol : float, default: 1e-8
        The tolerance for convergence.
    max_iter : int, default: 100
        The maximum number of iterations.
    *args : tuple
        Additional arguments for the function `f`.

    Returns
    -------
    sigma : float
        The value of sigma that solves the equation, constrained within the
        bounds [min_sigma, max_sigma].
    i : int
        The number of iterations performed. If convergence was reached, this
        is the iteration count. If max iterations were reached, this is equal
        to max_iter.

    Raises
    ------
    ValueError
        If the derivative of the function is near zero, causing a division by
        zero error.

    Notes
    -----
    The method employs Newton's method for root-finding, with safeguards to
    ensure that sigma stays within the bounds [min_sigma, max_sigma]. The
    method terminates when the absolute change in sigma is smaller than the
    specified tolerance `tol` or when the maximum number of iterations is
    reached. If the derivative is too small (near zero), a ValueError is raised.

    """

    sigma = sigma0
    for i in range(max_iter):
        # Evaulate function at current sigma
        f_val = f(sigma, *args)

        # Compute numerical derivative
        delta = tol * sigma
        f_deriv = (f(sigma + delta, *args) - f_val) / delta

        if np.abs(f_deriv) < tol:  # avoid division by zero
            raise ValueError("Derivative near zero, method fails.")

        # Update sigma using Newton-Raphson method.
        delta_sigma = -f_val / f_deriv
        sigma += delta_sigma

        # Apply safeguard to keep sigma within the interval
        # [min_sigma, max_sigma]
        if sigma < min_sigma:
            sigma = min_sigma
        elif sigma > max_sigma:
            sigma = max_sigma

        if np.abs(delta_sigma) < tol:
            return sigma, i  # root found, return solution and iterations

    return sigma, max_iter  # max iterations reached, return last estimate


def complement_gaussian_args(initial_params, fixed_params, fit_params):
    """Complements initial parameters for Gaussian fitting, with
    fixed values.  The end result should be a list of six elements,
    corresponding to 'peak', 'xbar', 'ybar', 'semimajor', 'semiminor'
    and 'theta', in that order

    Parameters
    ----------
    initial_params : np.ndarray
        A Numpy float array containing the initial values for at least one, but
        a maximum of six Gaussian parameters: 'peak', 'xbar', 'ybar',
        'semimajor', 'semiminor' and 'theta' , in that order.
    fixed_params : dict
        A dictionary where the keys are zero, one or more of these parameter
        names: 'peak', 'xbar', 'ybar', 'semimajor', 'semiminor' and 'theta'.
        The values are the fixed values for those parameters. It must complement
        initial_parms, such that the total number of parameters is six:
        len(initial_params) + len(fixed_params) == 6.
    fit_params : tuple
        A tuple of the six Gaussian parameters to be fitted. If a parameter is
        found in `fixed_params`, its value is used from there; otherwise, the
        value is taken from `initial_params`.
        Almost always this should be ('peak', 'xbar', 'ybar', 'semimajor',
        'semiminor', 'theta').
        len(fit_params) == 6.

    Returns
    -------
    gaussian_args : list[float]
        A list of Gaussian fitting arguments, where each value corresponds to
        either a fixed value or an initial value from `initial_params`.
        len(gaussian_args) == 6.

    Example
    -------
    >>> initial_parms = np.array([1.0, 4.0, 5.0, 6.0])
    >>> fixed_parms = {'center_x': 2.5, 'center_y': 3.5}
    >>> fit_parms = ('peak', 'center_x', 'center_y', 'semi-major axis',
    ...               'semi-minor axis', 'position angle')
    >>> complement_gaussian_args(initial_parms, fixed_parms, fit_parms)
    [1.0, 2.5, 3.5, 4.0, 5.0, 6.0]

    """
    paramlist = list(initial_params)
    gaussian_args = []
    for parameter in fit_params:
        if parameter in fixed_params:
            gaussian_args.append(fixed_params[parameter])
        else:
            gaussian_args.append(paramlist.pop(0))

    return gaussian_args
