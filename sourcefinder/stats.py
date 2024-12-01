"""
Generic utility routines for number handling and calculating (specific)
variances used by the TKP sourcefinder.
"""

import numpy as np
from numpy.ma import MaskedArray
from scipy.special import erf
import numba_scipy
from scipy.optimize import fsolve
from numba import njit, guvectorize, int32, float32
# CODE & NUMBER HANDLING ROUTINES


def find_true_std(sigma, clip_limit, clipped_std):
    # Solves the transcendental equation 2.25 from Spreeuw's thesis.
    help1 = clip_limit / (sigma * np.sqrt(2))
    help2 = np.sqrt(2 * np.pi) * erf(help1)
    return (sigma ** 2 * (help2 - 2 * np.sqrt(2) * help1 *
            np.exp(-help1 ** 2)) - clipped_std ** 2 * help2)


@njit
def f_sigma(sigma, sigma_meas, D):
    """
    Transcendental equation to be solved for sigma.
    """
    term1 = np.sqrt(2 * np.pi) * erf(D / (sigma * np.sqrt(2)))
    term2 = term1 - 2 * (D / sigma) * np.exp(-D ** 2 / (2 * sigma ** 2))

    return sigma_meas ** 2 * (term1 / term2) - sigma ** 2


# “This 'derivative' function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit
def derivative(f, sigma, sigma_meas, D, eps=1e-8):
    """
    Approximate the derivative of f at sigma using finite differences.
    """
    return (f(sigma + eps, sigma_meas, D) - f(sigma, sigma_meas, D)) / eps


# “This 'newton_1d_safeguard_sigma' function has been generated using
# ChatGPT 4.0. Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit
def newton_1d_safeguard_sigma(f, sigma0, sigma_meas, D, min_sigma, max_sigma,
                              tol=1e-8, max_iter=100):
    """
    Solve the transcendental equation for sigma using Newton's method with
    interval safeguards.
    f: function to find the root of.
    sigma0: initial guess for sigma.
    sigma_meas: measured sigma value.
    D: parameter D in the equation.
    min_sigma: minimum bound for sigma.
    max_sigma: maximum bound for sigma.
    tol: tolerance for convergence.
    max_iter: maximum number of iterations.
    """
    sigma = sigma0
    for i in range(max_iter):
        f_val = f(sigma, sigma_meas, D)
        if np.abs(f_val) < tol:
            return sigma, i  # root found, return solution and iterations

        df_val = derivative(f, sigma, sigma_meas, D)
        if np.abs(df_val) < tol:  # avoid division by zero
            raise ValueError("Derivative near zero, method fails.")

        delta_sigma = -f_val / df_val
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


@njit
def indep_pixels(n, correlation_lengths):
    corlengthlong, corlengthshort = correlation_lengths
    correlated_area = 0.25 * np.pi * corlengthlong * corlengthshort
    return n / correlated_area


def sigma_clip(data, kappa=2.0, max_iter=100,
               centref=np.median, distf=np.var, my_iterations=0, limit=None):
    """Iterative clipping

    By default, this performs clipping of the standard deviation about the
    median of the data. But by tweaking centref/distf, it could be much
    more general.

    max_iter sets the maximum number of iterations used.

    my_iterations is a counter for recursive operation of the code; leave it
    alone unless you really want to pretend to jump into the middle of a loop.

    sigma is subtle: if a callable is given, it is passed a copy of the data
    array and can calculate a clipping limit. See, for e.g., unbiased_sigma()
    defined above. However, if it isn't callable, sigma is assumed to just set
    a hard limit.

    To do: Improve documentation
            -Returns???
            -How does it make use of the beam? (It estimates the noise correlation)
    """
    if my_iterations >= max_iter:
        # Exceeded maximum number of iterations; return
        return data, my_iterations

    # Numpy 1.1 breaks std() for MaskedArray: see
    # <http://www.scipy.org/scipy/numpy/wiki/MaskedArray>.
    # MaskedArray.compressed() returns a 1-D array of non-masked data.
    if isinstance(data, MaskedArray):
        data = data.compressed()
    centre = centref(data)
    n = np.size(data)
    if n < 1:
        # This chunk is too small for processing; return an empty array.
        return np.array([]), 0, 0, 0

    clipped_var = distf(data)

    std = np.sqrt(clipped_var)

    if limit is not None:
        std_corr_for_clipping_bias = fsolve(find_true_std, std,
                                            args=(limit, std))[0]
    else:
        std_corr_for_clipping_bias = std

    limit = kappa * std_corr_for_clipping_bias

    newdata = data.compress(abs(data - centre) <= limit)

    if len(newdata) != len(data) and len(newdata) > 0:
        my_iterations += 1
        return sigma_clip(newdata, kappa, max_iter, centref, distf,
                          my_iterations, limit=limit)
    else:
        return newdata, std_corr_for_clipping_bias, centre, my_iterations


@guvectorize([(float32[:], int32[:], float32[:], float32[:])],
             '(k), () -> (), ()', target="parallel")
def data_clipper_dynamic(flat_data, number_of_non_nan_elements, mean, std):
    """"""
    # In this context we use the terms nan and masked interchangeably, since we
    # expect every image value that is a nan to be masked.
    # The first number_of_non_nan_elements of flat_data will be not nans.
    # We need at least two non-nan elements to compute a standard deviation.
    # However, here we apply a stricter policy and require all subimage data to
    # be unmasked. Without this requirement, we cannot guarantee that the
    # subimage data used to calculate a grid node value is centered on that
    # node.
    if number_of_non_nan_elements[0] == flat_data.size:
        limit = 0
        while True:
            # When there are sources, the median is a more robust
            # approximation of the mean of the background pixels than a plain
            # mean. Think of this approach as a mode approximation, since
            # background pixels should outnumber source pixels.
            mean[0] = np.median(flat_data)
            regular_std = np.std(flat_data)

            if limit:
                # The standard deviation of clipped data will be biased low,
                # correct for that.
                std[0], iterations = (
                    newton_1d_safeguard_sigma(f_sigma, regular_std, regular_std,
                                              limit, 0, limit))
            else:
                std[0] = regular_std

            limit = 2 * std[0]
            clipped_data = flat_data[flat_data < mean[0] + limit]
            clipped_data = clipped_data[clipped_data > mean[0] - limit]
            if clipped_data.size == flat_data.size:
                break
            flat_data = clipped_data

        plain_mean = np.mean(flat_data)
        # If the distribution of remaining pixel values is not too skewed, this
        # gives a better approximation of the mean of the background pixels.
        if np.fabs(plain_mean - mean[0]) / std[0] < 0.3:
            mean[0] = 2.5 * mean[0] - 1.5 * plain_mean
    else:
        mean[0] = 0
        std[0] = 0
