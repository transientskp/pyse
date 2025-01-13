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


@njit
def find_true_std(sigma, clipped_std, clip_limit):
    # Solves the transcendental equation 2.25 from Spreeuw's thesis.
    help1 = clip_limit / (sigma * np.sqrt(2))
    help2 = np.sqrt(2 * np.pi) * erf(help1)
    return (sigma ** 2 * (help2 - 2 * np.sqrt(2) * help1 *
            np.exp(-help1 ** 2)) - clipped_std ** 2 * help2)


# “This 'derivative' function has been generated using ChatGPT 4.0.
# Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
# @njit
# def derivative(f, sigma, sigma_meas, D, eps=1e-8):
#     """
#     Approximate the derivative of f at sigma using finite differences.
#     """
#     delta = eps * sigma
#     return (f(sigma + delta, sigma_meas, D) - f(sigma, sigma_meas, D)) / delta


# “This derivative' function has been generated using
# ChatGPT 4.0. Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit
def derivative(f, x, eps=1e-8, *args):
    """
    Approximate the derivative of f at sigma using finite differences.

    Parameters
    ----------
    f : function
        The function for which the derivative is to be approximated.
    x : float
        The value of the first argument of f at which the derivative is to be
        approximated.
    eps : float, default: 1e-8
        A small value for finite difference approximation.
    *args : tuple
        Additional arguments for the function `f`.

    Returns
    -------
    float
        The approximate derivative of f at x.

    """
    delta = eps * x
    return (f(x + delta, *args) - f(x, *args)) / delta


# “This 'newton_1d_safeguard_sigma' function has been generated using
# ChatGPT 4.0. Its AI-output has been verified for correctness, accuracy and
# completeness, adapted where needed, and approved by the author.”
@njit
def newton_raphson_root_finder(f, sigma0, min_sigma, max_sigma,
                               tol=1e-8, max_iter=100, *args):
    """
    Solve the transcendental equation for sigma using Newton's method with
    interval safeguards.

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
        # Compute function value
        f_val = f(sigma, *args)

        # Compute numerical derivative
        f_deriv = derivative(f, sigma, *args, eps=tol)

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


@njit
def indep_pixels(n, correlation_lengths):
    corlengthlong, corlengthshort = correlation_lengths
    correlated_area = 0.25 * np.pi * corlengthlong * corlengthshort
    return n / correlated_area


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
                    newton_raphson_root_finder(find_true_std, regular_std,
                                               0, limit,1e-8,100, regular_std,
                                               limit))
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
