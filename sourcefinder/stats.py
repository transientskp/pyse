"""
Generic utility routines for number handling and calculating (specific)
variances used by the TKP sourcefinder.
"""

import math
import numpy as np
from numba import njit, guvectorize, int32, float32
from sourcefinder.utils import newton_raphson_root_finder
# CODE & NUMBER HANDLING ROUTINES

@njit
def erf(val):
    return math.erf(val)

@njit
def find_true_std(sigma, clipped_std, clip_limit):
    # Solves the transcendental equation 2.25 from Spreeuw's thesis.
    help1 = clip_limit / (sigma * np.sqrt(2))
    help2 = np.sqrt(2 * np.pi) * erf(help1)
    return (sigma ** 2 * (help2 - 2 * np.sqrt(2) * help1 *
            np.exp(-help1 ** 2)) - clipped_std ** 2 * help2)


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
