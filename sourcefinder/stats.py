"""
Generic utility routines for number handling and calculating (specific)
variances used by the TKP sourcefinder.
"""

import numpy
from numpy.ma import MaskedArray
from scipy.special import erf
from scipy.special import erfcinv
from scipy.optimize import fsolve

from .utils import calculate_correlation_lengths


# CODE & NUMBER HANDLING ROUTINES
#
def var_helper(N):
    """Correct for the fact the rms noise is computed from a clipped
    distribution.

    That noise will always be lower than the noise from the complete
    distribution.  The correction factor is a function of the computed
    rms noise only.
    """
    term1 = numpy.sqrt(2. * numpy.pi) * erf(N / numpy.sqrt(2.))
    term2 = 2. * N * numpy.exp(-N ** 2 / 2.)
    return term1 / (term1 - term2)

def find_true_std(sigma, clip_limit, clipped_std):
    help1=clip_limit/(sigma*np.sqrt(2))
    help2=np.sqrt(2*np.pi)*erf(help1)
    return sigma**2*(help2-2*np.sqrt(2)*help1*np.exp(-help1**2))-clipped_std**2*help2

def indep_pixels(N, beam):
    corlengthlong, corlengthshort = calculate_correlation_lengths(
        beam[0], beam[1])
    correlated_area = 0.25 * numpy.pi * corlengthlong * corlengthshort
    return N / correlated_area

def sigma_clip(data, beam, kappa=1.5, max_iter=100,
               centref=numpy.median, distf=numpy.var, my_iterations=0,
               corr_clip=True):
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
    N = numpy.size(data)
    N_indep = indep_pixels(N, beam)
    if N_indep < 1:
        # This chunk is too small for processing; return an empty array.
        return numpy.array([]), 0, 0, 0

    # distf=numpy.var is a sample variance with the factor N/(N-1)
    # already built in, N being the number of pixels. So, we are
    # going to remove that and replace it by N_indep/(N_indep-1)
    clipped_var = distf(data) * (N - 1.) * N_indep / (N * (N_indep - 1.))
    unbiased_var = corr_clip * clipped_var

    # There is an extra factor c4 needed to get a unbiased standard
    # deviation, unbiased if we disregard clipping bias, see
    # http://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation\
    #         #Results_for_the_normal_distribution
    c4 = 1. - 0.25 / N_indep - 0.21875 / N_indep ** 2
    unbiased_std = numpy.sqrt(unbiased_var) / c4

    limit = kappa * unbiased_std

    newdata = data.compress(abs(data - centre) <= limit)

    if len(newdata) != len(data) and len(newdata) > 0:
        corr_clip = var_helper(my_sigma)
        my_iterations += 1
        return sigma_clip(newdata, beam, kappa, max_iter, centref, distf,
                          my_iterations, corr_clip=True)
    else:
        if corr_clip==True:
            unbiased_std=fsolve(find_true_std, unbiased_std, args=(limit,unbiased_std))[0]
        return newdata, unbiased_std, centre, my_iterations
