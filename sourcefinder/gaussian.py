"""
Definition of a two dimensional elliptical Gaussian.
"""

from numpy import exp, log, cos, sin


def gaussian(height, center_x, center_y, semimajor, semiminor, theta):
    """Return a 2D Gaussian function with the given parameters.

    Args:

        height (float): (z-)value of the 2D Gaussian

        center_x (float): x center of the Gaussian

        center_y (float): y center of the Gaussian

        semimajor (float): major axis of the Gaussian

        semiminor (float): minor axis of the Gaussian

        theta (float): angle of the 2D Gaussian in radians, measured
            between the semi-major and y axes, in counterclockwise
            direction.

    Returns:
        lambda: 2D Gaussian (function of pixel coords ``(x,y)``)
    """

    return lambda x, y: height * exp(
        -log(2.0) * (((cos(theta) * (x - center_x) +
                       sin(theta) * (y - center_y)) /
                      semiminor) ** 2.0 +
                     ((cos(theta) * (y - center_y) -
                       sin(theta) * (x - center_x)) /
                      semimajor) ** 2.))


def jac_gaussian(gaussianargs):
    """Return the Jacobian of a 2D anisotropic Gaussian

    Args:
        gaussianargs. This is a list containing the usual six
        Gaussian parameters:

            height (float): (z-)value of the 2D Gaussian

            center_x (float): x center of the Gaussian

            center_y (float): y center of the Gaussian

            semimajor (float): major axis of the Gaussian

            semiminor (float): minor axis of the Gaussian

            theta (float): angle of the 2D Gaussian in radians, measured
                between the semi-major and y axes, in counterclockwise
                direction.

    Returns:
        function: Jacobian of Gaussian, i.e. the derivatives along each of these
                six parameters of the Gaussian, as a function of pixel
                coordinates x and y, so six functions are returned.

    """
    height, center_x, center_y, semimajor, semiminor, theta = gaussianargs

    # First define some auxiliary quantities, which will return in many of the
    # partial derivatives.
    def a(x):
        return center_x - x

    b = semimajor

    def c(y):
        return center_y - y

    d = semiminor
    e = cos(theta)
    f = sin(theta)
    g = log(2)

    def term3(x, y):
        return e * a(x) + f * c(y)

    def term4(x, y):
        return e * c(y) - f * a(x)

    def term1(x, y):
        return term3(x, y) / d

    def term2(x, y):
        return term4(x, y) / b

    def arg(x, y):
        return g * (term1(x, y) ** 2 + term2(x, y) ** 2)

    def expon(x, y):
        return exp(-arg(x, y))

    def common(x, y):
        return 2 * g * height * expon(x, y)

    # Partial derivative of the Gausian along height.
    def dg_dh(x, y): return expon(x, y)

    # Along center_x
    def dg_dx0(x, y):
        return common(x, y) * (-term1(x, y) * e / d + term2(x, y) * f / b)

    # Along center_y
    def dg_dy0(x, y):
        return common(x, y) * (-term1(x, y) * f / d - term2(x, y) * e / b)

    # Along the semi-major axis
    def dg_dsmaj(x, y):
        return common(x, y) * term2(x, y) ** 2 / b

    # Along the semi-minor axus
    def dg_dsmin(x, y):
        return common(x, y) * term1(x, y) ** 2 / d

    # Along the position angle
    def dg_dtheta(x, y):
        return common(x, y) * term3(x, y) * term4(x, y) * (1 / b ** 2
                                                           - 1 / d ** 2)

    jacobian = {"peak": dg_dh,
                "xbar": dg_dx0,
                "ybar": dg_dy0,
                "semimajor": dg_dsmaj,
                "semiminor": dg_dsmin,
                "theta": dg_dtheta}

    return jacobian
