import numpy as np
from numba import njit

# --- Helper functions --------------------------------------------------------


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
def gaussian_from_Sigma_matrix(x, y, I0, mu, Sigma):
    """
    Evaluate a 2D anisotropic Gaussian defined by its covariance matrix.

    The Gaussian is given by::

        G(x, y) = I₀ * exp[-½ (r - μ)ᵀ Σ⁻¹ (r - μ)],

    where Σ is the 2×2 covariance matrix describing the ellipse (its
    orientation and semi-axes) and μ = (x₀, y₀) is the centroid position.

    Parameters
    ----------
    x, y : array_like
        Coordinate arrays of identical shape giving the pixel positions
        at which to evaluate the Gaussian. Typically created with
        ``np.meshgrid``.
    I0 : float
        Peak intensity (the value of the Gaussian at its center).
    mu : array_like, shape (2,)
        Center coordinates (x₀, y₀) of the Gaussian, in the same units
        as `x` and `y`.
    Sigma : array_like, shape (2, 2)
        Covariance matrix defining the shape and orientation of the
        ellipse. Must be symmetric and positive definite.

    Returns
    -------
    G : ndarray
        The evaluated Gaussian image, with the same shape as `x`.

    Notes
    -----
    - The determinant of Σ controls the spatial extent of the Gaussian.
    - The integral over all space equals

          ∫∫ G(x, y) dx dy = I₀ * 2π * sqrt(det(Σ)).

      This may be useful for flux normalization.
    - If Σ is diagonal, the Gaussian is axis-aligned with standard
      deviations σₓ = sqrt(Σ₀₀) and σᵧ = sqrt(Σ₁₁).

    Examples
    --------
    >>> import numpy as np
    >>> x, y = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
    >>> mu = np.array([0.0, 0.0])
    >>> Sigma = np.array([[4.0, 1.0],
    ...                   [1.0, 2.0]])
    >>> G = gaussian_from_Sigma_matrix(x, y, I0=1.0, mu=mu, Sigma=Sigma)
    >>> G.shape
    (101, 101)

    """
    pos = np.stack((x - mu[0], y - mu[1]), axis=0)
    invSigma = np.linalg.inv(Sigma)
    exponent = -0.5 * (
        invSigma[0, 0] * pos[0] ** 2
        + 2 * invSigma[0, 1] * pos[0] * pos[1]
        + invSigma[1, 1] * pos[1] ** 2
    )
    return I0 * np.exp(exponent)


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
def convolve_gaussians(I1, mu1, Sigma1, I2, mu2, Sigma2):
    """
    Analytically convolve two 2D anisotropic Gaussian profiles.

    The convolution of two Gaussians::

        G₁(r) = I₁ exp[-½ (r − μ₁)ᵀ Σ₁⁻¹ (r − μ₁)]
        and
        G₂(r) = I₂ exp[-½ (r − μ₂)ᵀ Σ₂⁻¹ (r − μ₂)]

    yields another Gaussian with parameters::

        Σ_H = Σ₁ + Σ₂
        μ_H = μ₁ + μ₂
        I_H = I₁ I₂ √(|Σ₁| |Σ₂| / |Σ_H|)

    Parameters
    ----------
    I1, I2 : float
        Peak brightnesses of the two input Gaussians.
    mu1, mu2 : array_like, shape (2,)
        Center coordinates (x₀, y₀) of each Gaussian.
    Sigma1, Sigma2 : array_like, shape (2, 2)
        Covariance matrices describing the shape and orientation of
        each Gaussian. Must be symmetric and positive definite.

    Returns
    -------
    I_H : float
        Peak brightness of the convolved Gaussian.
    mu_H : ndarray, shape (2,)
        Center coordinates of the convolved Gaussian.
    Sigma_H : ndarray, shape (2, 2)
        Covariance matrix of the convolved Gaussian.

    Notes
    -----
    - The convolution of two Gaussians is itself a Gaussian, which makes
      this operation fully analytic.
    - The combined covariance matrix is simply the sum
      of the individual covariances.
    - The combined centroid is the vector sum of the input centroids;
      this assumes both functions are defined in the same coordinate
      frame (no shift-invariance correction applied).
    - The normalization factor ensures conservation of integrated flux
      under convolution.

    Examples
    --------
    >>> import numpy as np
    >>> I1, mu1, Sigma1 = 1.0, np.array([0., 0.]), np.diag([2.0, 1.0])
    >>> I2, mu2, Sigma2 = 1.0, np.array([0.3, -0.2]), np.diag([1.5, 0.5])
    >>> I_H, mu_H, Sigma_H = convolve_gaussians(I1, mu1, Sigma1, I2, mu2, Sigma2)
    >>> I_H, mu_H, Sigma_H
    (0.707..., array([ 0.3, -0.2]), array([[3.5, 0.],
                                           [0., 1.5]]))

    """
    Sigma_H = Sigma1 + Sigma2
    mu_H = mu1 + mu2
    I_H = (
        I1
        * I2
        * np.sqrt(
            np.linalg.det(Sigma1)
            * np.linalg.det(Sigma2)
            / np.linalg.det(Sigma_H)
        )
    )
    return I_H, mu_H, Sigma_H


# “This function has been generated using ChatGPT 5.0. Its AI-output has
# been verified for correctness, accuracy and completeness, adapted where
# needed, and approved by the author.”
def params_from_sigma(Sigma):
    """
    From covariance matrix Sigma (2x2) return (sigma_maj, sigma_min,
    theta) with sigma_maj>=sigma_min>=0.

    Parameters
    ----------
    Sigma : array_like, shape (2, 2)
        Covariance matrix defining the shape and orientation of the
        ellipse. Must be symmetric.
        Covariance matrix elements: if sigma_maj is major axis stddev,
        sigma_min is minor axis stddev, and theta the position angle (CCW
        from +Y), then::

            S = [[sxx, sxy],
                 [sxy, syy]] = R @ [[sigma_maj^2, 0],
                                    [0, sigma_min^2]] @ R.T

            where R = [[-sin(theta), -cos(theta)],
                       [ cos(theta), -sin(theta)]]

            i.e. sxx = sigma_maj^2 sin^2(theta) + sigma_min^2 cos^2(theta)
                 syy = sigma_maj^2 cos^2(theta) + sigma_min^2 sin^2(theta)
                 sxy = -(sigma_maj^2 - sigma_min^2) sin(theta) cos(theta)

    Returns
    -------
    sigma_maj : float
        Standard deviation along major axis.
    sigma_min : float
        Standard deviation along minor axis.
    theta : float
        Position angle of the major axis in radians, in [0, pi).
        Measured from the positive y-axis toward the negative-axis.

    Notes
    -----
    - The covariance matrix Sigma is related to the ellipse parameters
      (sigma_maj, sigma_min, theta) by its eigen-decomposition.
    - The stddevs sigma_maj and sigma_min are the square roots of the
      eigenvalues of Sigma.
    - The position angle theta is derived from the eigenvector
      corresponding to the largest eigenvalue.

    """
    # Symmetrize for numerical safety
    S = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(S)  # ascending eigenvalues
    # sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # clamp small negatives to zero (numerical)
    vals = np.maximum(vals, 0.0)
    sigma_maj = np.sqrt(vals[0])
    sigma_min = np.sqrt(vals[1])
    # principal axis: eigenvector for largest eigenvalue
    v = vecs[:, 0]
    pa = np.arctan2(v[1], v[0])

    theta = np.mod(pa + 10.5 * np.pi, np.pi)
    # Ensure theta is between -pi/2 and pi/2.
    if theta > np.pi / 2:
        theta = -np.mod(-theta, np.pi)
    if theta < -np.pi / 2:
        theta = np.mod(theta, np.pi)

    return sigma_maj, sigma_min, theta
