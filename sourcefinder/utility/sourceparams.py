from enum import Enum
import pandas as pd


class SourceParams(str, Enum):
    """Enumeration of source parameters that can be measured and stored."""

    PEAK = "peak"
    PEAK_ERR = "peak_err"
    FLUX = "flux"
    FLUX_ERR = "flux_err"
    X = "x"
    X_ERR = "x_err"
    Y = "y"
    Y_ERR = "y_err"
    SMAJ = "smaj"
    SMAJ_ERR = "smaj_err"
    SMIN = "smin"
    SMIN_ERR = "smin_err"
    THETA = "theta"
    THETA_ERR = "theta_err"
    SMAJ_DC = "smaj_dc"
    SMAJ_DC_ERR = "smaj_dc_err"
    SMIN_DC = "smin_dc"
    SMIN_DC_ERR = "smin_dc_err"
    THETA_DC = "theta_dc"
    THETA_DC_ERR = "theta_dc_err"
    RA = "ra"
    RA_ERR = "ra_err"
    DEC = "dec"
    DEC_ERR = "dec_err"
    SMAJ_ASEC = "smaj_asec"
    SMAJ_ASEC_ERR = "smaj_asec_err"
    SMIN_ASEC = "smin_asec"
    SMIN_ASEC_ERR = "smin_asec_err"
    THETA_CELES = "theta_celes"
    THETA_CELES_ERR = "theta_celes_err"
    THETA_DC_CELES = "theta_dc_celes"
    THETA_DC_CELES_ERR = "theta_dc_celes_err"
    ERROR_RADIUS = "error_radius"
    SIG = "sig"
    CHISQ = "chisq"
    REDUCED_CHISQ = "reduced_chisq"

    def describe(self) -> str:
        """Return a description of the source parameter."""
        return _source_params_descriptions[self.value]


_source_params_descriptions = {
    "peak": "Peak spectral brightness of the source (Jy/beam)",
    "peak_err": (
        "1-sigma uncertainty in the peak spectral "
        "brightness of the source (Jy/beam)"
    ),
    "flux": (
        "Flux density of the source, calculated as 'pi * peak "
        "spectral brightness * semi- major axis * semi-minor "
        "axis / beamsize' (Jy)"
    ),
    "flux_err": "1-sigma uncertainty in the flux density (Jy)",
    "x": (
        "x-position (float) of the barycenter of the source, "
        "correponding to the row index of the Numpy array with "
        "image data. After loading a FITS image, the data is "
        "transposed such that x and y are aligned with ds9 viewing, "
        "except for an offset of 1 pixel, since the bottom left "
        "pixel in ds9 has x=y=1"
    ),
    "x_err": (
        "1-sigma uncertainty in the x-position (float) of the "
        "barycenter of the source, corresponding to the row index "
        "of the Numpy array with image data"
    ),
    "y": (
        "y-position (float) of the barycenter of the source, "
        "correponding to the column index of the Numpy array with "
        "image data. After loading a FITS image, the data is "
        "transposed such that x and y are aligned with ds9 viewing, "
        "except for an offset of 1 pixel, since the bottom left "
        "pixel in ds9 has x=y=1"
    ),
    "y_err": (
        "1-sigma uncertainty in the y-position (float) of the "
        "barycenter of the source, corresponding to the column "
        "index of the Numpy array with image data"
    ),
    "smaj": (
        "Semi-major axis of the Gaussian profile, "
        "not deconvolved from the clean beam (pixels)"
    ),
    "smaj_err": (
        "1-sigma uncertainty in the semi-major axis, "
        "not deconvolved from the clean beam (pixels)"
    ),
    "smin": (
        "Semi-minor axis of the Gaussian profile, "
        "not deconvolved from the clean beam (pixels)"
    ),
    "smin_err": (
        "1-sigma uncertainty in the semi-minor axis, "
        "not deconvolved from the clean beam (pixels)"
    ),
    "theta": (
        "Position angle of the major axis of the "
        "Gaussian profile, measured from the positive y-axis "
        " towards the negative x-axis (radians)"
    ),
    "theta_err": (
        "1-sigma uncertainty in the position angle "
        "of the major axis of the Gaussian profile, "
        "measured from the positive y-axis towards the negative x-axis "
        "(radians)"
    ),
    "smaj_dc": (
        "Semi-major axis of the Gaussian profile, "
        "deconvolved from the clean beam (pixels)"
    ),
    "smaj_dc_err": (
        "1-sigma uncertainty in the semi-major axis, "
        "deconvolved from the clean beam (pixels)"
    ),
    "smin_dc": (
        "Semi-minor axis of the Gaussian profile, "
        "deconvolved from the clean beam (pixels)"
    ),
    "smin_dc_err": (
        "1-sigma uncertainty in the semi-minor axis, "
        "deconvolved from the clean beam (pixels)"
    ),
    "theta_dc": (
        "Position angle of the major axis of the "
        "Gaussian profile, deconvolved from the clean beam, "
        "measured from the positive y-axis towards the negative x-axis ("
        "degrees)"
    ),
    "theta_dc_err": (
        "1-sigma uncertainty in the position angle "
        "of the major axis of the Gaussian profile, "
        "deconvolved from the clean beam, measured from the positive y-axis "
        "towards the negative x-axis (degrees)"
    ),
    "ra": "Right ascension of the source (degrees)",
    "ra_err": "1-sigma uncertainty in right ascension (degrees)",
    "dec": "Declination of the source (degrees)",
    "dec_err": "1-sigma uncertainty in declination (degrees)",
    "smaj_asec": (
        "Semi-major axis of the Gaussian profile, "
        "not deconvolved from the clean beam (arcseconds)"
    ),
    "smaj_asec_err": (
        "1-sigma uncertainty in the semi-major axis, "
        "not deconvolved from the clean beam "
        "(arcseconds)"
    ),
    "smin_asec": (
        "Semi-minor axis of the Gaussian profile, "
        "not deconvolved from the clean beam (arcsecond)"
    ),
    "smin_asec_err": (
        "1-sigma uncertainty in the semi-minor axis, "
        "not deconvolved from the clean beam "
        "(arcseconds)"
    ),
    "theta_celes": (
        "Position angle of the major axis of the "
        "Gaussian profile, measured east from local north "
        "(degrees)"
    ),
    "theta_celes_err": (
        "1-sigma uncertainty in the position angle "
        "of the major axis of the Gaussian profile, "
        "measured east from local north (degrees)"
    ),
    "theta_dc_celes": (
        "Position angle of the major axis of the "
        "Gaussian profile, deconvolved from the clean beam, "
        "measured east from local north (degrees)"
    ),
    "theta_dc_celes_err": (
        "1-sigma uncertainty in the position angle "
        "of the major axis of the Gaussian profile, "
        "deconvolved from the clean beam, measured east from local north "
        "(degrees)"
    ),
    "error_radius": (
        "The absolute angular error on the position of the source "
        "(arcseconds). This is a pessimistic estimate, because we try "
        "all possible combinations of the x and y errors, and take the "
        "maximum for the four combinations."
    ),
    "sig": (
        "The significance of a detection (float) is defined as "
        "the maximum signal-to-noise ratio across the island. "
        "Often this will be the ratio of the maximum pixel value "
        "of the source divided by the noise at that position."
    ),
    "chisq": (
        "The chi-squared value of the Gaussian model relative to "
        "the data (float). Can be a Gaussian model derived from a fit "
        "or from moments. See the measuring.goodness_of_fit docstring "
        "for some important notes."
    ),
    "reduced_chisq": (
        "The reduced chi-squared value of the Gaussian "
        "model relative to the data (float). Can be a "
        "Gaussian model derived from a fit or from "
        "moments. See the measuring.goodness_of_fit "
        "docstring for some important notes."
    ),
}

# Ensure that all source parameters have a description
assert all(p.value in _source_params_descriptions for p in SourceParams)

# This should render source parameter descriptions in Sphinx/RTD.
for member in SourceParams:
    member.__doc__ = SourceParams.describe(member)

# Set default set of source parameters to store in a file, e.g. a .csv file.
_file_fields = [
    "PEAK",
    "PEAK_ERR",
    "FLUX",
    "FLUX_ERR",
    "X",
    "Y",
    "RA",
    "RA_ERR",
    "DEC",
    "DEC_ERR",
    "SMAJ_ASEC",
    "SMAJ_ASEC_ERR",
    "SMIN_ASEC",
    "SMIN_ASEC_ERR",
    "THETA_CELES",
    "THETA_CELES_ERR",
    "SIG",
    "REDUCED_CHISQ",
]


def make_measurements_dataframe(
    moments_of_sources,
    sky_barycenters,
    ra_errors,
    dec_errors,
    smaj_asec,
    errsmaj_asec,
    smin_asec,
    errsmin_asec,
    theta_celes_values,
    theta_celes_errors,
    theta_dc_celes_values,
    theta_dc_celes_errors,
    error_radii,
    sig,
    chisq,
    reduced_chisq,
):
    """
    Create a Pandas DataFrame with parameters related to detected sources
    from a subset of the tuple of Numpy ndarrays returned by the
    `extract.source_measurements_vectorised` function.
    """
    columns = {
        SourceParams.PEAK: moments_of_sources[:, 0, 0],
        SourceParams.PEAK_ERR: moments_of_sources[:, 1, 0],
        SourceParams.FLUX: moments_of_sources[:, 0, 1],
        SourceParams.FLUX_ERR: moments_of_sources[:, 1, 1],
        SourceParams.X: moments_of_sources[:, 0, 2],
        SourceParams.X_ERR: moments_of_sources[:, 1, 2],
        SourceParams.Y: moments_of_sources[:, 0, 3],
        SourceParams.Y_ERR: moments_of_sources[:, 1, 3],
        SourceParams.SMAJ: moments_of_sources[:, 0, 4],
        SourceParams.SMAJ_ERR: moments_of_sources[:, 1, 4],
        SourceParams.SMIN: moments_of_sources[:, 0, 5],
        SourceParams.SMIN_ERR: moments_of_sources[:, 1, 5],
        SourceParams.THETA: moments_of_sources[:, 0, 6],
        SourceParams.THETA_ERR: moments_of_sources[:, 1, 6],
        SourceParams.SMAJ_DC: moments_of_sources[:, 0, 7],
        SourceParams.SMAJ_DC_ERR: moments_of_sources[:, 1, 7],
        SourceParams.SMIN_DC: moments_of_sources[:, 0, 8],
        SourceParams.SMIN_DC_ERR: moments_of_sources[:, 1, 8],
        SourceParams.THETA_DC: moments_of_sources[:, 0, 9],
        SourceParams.THETA_DC_ERR: moments_of_sources[:, 1, 9],
        SourceParams.RA: sky_barycenters[:, 0],
        SourceParams.RA_ERR: ra_errors.ravel(),
        SourceParams.DEC: sky_barycenters[:, 1],
        SourceParams.DEC_ERR: dec_errors.ravel(),
        SourceParams.SMAJ_ASEC: smaj_asec,
        SourceParams.SMAJ_ASEC_ERR: errsmaj_asec,
        SourceParams.SMIN_ASEC: smin_asec,
        SourceParams.SMIN_ASEC_ERR: errsmin_asec,
        SourceParams.THETA_CELES: theta_celes_values,
        SourceParams.THETA_CELES_ERR: theta_celes_errors.ravel(),
        SourceParams.THETA_DC_CELES: theta_dc_celes_values.ravel(),
        SourceParams.THETA_DC_CELES_ERR: theta_dc_celes_errors.ravel(),
        SourceParams.ERROR_RADIUS: error_radii.ravel(),
        SourceParams.SIG: sig,
        SourceParams.CHISQ: chisq,
        SourceParams.REDUCED_CHISQ: reduced_chisq,
    }

    return pd.DataFrame(columns)


def describe_dataframe_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        col: _source_params_descriptions[col]
        for col in df.columns
        if col in _source_params_descriptions
    }
