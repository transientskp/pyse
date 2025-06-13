from collections import defaultdict
from dataclasses import asdict
from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import field
from dataclasses import is_dataclass
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from types import UnionType
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Container
from typing import Type
from typing import TypeVar
from warnings import warn
from enum import Enum

T = TypeVar("T")


def unguarded_is_dataclass(_type: Type[T], /) -> bool:
    """Remove :ref:`TypeGuard` from is_dataclass.

    see: https://github.com/python/mypy/issues/14941
    """
    return is_dataclass(_type)


# map of types that maybe converted to match the expected type
_compat_types: defaultdict[type, set[type]] = defaultdict(set, {int: {float}})


def assert_t(key: str, value, *types: type):
    """Assert value is of one of the types

    `key` is the TOML configuration key the value is associated to.
    It is used to generate a meaningful error message.

    """
    assert len(types) > 0, "need at least one type to assert"
    msg = f"{key}: type({value!r}) "
    if len(types) > 1:
        msg += f"∉ {{{', '.join(map(str, types))}}}"
    else:
        msg += f"!= {types[0]}"

    try:
        assert isinstance(value, types), msg
    except AssertionError:
        # NOTE: check if types are compatible
        if not _compat_types[type(value)].intersection(types):
            raise


def validate_nested(key: str, value, origin_t, args):
    """Validate nested types allowed in TOML

    `key` is the TOML configuration key being validated.  `value`
    should be of type `origin_t[args]`.  It is passed to this function
    separately to avoid recomputing the type again.

    When the type is a `list`, the value is tested recursively.  On
    recursive calls, the list index is appended to the key.  For
    `dict`s, iterate over all key-value pairs and validated.

    """
    # NOTE: only support TOML types
    if issubclass(origin_t, list):
        assert_t(key, value, list)
        # NOTE: unspecified type => Any; can't check
        if not args:
            return
        for i, v in enumerate(value):
            validate_types(f"{key}[{i}]", v, args[0])
    elif issubclass(origin_t, dict):
        assert_t(key, value, dict)
        for k, v in value.items():
            validate_types(f"{key}[{k!r}]", v, args[1])
    else:
        warn(f"{key}: unsupported type {origin_t[args]}, cannot validate")


def validate_types(key: str, value, type_: type):
    """Validate types, dispatch on generic or POD types

    `key` is the TOML configuration key the value is associated to.
    It is used to generate a meaningful error message.

    """
    match get_origin(type_):
        case type() as origin_t if issubclass(origin_t, Container):
            validate_nested(key, value, origin_t, get_args(type_))
        case type() as origin_t if issubclass(origin_t, UnionType):
            assert_t(key, value, *get_args(type_))
        case type():
            warn(f"{key}: unsupported type {type_}, cannot validate")
        case None:
            # NOTE: plain old data types
            assert_t(key, value, type_)


@dataclass(frozen=True)
class _Validate:
    def __post_init__(self):
        for (key, type_), val in zip(get_type_hints(self).items(), astuple(self)):
            validate_types(key, val, type_)


_structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

class SourceParam(str, Enum):
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
    FLUX = "flux"
    FLUX_ERR = "flux_err"
    PEAK = "peak"
    PEAK_ERR = "peak_err"
    X = "x"
    Y = "y"
    SIG = "sig"
    REDUCED_CHISQ = "reduced_chisq"

    def describe(self):
        return {
            "ra": "Right ascension of the source (degrees)",
            "ra_err": "1-sigma uncertainty in right ascension (degrees)",
            "dec": "Declination of the source (degrees)",
            "dec_err": "1-sigma uncertainty in declination (degrees)",
            "smaj_asec": ("Semi-major axis of the Gaussian profile, "
                          "not deconvolved from the clean beam (arcseconds)"),
            "smaj_asec_err": ("1-sigma uncertainty in the semi-major axis, "
                              "not deconvolved from the clean beam "
                              "(arcseconds)"),
            "smin_asec": ("Semi-minor axis of the Gaussian profile, "
                          "not deconvolved from the clean beam (arcsecond)"),
            "smin_asec_err": ("1-sigma uncertainty in the semi-minor axis, "
                              "not deconvolved from the clean beam "
                              "(arcseconds)"),
            "theta_celes": ("Position angle of the major axis of the "
                           "Gaussian profile, measured east from local north "
                           "(degrees)"),
            "theta_celes_err": ("1-sigma uncertainty in the position angle "
                                "of the major axis of the Gaussian profile, "
                                "measured east from local north (degrees)"),
            "flux": ("Flux density of the source, calculated as 'pi * peak "
                     "spectral brightness * semi- major axis * semi-minor "
                     "axis / beamsize' (Jy)"),
            "flux_err": "1-sigma uncertainty in the flux density (Jy)",
            "peak": "Peak spectral brightness of the source (Jy/beam)",
            "peak_err": ("1-sigma uncertainty in the peak spectral "
                         "brightness of the source (Jy/beam)"),
            "x": ("x-position (float) of the barycenter of the source, "
                  "correponding to the row index of the Numpy array with "
                  "image data. After loading a FITS image, the data is "
                  "transposed such that x and y are aligned with ds9 viewing, "
                  "except for an offset of 1 pixel, since the bottom left "
                  "pixel in ds9 has x=y=1"),
            "y": ("y-position (float) of the barycenter of the source, "
                  "correponding to the column index of the Numpy array with "
                  "image data. After loading a FITS image, the data is "
                  "transposed such that x and y are aligned with ds9 viewing, "
                  "except for an offset of 1 pixel, since the bottom left "
                  "pixel in ds9 has x=y=1"),
            "sig": ("The significance of a detection (float) is defined as "
                    "the maximum signal-to-noise ratio across the island. "
                    "Often this will be the ratio of the maximum pixel value "
                    "of the source divided by the noise at that position."),
            "reduced_chisq": ("The reduced chi-squared value of the Gaussian "
                              "model relative to the data (float). Can be a "
                              "Gaussian model derived from a fit or from "
                              "moments. See the measuring.goodness_of_fit "
                              "docstring for some important notes.")
        }[self.value]

_source_params = [p.value for p in SourceParam.__members__.values()]


@dataclass(frozen=True)
class ImgConf(_Validate):
    """Configuration that should cover all the specifications for processing the image."""

    interpolate_order: int = 1  # Order of interpolation to use for
    # the background mean and background standard deviation (rms) maps (e.g. 1
    # for linear)
    median_filter: int = 0      # Size of the median filter to apply to
    # background and RMS grids prior to interpolating. This is used to
    # discard outliers. Use 0 to disable.
    mf_threshold: int = 0       # Threshold (Jy/beam) used with the median
    # filter if median_filter is non-zero. This is used to only discard
    # outliers (i.e. extreme background mean or rms node values) beyond a
    # certain threshold. Use 0 to disable.
    rms_filter: float = 0.001   # Any interpolated background standard
    # deviation (rms) value should be above this threshold times the mean of
    # all background standard deviation (rms) node values. This is used to
    # avoid picking up sources towards the edges of the image where the values
    # of the background rms map may be the result of poor interpolation,
    # i.e. are the result of extrapolation rather than interpolation. Use 0 to
    # disable.
    deblend_mincont: float = 0.005  # Minimum flux density fraction (relative
    # to the original, i.e. unblended, island) required for a subisland to be
    # considered a valid deblended component.

    # The "structuring element" defines island connectivity as in
    # "4-connectivity" and "8-connectivity". These two are the only reasonable
    # choices, since the structuring element must be centrosymmetric.
    # The structuring element is applied in scipy.ndimage.label, so check its
    # documentation for some background on its use.
    structuring_element: list[list[int]] = field(
        default_factory=lambda: _structuring_element
    )
    vectorized: bool = False               # Measure sources in a vectorized
    # way. Expect peak spectral brightnesses with a lower bias (downwards) than
    # for Gaussian fits (also downwards), but with a higher bias (upwards
    # for both) for the elliptical axes.
    allow_multiprocessing: bool = True     # Allow multiprocessing for Gaussian
    # fitting in parallel.
    margin: int = 0                        # Margin in pixels to ignore around
    # the edge of the image.
    radius: float = 0.0                    # Radius in pixels (from
    # image center around sources to include in analysis.
    back_size_x: int | None = 32           # Subimage size for estimation of
    # background node values (X direction). The nodes are centred on the
    # subimages.
    back_size_y: int | None = 32            # Subimage size for estimation of
    # background node values (Y direction). The nodes are centred on the
    # subimages.
    eps_ra: float = 0.0                    # Calibration uncertainty in right
    # ascension (degrees), see equation 27a of the NVSS paper.
    eps_dec: float = 0.0                   # Calibration uncertainty in
    # declination (degrees), see equation 27b of the NVSS paper.
    detection: float = 10.0                # Detection threshold as multiple of
    # the background standard deviation (rms) map, after the background mean
    # values have been subtracted from the image.
    analysis: float = 3.0                  # Analysis threshold as multiple of
    # the background standard deviation (rms) map, after the background mean
    # values have been subtracted from the image.
    fdr: bool = False                      # Use False Detection Rate (FDR)
    # algorithm for determining detection threshold.
    alpha: float = 1e-2                    # FDR alpha value (float,
    # default 0.01) that sets an upper limit on the fraction of pixels
    # erroneously detected as source pixels, relative to all source pixels.
    # This requirement should be met when averaged over a large ensemble of
    # images, but problems were encountered with alpha as low as 0.001,
    # see paragraph 3.6 of Spreeuw's thesis.
    deblend_thresholds: int = 0            # Number of deblending
    # subthresholds; 0 to disable.
    grid: int | None = None                # Background subimage size used
    # as fallback for back_size_x and back_size_y. If both are not set,
    # this implies back_size_x=backsize_y=grid, i.e. the subimages are squares.
    bmaj: float | None = None              # Set beam: Major axis of
    # restoring beam (degrees).
    bmin: float | None = None              # Set beam: Minor axis of
    # restoring beam (degrees).
    bpa: float | None = None               # Set beam: Restoring beam position
    # angle (degrees).
    force_beam: bool = False               # Force source shape to align
    # restoring beam shape (bmaj, bmin, bpa) for Gauss fits and vetorized
    # source measurement, i.e. when vectorized=True (as of 2025-06-13:
    # upcoming, issue #131).
    detection_image: str | None = None     # Path to detection map. PySE will
    # identify sources and the positions of pixels which comprise them on the
    # detection image, but then use the corresponding pixels on the target
    # images to perform measurements. Of course, the detection image and
    # the target image(s) must have the same pixel dimensions. Note that only
    # a single detection image may be specified, and the same pixels are then
    # used on all target images. Note further that this detection-image option
    # is incompatible with --fdr
    fixed_posns: str | None = None         # JSON __list__ of RA, Dec pairs of
    # coordinates to measure sources at (disables blind extraction and
    # vectorized source measurements).
    fixed_posns_file: str | None = None    # Path to JSON file with RA, Dec
    # pairs of coordinates to measure sources at (disables blind extraction
    # and vectorized source measurements).
    ffbox: float = 3.0                     # When fitting to a fixed position,
    # a square “box” of pixels is chosen around the requested position, and the
    # optimization procedure allows the source position to vary within that
    # box. The size of the box may be changed with this option. Note that this
    # parameter is given in units of the major axis of the beam in pixels.
    ew_sys_err: float = 0.                 # Systematic error in east-west
    # direction, see paragraph 5.2.3 of the NVSS paper. Note that this
    # parameter is currently not applied in PySE, because it should be
    # considered a final step before entering source parameters in a catalog,
    # i.e. it is simply returned to allow for systematic positional offset
    # cf. the NVSS. Therefore, its unit (degrees, arcseconds) is up to the
    # user.
    ns_sys_err: float = 0.                 # Systematic error in north-south
    # direction, see paragraph 5.2.3 of the NVSS paper. Note that this
    # parameter is currently not applied in PySE, because it should be
    # considered a final step before entering source parameters in a catalog,
    # i.e. it is simply returned to allow for systematic positional offset
    # cf. the NVSS. Therefore, its unit (degrees, arcseconds) is up to the
    # user.



@dataclass(frozen=True)
class ExportSettings(_Validate):
    """Selection of output, related to detected sources and/or intermediate image processing products"""

    output_dir: str = "."                   # Directory in which to write the output files
    file_type: str = "csv"                  # Output file type (default: csv).
    skymodel: bool = False                  # Generate sky model.
    csv: bool = False                       # Generate CSV text file (e.g., for TopCat).
    regions: bool = False                   # Generate DS9 region file(s).
    rmsmap: bool = False                    # Generate RMS map.
    sigmap: bool = False                    # Generate significance map.
    residuals: bool = False                 # Generate residual maps.
    islands: bool = False                   # Generate island maps.
    source_params: list[str] = field(       # Source parameters to include in the output.
        default_factory=lambda: _source_params
    )


@dataclass(frozen=True)
class Conf:
    image: ImgConf
    export: ExportSettings

    def __post_init__(self):  # noqa: D105
        for key, field_t in get_type_hints(self).items():
            value = getattr(self, key)
            if unguarded_is_dataclass(field_t) and isinstance(value, dict):
                # NOTE: have to do it like this since inherited
                # dataclasses are frozen
                super().__setattr__(key, field_t(**value))

def normalize_none_values(val):
    if isinstance(val, dict):
        return {k: normalize_none_values(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [normalize_none_values(v) for v in val]
    elif isinstance(val, str) and val.strip().lower() == "none":
        return None
    else:
        return val

def read_conf(path: str | Path):
    data_raw = tomllib.loads(Path(path).read_text())
    data = normalize_none_values(data_raw)
    conf = data.get("tool", {}).get("pyse", {})
    if not conf:
        match data:
            case {"tool": {"pyse": dict(), **_rest1}, **_rest2}:
                raise KeyError("tool.pyse: empty section in config file")
            case {"tool": dict(), **_rest}:
                raise KeyError("tool.pyse: section for PySE missing in config file")
            case _:
                raise KeyError("tool: top-level section missing in config file")
    return Conf(**conf)
