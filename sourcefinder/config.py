from collections import defaultdict
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

from sourcefinder.utility.sourceparams import SourceParams, _file_fields

T = TypeVar("T")


def _is_dataclass(_type: Type[T], /) -> bool:
    """Remove ``TypeGuard`` from is_dataclass.

    see: https://github.com/python/mypy/issues/14941

    """
    return is_dataclass(_type)


# map of types that maybe converted to match the expected type
_compat_types: defaultdict[type, set[type]] = defaultdict(set, {int: {float}})


def assert_t(key: str, value, *types: type):
    """Assert value is of one of the types

    ``key`` is the TOML configuration key the value is associated to.
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

    ``key`` is the TOML configuration key being validated.  ``value``
    should be of type ``origin_t[args]``.  It is passed to this
    function separately to avoid recomputing the type again.

    When the type is a ``list``, the value is tested recursively.  On
    recursive calls, the list index is appended to the key.  For
    ``dict``-s, iterate over all key-value pairs and validated.

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

    ``key`` is the TOML configuration key the value is associated to.
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
        for (key, type_), val in zip(
            get_type_hints(self).items(), astuple(self)
        ):
            validate_types(key, val, type_)


_structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


_source_params = [p.value for p in SourceParams.__members__.values()]

_source_params_file = [SourceParams[field].value for field in _file_fields]


@dataclass(frozen=True)
class ImgConf(_Validate):
    """Configuration that should cover all the specifications for processing the image."""

    interpolate_order: int = 1
    """Order of interpolation to use for the background mean and
    background standard deviation (rms) maps (e.g. 1 for linear)

    """

    median_filter: int = 0
    """Size of the median filter to apply to background and RMS grids
    prior to interpolating. This is used to discard outliers. Use 0 to
    disable.

    """

    mf_threshold: int = 0
    """Threshold (Jy/beam) used with the median filter if
    median_filter is non-zero. This is used to only discard outliers
    (i.e. extreme background mean or rms node values) beyond a certain
    threshold. Use 0 to disable.

    """

    rms_filter: float = 0.001
    """Any interpolated background standard deviation (rms) value
    should be above this threshold times the median of all background
    standard deviation (rms) node values. This is used to avoid
    picking up sources towards the edges of the image where the values
    of the background rms map may be the result of poor interpolation,
    i.e. are the result of extrapolation rather than
    interpolation. Use 0 to disable.

    """

    deblend_mincont: float = 0.005
    """Minimum flux density fraction (relative to the original,
    i.e. unblended, island) required for a subisland to be considered
    a valid deblended component.

    """

    structuring_element: list[list[int]] = field(
        default_factory=lambda: _structuring_element
    )
    """The "structuring element" defines island connectivity as in
    "4-connectivity" and "8-connectivity". These two are the only
    reasonable choices, since the structuring element must be
    centrosymmetric.  The structuring element is applied in
    scipy.ndimage.label, so check its documentation for some
    background on its use.

    """

    vectorized: bool = False
    """Measure sources in a vectorized way. Expect peak spectral
    brightnesses with a lower bias (downwards) than for Gaussian fits
    (also downwards), but with a higher bias (upwards for both) for
    the elliptical axes.

    """

    nr_threads: int | None = None
    """The number of threads used to parallelize Gaussian fits to detected 
    sources.
    Note: this does not change numba's 'num threads' for parallel numba operations.
    """

    margin: int = 0
    """Margin in pixels to ignore near the edge of the image, i.e.
    sources within this margin will not be detected."""

    radius: float = 0.0
    """Radius in pixels (from image center) considered valid, i.e. sources
    beyond this radius will not be detected.

    """

    back_size_x: int | None = 32
    """Subimage size for estimation of background node values (X
    direction). The nodes are centred on the subimages.

    """

    back_size_y: int | None = 32
    """Subimage size for estimation of background node values (Y
    direction). The nodes are centred on the subimages.

    """

    eps_ra: float = 0.0
    """Calibration uncertainty in right ascension (degrees), see
    equation 27a of the NVSS paper.

    """

    eps_dec: float = 0.0
    """Calibration uncertainty in declination (degrees), see equation
    27b of the NVSS paper.

    """

    clean_bias: float = 0.0
    """Clean bias to subtract from the peak brightnesses (Jy/beam), see
    parapagraph 5.2.5 and equation 34 of the NVSS paper.

    """

    clean_bias_error: float = 0.0
    """1-sigma uncertainty in clean bias (Jy/beam), see parapagraph 5.2.5 and
    equation 37 of the NVSS paper.

    """

    frac_flux_cal_error: float = 0.0
    """Intensity-proportional calibration uncertainty, see paragraph 5.2.5 and 
    equation 37 of the NVSS paper.

    """

    alpha_maj1: float = 2.5
    """First exponent for scaling errors along the fitted major 
    axis, see equation 26 and paragraph 5.2.3 of the NVSS paper and 
    equation 41 and paragraph 3 of Condon's (1997) "Errors in Elliptical 
    Gaussian Fits".

    """

    alpha_maj2: float = 0.5
    """Second exponent for scaling errors along the fitted major 
    axis, see equation 26 and paragraph 5.2.3 of the NVSS paper and
    equation 41 and paragraph 3 of Condon's (1997) "Errors in Elliptical
    Gaussian Fits".

    """

    alpha_min1: float = 0.5
    """First exponent for scaling errors along the fitted minor 
    axis and for scaling errors in the position angle, see equation 26 and 
    paragraph 5.2.3 of the NVSS paper and equation 41 and paragraph 3 of 
    Condon's (1997) "Errors in Elliptical Gaussian Fits".

    """

    alpha_min2: float = 2.5
    """Second exponent for scaling errors along the fitted minor 
    axis and for scaling errors in the position angle, see equation 26 and 
    paragraph 5.2.3 of the NVSS paper and equation 41 and paragraph 3 of 
    Condon's (1997) "Errors in Elliptical Gaussian Fits".

    """

    alpha_brightness1: float = 1.5
    """First exponent for scaling errors in peak brightness, see
    equation 26 and paragraph 5.2.5 of the NVSS paper and equation 41
    and paragraph 3 of Condon's (1997) "Errors in Elliptical Gaussian
    Fits".

    """

    alpha_brightness2: float = 1.5
    """Second exponent for scaling errors in peak brightness, see
    equation 26 and paragraph 5.2.5 of the NVSS paper and equation 41
    and paragraph 3 of Condon's (1997) "Errors in Elliptical Gaussian
    Fits".

    """

    detection_thr: float = 10.0
    """Detection threshold as multiple of the background standard
    deviation (rms) map, after the background mean values have been
    subtracted from the image.

    """

    analysis_thr: float = 3.0
    """Analysis threshold as multiple of the background standard
    deviation (rms) map, after the background mean values have been
    subtracted from the image.

    """

    fdr: bool = False
    """Use False Detection Rate (FDR) algorithm for determining
    detection threshold.

    """

    alpha: float = 1e-2
    """FDR alpha value (float, default 0.01) that sets an upper limit
    on the fraction of pixels erroneously detected as source pixels,
    relative to all source pixels.  This requirement should be met
    when averaged over a large ensemble of images, but problems were
    encountered with alpha as low as 0.001, see paragraph 3.6 of
    Spreeuw's thesis.

    """

    deblend_nthresh: int = 0
    """Number of deblending subthresholds; 0 to disable."""

    grid: int | None = None
    """Background subimage size used as fallback for back_size_x and
    back_size_y. If both are not set, this implies
    back_size_x=backsize_y=grid, i.e. the subimages are squares.

    """

    bmaj: float | None = None
    """Set beam: Major axis of restoring beam (degrees)."""

    bmin: float | None = None
    """Set beam: Minor axis of restoring beam (degrees)."""

    bpa: float | None = None
    """Set beam: Restoring beam position angle (degrees)."""

    force_beam: bool = False
    """Force source shape to align restoring beam shape (bmaj, bmin,
    bpa) for Gauss fits and vetorized source measurement, i.e. when
    vectorized=True (as of 2025-06-13: upcoming, issue #131).

    """

    detection_image: str | None = None
    """Path to detection map. PySE will identify sources and the
    positions of pixels which comprise them on the detection image,
    but then use the corresponding pixels on the target images to
    perform measurements. Of course, the detection image and the
    target image(s) must have the same pixel dimensions. Note that
    only a single detection image may be specified, and the same
    pixels are then used on all target images. Note further that this
    detection-image option is incompatible with --fdr

    """

    fixed_posns: str | None = None
    """JSON __list__ of RA, Dec pairs of coordinates to measure
    sources at (disables blind extraction and vectorized source
    measurements).

    """

    fixed_posns_file: str | None = None
    """Path to JSON file with RA, Dec pairs of coordinates to measure
    sources at (disables blind extraction and vectorized source
    measurements).

    """

    ffbox: float = 3.0
    """When fitting to a fixed position, a square “box” of pixels is
    chosen around the requested position, and the optimization
    procedure allows the source position to vary within that box. The
    size of the box may be changed with this option. Note that this
    parameter is given in units of the major axis of the beam in
    pixels.

    """

    ew_sys_err: float = 0.0
    """Systematic error in east-west direction, see paragraph 5.2.3
    of the NVSS paper. Note that this parameter is currently not
    applied in PySE, because it should be considered a final step
    before entering source parameters in a catalog, i.e. it is simply
    returned to allow for systematic positional offset cf. the
    NVSS. Therefore, its unit (degrees, arcseconds) is up to the user.

    """

    ns_sys_err: float = 0.0
    """Systematic error in north-south direction, see paragraph 5.2.3
    of the NVSS paper. Note that this parameter is currently not
    applied in PySE, because it should be considered a final step
    before entering source parameters in a catalog, i.e. it is simply
    returned to allow for systematic positional offset cf. the
    NVSS. Therefore, its unit (degrees, arcseconds) is up to the user.

    """


@dataclass(frozen=True)
class ExportSettings(_Validate):
    """Selection of output, related to detected sources and/or intermediate
    image processing products"""

    output_dir: str = "."
    """Directory in which to write the output files."""

    file_type: str = "csv"
    """Output file type (default: csv)."""

    skymodel: bool = False
    """Generate sky model."""

    csv: bool = False
    """Generate CSV text file (e.g., for TopCat)."""

    regions: bool = False
    """Generate DS9 region file(s)."""

    rmsmap: bool = False
    """Generate RMS map."""

    sigmap: bool = False
    """Generate significance map."""

    residuals: bool = False
    """Generate residual maps."""

    islands: bool = False
    """Generate island maps."""

    reconvert: bool = True
    """ Only applies to vectorized source meaurements,
    i.e. when both ImgConf.vectorized==True and ImgConf.deblend_thresholds=0.
    If True, the results will be converted to the same format as
    for non-vectorized source measurements, i.e. a
    `utility.containers.ExtractionResults` object. If False,
    the results will be stored in a Pandas DataFrame, which is much
    faster. """

    source_params: list[str] = field(default_factory=lambda: _source_params)
    """Collect all possible source parameters."""

    source_params_file: list[str] = field(
        default_factory=lambda: _source_params_file
    )
    """ Source parameters to include a file for storage."""


@dataclass(frozen=True)
class Conf:
    image: ImgConf
    export: ExportSettings

    def __post_init__(self):  # noqa: D105
        for key, field_t in get_type_hints(self).items():
            value = getattr(self, key)
            if _is_dataclass(field_t) and isinstance(value, dict):
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
    if path is None:
        data = {"tool": {"pyse": {"image": {}, "export": {}}}}
    else:
        data_raw = tomllib.loads(Path(path).read_text())
        data = normalize_none_values(data_raw)

    conf = data.get("tool", {}).get("pyse", {})
    if not conf:
        match data:
            case {"tool": {"pyse": dict(), **_rest1}, **_rest2}:
                raise KeyError("tool.pyse: empty section in config file")
            case {"tool": dict(), **_rest}:
                raise KeyError(
                    "tool.pyse: section for PySE missing in config file"
                )
            case _:
                raise KeyError(
                    "tool: top-level section missing in config file"
                )
    return Conf(**conf)
