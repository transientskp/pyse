from collections import defaultdict
from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import field, fields
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

from sourcefinder.utility.sourceparams import SourceParams, file_fields

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

_source_params_file = [
    SourceParams.__members__[field].value for field in file_fields
]

IMGCONF_HELP: dict[str, str] = {
    "interpolate_order": """
Order of interpolation - e.g. 1 for linear - to use to derive the background 
mean and background standard deviation (rms) maps from the corresponding 
background grid values. The nodes of the background grids are centred on the 
subimages of size back_size_x by back_size_y.
""",
    "median_filter": """
Size of the median filter to apply to background and RMS grids prior to 
interpolating. This is used to discard outliers. Use 0 to disable.
""",
    "mf_threshold": """
Threshold (Jy/beam) used with the median filter if median_filter is 
non-zero. This is used to only discard outliers (i.e. extreme background 
mean or rms node values) beyond a certain threshold. Use 0 to disable.
""",
    "rms_filter": """
Any interpolated background standard deviation (rms) value
should be above this threshold times the median of all background
standard deviation (rms) node values. This is used to avoid
picking up sources towards the edges of the image where the values
of the background rms map may be the result of poor interpolation,
i.e. are the result of extrapolation rather than
interpolation. Use 0 to disable.
""",
    "deblend_mincont": """
Minimum flux density fraction (relative to the original, i.e. unblended, 
island) required for a subisland to be considered
a valid deblended component.
""",
    "structuring_element": """
The 'structuring element' defines island connectivity as in
'4-connectivity' and '8-connectivity' as a Python-style nested list,
e.g. '[[1,1,1], [1,1,1], [1,1,1]]' for 8-connectivity and '[[0,1,0], [1,
1,1], [0,1,0]]' for 4-connectivity. These two are the only
reasonable choices, since the structuring element must be
centrosymmetric.  The structuring element is applied in
scipy.ndimage.label, so check its documentation for some
background on its use.
""",
    "vectorized": """
Measure sources using a vectorized implementation of the
'tweaked moments' method. Compared to Gaussian fitting, this approach is
much faster and more suitable for large numbers of sources. Peak
spectral brightnesses remain biased downward,
but generally show a smaller negative bias than Gaussian fits. The derived
elliptical source axes (major and minor) tend to be biased upward,
typically more so than for Gaussian fits.
""",
    "nr_threads": """
The number of threads used to parallelize Gaussian fits to detected
sources. This integer sets 'max_workers' in
'concurrent.futures.ThreadpoolExecutor'.
Expect speedups when using a free-threading version of Python,
but this integer is probably best set to 1 if you're not.
Note: this does not change numba's 'NUMBA_NUM_THREADS' for the numerous
parallel numba operations in PySE.
""",
    "margin": """
Margin in pixels to ignore near the edges of the image, i.e.
sources within this margin will not be detected.
""",
    "radius": """
Radius in pixels (from image center) considered valid, i.e. sources
beyond this radius will not be detected.
""",
    "back_size_x": """
Subimage size for estimation of background node values (X-
direction). The nodes are centred on the subimages.
""",
    "back_size_y": """
Subimage size for estimation of background node values (Y-
direction). The nodes are centred on the subimages.
""",
    "grid": """
Background subimage size used as fallback for back_size_x and
back_size_y. If both are not set, this implies
back_size_x=backsize_y=grid, i.e. the subimages are squares.
""",
    "eps_ra": """
Calibration uncertainty in right ascension (degrees), see
equation 27a of the NVSS paper.
""",
    "eps_dec": """
Calibration uncertainty in declination (degrees), see equation
27b of the NVSS paper.
""",
    "clean_bias": """
Clean bias to subtract from the peak brightnesses (Jy/beam), see
parapagraph 5.2.5 and equation 34 of the NVSS paper.
""",
    "clean_bias_error": """
1-sigma uncertainty in clean bias (Jy/beam), see parapagraph 5.2.5 and
equation 37 of the NVSS paper.
""",
    "frac_flux_cal_error": """
Intensity-proportional calibration uncertainty, see paragraph 5.2.5 and 
equation 37 of the NVSS paper.
""",
    "alpha_maj1": """
First exponent for scaling errors along the fitted major 
axis, see equation 26 and paragraph 5.2.3 of the NVSS paper and 
equation 41 and paragraph 3 of Condon's (1997) "Errors in Elliptical Gaussian Fits".
""",
    "alpha_maj2": """
Second exponent for scaling errors along the fitted major 
axis, see equation 26 and paragraph 5.2.3 of the NVSS paper and
equation 41 and paragraph 3 of Condon's (1997) "Errors in Elliptical Gaussian Fits".
""",
    "alpha_min1": """
First exponent for scaling errors along the fitted minor 
axis and for scaling errors in the position angle, see equation 26 and 
paragraph 5.2.3 of the NVSS paper and equation 41 and paragraph 3 of 
Condon's (1997) "Errors in Elliptical Gaussian Fits".
""",
    "alpha_min2": """
Second exponent for scaling errors along the fitted minor 
axis and for scaling errors in the position angle, see equation 26 and 
paragraph 5.2.3 of the NVSS paper and equation 41 and paragraph 3 of 
Condon's (1997) "Errors in Elliptical Gaussian Fits".
""",
    "alpha_brightness1": """
First exponent for scaling errors in peak brightness, see
equation 26 and paragraph 5.2.5 of the NVSS paper and equation 41
and paragraph 3 of Condon's (1997) "Errors in Elliptical Gaussian
Fits".
""",
    "alpha_brightness2": """
Second exponent for scaling errors in peak brightness, see
equation 26 and paragraph 5.2.5 of the NVSS paper and equation      
41 and paragraph 3 of Condon's (1997) "Errors in Elliptical Gaussian
Fits".
""",
    "detection_thr": """
Detection threshold as multiple of the background standard
deviation (rms) map, after the background mean values have been
subtracted from the image.
""",
    "analysis_thr": """
Analysis threshold as multiple of the background standard
deviation (rms) map, after the background mean values have been
subtracted from the image. Island pixels above the analysis threshold 
are used for the measurement of the source. The analysis threshold must be 
lower than or equal to the detection threshold.
""",
    "fdr": """
Use False Detection Rate (FDR) algorithm for determining the
detection threshold.
""",
    "alpha": """
FDR alpha value (float, default 0.01) that sets an upper limit
on the fraction of pixels erroneously detected as source pixels,
relative to all source pixels.  This requirement should be met
when averaged over a large ensemble of images, but problems were
encountered with alpha as low as 0.001, see paragraph 3.6 of
Spreeuw's thesis.
""",
    "deblend_nthresh": """
Number of deblending subthresholds; 0 to disable.
""",
    "bmaj": """
Major axis of restoring beam (degrees).
""",
    "bmin": """
Minor axis of restoring beam (degrees).
""",
    "bpa": """
Restoring beam position angle (degrees).
""",
    "force_beam": """
Force source shape to align restoring beam shape (bmaj, bmin,
bpa) for Gaussian fits and vetorized source measurement, i.e. when
vectorized=True.
""",
    "detection_image": """
Path to detection map. PySE will identify sources and the
positions of pixels which comprise them on the detection image,
but then use the corresponding pixels on the target images to
perform measurements. Of course, the detection image and the
target image(s) must have the same pixel dimensions. Note that
only a single detection image may be specified, and the same
pixels are then used on all target images. Note further that this
detection-image option is incompatible with --fdr
""",
    "fixed_posns": """
JSON __list__ of RA, Dec pairs of coordinates to measure
sources at (disables blind extraction and vectorized source
measurements).
""",
    "fixed_posns_file": """
Path to JSON file with RA, Dec pairs of coordinates to measure
sources at (disables blind extraction and vectorized source
measurements).
""",
    "ffbox": """
When fitting to a fixed position, a square “box” of pixels is
chosen around the requested position, and the optimization
procedure allows the source position to vary within that box. The
size of the box may be changed with this option. Note that this
parameter is given in units of the major axis of the beam in
pixels.
""",
    "ew_sys_err": """
Systematic error in east-west direction, see paragraph 5.2.3
of the NVSS paper. Note that this parameter is currently not
applied in PySE, because it should be considered a final step
before entering source parameters in a catalog, i.e. it is simply
returned to allow for systematic positional offset cf. the
NVSS. Therefore, its unit (degrees, arcseconds) is up to the user.
""",
    "ns_sys_err": """
Systematic error in north-south direction, see paragraph 5.2.3
of the NVSS paper. Note that this parameter is currently not
applied in PySE, because it should be considered a final step
before entering source parameters in a catalog, i.e. it is simply
returned to allow for systematic positional offset cf. the
NVSS. Therefore, its unit (degrees, arcseconds) is up to the user.
""",
    "remove_edge_sources": """
When source pixels - with values above the analysis threshold - 
connect with the edge of a map or with masked pixels, do not measure the 
source properties. Consequently, the parameters of this source will not be 
returned. The idea here is that, when source pixels are adjacent to 
edges or masked pixels, pixels needed for a measurement that is symmetrical
relative to the source's barycenter will likely be missing, which will
compromise the measurement.
""",
}
"""Configuration options for image processing and source extraction.
"""


@dataclass(frozen=True)
class ImgConf(_Validate):
    """Configuration that should cover all the specifications for processing the image."""

    interpolate_order: int = 1

    median_filter: int = 0

    mf_threshold: int = 0

    rms_filter: float = 0.001

    deblend_mincont: float = 0.005

    structuring_element: list[list[int]] = field(
        default_factory=lambda: _structuring_element
    )

    vectorized: bool = True

    nr_threads: int | None = None

    margin: int = 0

    radius: float = 0.0

    back_size_x: int | None = None

    back_size_y: int | None = None

    grid: int | None = 64

    eps_ra: float = 0.0

    eps_dec: float = 0.0

    clean_bias: float = 0.0

    clean_bias_error: float = 0.0

    frac_flux_cal_error: float = 0.0

    alpha_maj1: float = 2.5

    alpha_maj2: float = 0.5

    alpha_min1: float = 0.5

    alpha_min2: float = 2.5

    alpha_brightness1: float = 1.5

    alpha_brightness2: float = 1.5

    detection_thr: float = 10.0

    analysis_thr: float = 3.0

    fdr: bool = False

    alpha: float = 1e-2

    deblend_nthresh: int = 0

    bmaj: float | None = None

    bmin: float | None = None

    bpa: float | None = None

    force_beam: bool = False

    detection_image: str | None = None

    fixed_posns: str | None = None

    fixed_posns_file: str | None = None

    ffbox: float = 3.0

    ew_sys_err: float = 0.0

    ns_sys_err: float = 0.0

    remove_edge_sources: bool = True


def _dataclass_field_names(cls):
    return {f.name for f in fields(cls) if not f.name.startswith("_")}


def test_imgconf_help_is_complete():
    field_names = _dataclass_field_names(ImgConf)
    help_names = set(IMGCONF_HELP.keys())

    missing = field_names - help_names

    assert (
        not missing
    ), "IMGCONF_HELP is missing descriptions for: " + ", ".join(
        sorted(missing)
    )


test_imgconf_help_is_complete()

EXPORTSETTINGS_HELP: dict[str, str] = {
    "output_dir": "Directory in which to write the output files.",
    "file_type": """
Output file type (default: csv). As of 20260114 this attribute does not seem 
to be effectuated; csv is the only supported output file type, through the 
'csv' attribute. This attribute should provide for a range of output formats, 
e.g. HDF5 anc CSV and replace the 'csv' attribute.
""",
    "skymodel": "Generate a sky model.",
    "csv": "Generate a CSV text file (e.g., for TopCat).",
    "regions": "Generate DS9 region file(s).",
    "rmsmap": """
Generate map with the root-mean-square (RMS) of the background noise.
""",
    "sigmap": """
Generate a significance map, i.e. the observational data - with mean 
background subtracted - divided by the RMS map.
""",
    "residuals": """
Generate a residuals map, i.e. a map where the Gaussian reconstructions of 
all detected sources have been subtracted from the observational data, with 
mean background subtracted.
""",
    "islands": """
Generate an islands map, i.e. a map with the Gaussian reconstructions of all 
detected sources.
""",
    "pandas_df": """
If True, the measured and derived source parameters will be returned as a 
Pandas DataFrame. If false, they will be returned as a 
`utility.containers.ExtractionResults` object. As of 20260114 not yet 
available for the command-line interface.
""",
    "source_params": """List of source parameters to collect.""",
    "source_params_file": """
List of source parameters to include in a file for storage.
""",
}
"""Configuration options for export of source finding results.
"""


@dataclass(frozen=True)
class ExportSettings(_Validate):
    """Selection of output, related to detected sources and/or intermediate
    image processing products"""

    output_dir: str = "."

    file_type: str = "csv"

    skymodel: bool = False

    csv: bool = False

    regions: bool = False

    rmsmap: bool = False

    sigmap: bool = False

    residuals: bool = False

    islands: bool = False

    pandas_df: bool = True

    source_params: list[str] = field(default_factory=lambda: _source_params)

    source_params_file: list[str] = field(
        default_factory=lambda: _source_params_file
    )


def test_exportsettings_help_is_complete():
    field_names = _dataclass_field_names(ExportSettings)
    help_names = set(EXPORTSETTINGS_HELP.keys())

    missing = field_names - help_names

    assert (
        not missing
    ), "EXPORTSETTINGS_HELP is missing descriptions for: " + ", ".join(
        sorted(missing)
    )


test_exportsettings_help_is_complete()


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
    elif isinstance(val, str):
        if val.strip().lower() == "none":
            return None
        return val
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
