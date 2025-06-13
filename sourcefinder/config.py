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
_source_params = [
    "ra",
    "ra_err",
    "dec",
    "dec_err",
    "smaj_asec",
    "smaj_asec_err",
    "smin_asec",
    "smin_asec_err",
    "theta_celes",
    "theta_celes_err",
    "flux",
    "flux_err",
    "peak",
    "peak_err",
    "x",
    "y",
    "sig",
    "reduced_chisq"
]



@dataclass(frozen=True)
class ImgConf(_Validate):
    """Configuration that should cover all the specifications for processing the image."""

    interpolate_order: int = 1  # Order of interpolation to use (e.g. 1 for linear)
    median_filter: int = 0      # Size of the median filter to apply to background and RMS grids prior to interpolating. Use 0 to disable.
    mf_threshold: int = 0       # Threshold used with the median filter if median_filter is non-zero. Sources below this are discarded.
    rms_filter: float = 0.001   # Minimum RMS value to use as filter for the image noise.
    deblend_mincont: float = 0.005  # Minimum contrast for deblending islands into separate sources.

    # The "structuring element" defines island connectivity as in
    # "4-connectivity" and "8-connectivity". These two are the only reasonable
    # choices, since the structuring element must be centrosymmetric.
    # The structuring element is applied in scipy.ndimage.label, so check its
    # documentation for some background on its use.
    structuring_element: list[list[int]] = field(
        default_factory=lambda: _structuring_element
    )
    vectorized: bool = False               # Use vectorized operations where applicable (faster, but skips Gaussian fitting).
    allow_multiprocessing: bool = True     # Allow multiprocessing for Gaussian fitting in parallel.
    margin: int = 0                        # Margin in pixels to ignore around the edge of the image.
    radius: float = 0.0                    # Radius in pixels around sources to include in analysis.
    back_size_x: int | None = 32           # Background estimation box size (X direction).
    back_size_y: int | None = 32           # Background estimation box size (Y direction).
    eps_ra: float = 0.0                    # RA matching tolerance in arcseconds.
    eps_dec: float = 0.0                   # Dec matching tolerance in arcseconds.
    detection: float = 10.0                # Detection threshold.
    analysis: float = 3.0                  # Analysis threshold.
    fdr: bool = False                      # Use False Detection Rate (FDR) algorithm.
    alpha: float = 1e-2                    # FDR alpha value (significance level).
    deblend_thresholds: int = 0            # Number of deblending subthresholds; 0 to disable.
    grid: int | None = None                # Background grid segment size used as fallback for back-size-x and back-size-y.
    bmaj: float | None = None              # Set beam: Major axis of beam (degrees).
    bmin: float | None = None              # Set beam: Minor axis of beam (degrees).
    bpa: float | None = None               # Set beam: Beam position angle (degrees).
    force_beam: bool = False               # Force fit axis lengths to beam size.
    detection_image: str | None = None     # Path to image used for detection (can be different from analysis image).
    fixed_posns: str | None = None         # JSON list of coordinates to force-fit (disables blind extraction).
    fixed_posns_file: str | None = None    # Path to file with coordinates to force-fit (disables blind extraction).
    ffbox: float = 3.0                     # Forced fitting box size as a multiple of beam width.
    ew_sys_err: float = 0.                 # Systematic error in east-west direction
    ns_sys_err: float = 0.                 # Systematic error in north-south direction


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
