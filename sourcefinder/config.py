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
            assert_t(f"{key}[{k!r}]", v, args[1])
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
    "dec",
    "peak",
    "flux",
    "sig",
    "smaj_asec",
    "smin_asec",
    "theta_celes",
    "ew_sys_err",
    "ns_sys_err",
    "error_radius",
    "gaussian",
    "chisq",
    "reduced_chisq",
]


@dataclass(frozen=True)
class ImgConf(_Validate):
    interpolate_order: int = 1
    median_filter: int = 0
    mf_threshold: int = 0
    rms_filter: float = 0.001
    deblend_mincont: float = 0.005
    structuring_element: list[list[int]] = field(
        default_factory=lambda: _structuring_element
    )
    vectorized: bool = False
    margin: int = 0
    radius: float = 0.0
    back_size_x: int = 32
    back_size_y: int = 32
    residuals: bool = False
    islands: bool = False
    eps_ra: float = 0.0
    eps_dec: float = 0.0


@dataclass(frozen=True)
class ExportSettings(_Validate):
    file_type: str = "csv"
    source_params: list[str] = field(default_factory=lambda: _source_params)


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


def read_conf(path: str | Path):
    data = tomllib.loads(Path(path).read_text())
    conf = data.get("tool", {}).get("pyse", {})
    if not conf:
        match data:
            case {"tool": {"pyse": dict()}, **rest}:
                raise KeyError("tool.pyse: empty section in config file")
            case {"tool": dict(), **rest}:
                raise KeyError("tool.pyse: section for PySE missing in config file")
            case _:
                raise KeyError("tool: top-level section missing in config file")
    return Conf(**conf)
