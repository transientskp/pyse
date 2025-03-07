from dataclasses import asdict
from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import field
from dataclasses import is_dataclass
from pathlib import Path
import tomllib
from types import UnionType
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Mapping
from typing import Container
from typing import Type
from typing import TypeVar

T = TypeVar("T")


def unguarded_is_dataclass(_type: Type[T], /) -> bool:
    """Remove :ref:`TypeGuard` from is_dataclass.

    see: https://github.com/python/mypy/issues/14941
    """
    return is_dataclass(_type)


def validate_nested(value, type_):
    origin_t = get_origin(type_)
    args = get_args(origin_t)
    if issubclass(origin_t, list):
        assert isinstance(value, list), f"type({value}) != list"
        if args:
            for i in value:
                validate_nested(i, args[0])
    elif issubclass(origin_t, tuple):
        assert isinstance(value, tuple), f"type({value}) != tuple"
        for i, t in zip(value, args):
            assert isinstance(i, t), f"type({i}) != {t}"
    elif issubclass(origin_t, dict):
        assert isinstance(value, dict)
        for i in value.values():
            assert isinstance(i, args[1]), f"type({i}) != {args[1]}"
    else:
        pass


@dataclass(frozen=True)
class _Validate:
    def __post_init__(self):
        for (key, type_), val in zip(get_type_hints(self).items(), astuple(self)):
            match get_origin(type_):
                case type() as origin_t:
                    if issubclass(origin_t, Container):
                        validate_nested(val, type_)
                    elif issubclass(origin_t, UnionType):
                        args = get_args(type_)
                        assert isinstance(val, args), f"type({val}) != {args}"
                    else:
                        pass
                case None:
                    assert isinstance(val, type_), f"type({val}) != {type_}"


_mutable_defaults = {
    "structuring_element": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    "source_params": [
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
    ],
}


@dataclass(frozen=True)
class ImgConf(_Validate):
    interpolate_order: int = 1
    median_filter: int = 0
    mf_threshold: int = 0
    rms_filter: float = 0.001
    deblend_mincont: float = 0.005
    structuring_element: list[list[int]] = field(
        default_factory=lambda: _mutable_defaults["structuring_element"]
    )
    vectorized: bool = False
    sep: int | None = None
    margin: float | None = None
    radius: float | None = None
    back_size_x: float | None = None
    back_size_y: float | None = None
    residuals: float | None = None
    islands: int | None = None
    eps_ra: float | None = None
    eps_dec: float | None = None


@dataclass(frozen=True)
class ExportSettings(_Validate):
    file_type: str = "csv"
    source_params: list[str] = field(
        default_factory=lambda: _mutable_defaults["source_params"]
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
