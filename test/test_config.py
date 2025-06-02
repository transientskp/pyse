from types import NoneType

import pytest

from sourcefinder.config import assert_t
from sourcefinder.config import Conf
from sourcefinder.config import ExportSettings
from sourcefinder.config import ImgConf
from sourcefinder.config import read_conf
from sourcefinder.config import validate_nested
from sourcefinder.config import validate_types

from .conftest import DATAPATH

_image = {"rms_filter": 0.1, "structuring_element": [[2] * 3] * 3}
_export = {"file_type": "hdf5", "source_params": ["foo", "bar"]}

@pytest.mark.parametrize(
    "conf_t, conf",
    [
        (ImgConf, _image),
        (ExportSettings, _export),
        (Conf, {"image": _image, "export": _export}),
    ],
)
def test_configs(conf_t, conf):
    assert conf_t(**conf)


@pytest.mark.parametrize("path", [f"{DATAPATH}/config.toml"])
def test_read_conf(path):
    conf = read_conf(path)
    assert conf


@pytest.mark.parametrize(
    "key, val, types",
    [
        ("a", 42, (int,)),
        ("a", 42, (NoneType, int)),  # uniontype
        ("a", 3.14, (float,)),
        ("a", "word", (str,)),
        ("a", 42, (float,)),  # compatible
    ],
)
def test_assert_t(key, val, types):
    assert_t(key, val, *types)


@pytest.mark.parametrize(
    "key, val, types",
    [
        ("floatnotint", 3.14, (int,)),
        ("strnotint", "word", (int,)),
        ("intnoncompat", 42, (str,)),
        ("notinunion", "word", (int, NoneType)),
    ],
)
def test_assert_t_err(key, val, types):
    msg_re = rf"{key}:.+"
    if len(types) > 1:
        msg_re += rf"{{{', '.join(map(str, types))}}}"
    else:
        msg_re += rf"{types[0]}"
    with pytest.raises(AssertionError, match=msg_re):
        assert_t(key, val, *types)


@pytest.mark.parametrize(
    "key, value, origin_t, args",
    [
        ("listany", ["a", 1, True], list, ()),
        ("liststr", list("abc"), list, (str,)),
        ("dictionary", {"a": 1, "b": 2}, dict, (str, int)),
        ("dictionarymixed", {"a": 3.14, "b": 2}, dict, (str, float)),  # compatible
    ],
)
def test_nested(key, value, origin_t, args):
    validate_nested(key, value, origin_t, args)


@pytest.mark.parametrize(
    "key, value, origin_t, args, idx",
    [
        ("liststr", ["a", 1], list, (str,), "[1]"),
        ("dictionary", {"a": 1, "b": "2"}, dict, (str, int), "['b']"),
    ],
)
def test_nested_err(key, value, origin_t, args, idx):
    with pytest.raises(AssertionError) as exc_info:
        validate_nested(key, value, origin_t, args)
    assert f"{key}{idx}" in exc_info.value.args[0]


@pytest.mark.parametrize(
    "key,value,origin_t,args", [("tuple", (1, 2), tuple, (int, int))]
)
def test_nested_warn(key, value, origin_t, args):
    with pytest.warns(UserWarning) as records:
        validate_nested(key, value, origin_t, args)
    assert key in records[0].message.args[0]
    assert f"{origin_t[args]}" in records[0].message.args[0]


@pytest.mark.parametrize(
    "key, value, type_",
    [
        ("listany", ["a", 1, True], list),
        ("liststr", list("abc"), list[str,]),
        ("dictionary", {"a": 1, "b": 2}, dict[str, int]),
        ("dictionarymixed", {"a": 3.14, "b": 2}, dict[str, float]),  # compatible
        ("union", 42, (NoneType | int)),  # uniontype
    ],
)
def test_validate_types(key, value, type_):
    validate_types(key, value, type_)
