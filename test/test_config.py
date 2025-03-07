from sourcefinder.config import Conf
from sourcefinder.config import ImgConf
from sourcefinder.config import ExportSettings
from sourcefinder.config import read_conf

import pytest

_image = {"rms_filter": 0.1, "structuring_element": [[2]*3] * 3}
_export = {"file_type": "hdf5", "source_params": ["foo", "bar"]}
@pytest.mark.parametrize("conf_t, conf", [(ImgConf, _image), (ExportSettings, _export), (Conf, {"image": _image, "export": _export})])
def test_configs(conf_t, conf):
    assert conf_t(**conf)
