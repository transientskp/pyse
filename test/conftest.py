import os
from pathlib import Path
import warnings

DEFAULT = str((Path(__file__).parent / "data").absolute())
DATAPATH = os.environ.get("TKP_TESTPATH", DEFAULT)

if not os.access(DATAPATH, os.X_OK):
    warnings.warn(f"can't access {DATAPATH}")
