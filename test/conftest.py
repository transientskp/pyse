import os
from pathlib import Path
import warnings
import multiprocessing

DEFAULT = str((Path(__file__).parent / "data").absolute())
DATAPATH = os.environ.get("TKP_TESTPATH", DEFAULT)

multiprocessing.set_start_method("forkserver", force=True)

if not os.access(DATAPATH, os.X_OK):
    warnings.warn(f"can't access {DATAPATH}")
