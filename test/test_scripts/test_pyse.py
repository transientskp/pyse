import subprocess
from pathlib import Path
import pandas as pd


def run_pyse(files, extra_args, out_dir):
    cli_args = ["pyse", *files, "--config-file", "test/data/config.toml"]
    cli_args.extend(extra_args)
    cli_args.extend(["--output-dir", str(out_dir)])

    test = subprocess.Popen(cli_args, stdout=subprocess.PIPE)
    raw_output = test.communicate()[0]
    output = raw_output.decode("utf8").split("\n")
    return output

def test_pyse_export(tmpdir):
    files = ["test/data/GRB120422A-120429.fits"]
    export_args = [
        "--skymodel",
        "--csv",
        "--regions",
        "--rmsmap",
        "--sigmap",
        "--residuals",
        "--islands",
    ]
    extra_args = ["--detection-thr", "6", "--analysis-thr", "5", *export_args]
    output = run_pyse(files, extra_args, tmpdir)

    # Check CSV
    assert Path(f"{tmpdir}/GRB120422A-120429.csv").exists()
    df = pd.read_csv(f"{tmpdir}/GRB120422A-120429.csv")
    assert len(df) == 1

    # Check skymodel
    assert Path(f"{tmpdir}/GRB120422A-120429.skymodel").exists()

    # Check regions
    assert Path(f"{tmpdir}/GRB120422A-120429.reg").exists()

    # Check rmsmap
    assert Path(f"{tmpdir}/GRB120422A-120429.rms.fits").exists()

    # Check sigmap
    assert Path(f"{tmpdir}/GRB120422A-120429.sig.fits").exists()

    # Check residuals
    assert Path(f"{tmpdir}/GRB120422A-120429.residuals.fits").exists()

    # Check islands
    assert Path(f"{tmpdir}/GRB120422A-120429.islands.fits").exists()

def test_pyse_fixed_posns(tmpdir):
    files = ["test/data/GRB120422A-120429.fits"]
    export_args = [
        "--csv",
    ]
    extra_args = ["--detection-thr", "6", "--analysis-thr", "5", "--fixed-posns", "[[136.896, 14.0222]]", *export_args]
    nr_sources_per_image = run_pyse(files, extra_args, tmpdir)

    # Check CSV
    assert Path(f"{tmpdir}/GRB120422A-120429.csv").exists()
    df = pd.read_csv(f"{tmpdir}/GRB120422A-120429.csv")
    assert len(df) == 1
