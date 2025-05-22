import subprocess
from pathlib import Path

import pytest

def run_pyse(files, extra_args, out_dir):
    cli_args = ["pyse", *files, "--config_file", "test/data/config.toml"]
    cli_args.extend(extra_args)
    cli_args.extend(["--output_dir", str(out_dir)])

    test = subprocess.Popen(cli_args, stdout=subprocess.PIPE)
    raw_output = test.communicate()[0]
    output = raw_output.decode("utf8").split("\n")

    nr_sources_per_image = []
    for line in output:
        if "Number of sources" in line:
            nr_sources = int(line.split(" = ")[-1].strip())
            nr_sources_per_image.append(nr_sources)

    if len(nr_sources_per_image) != len(files):
        raise ValueError(f"Not all images finished correctly. Found {len(nr_sources_per_image)} sets of sources with {len(files)} input images. ")

    return nr_sources_per_image

def test_pyse_simple(tmpdir):
    files = ["test/data/GRB120422A-120429.fits"]
    extra_args = ["--detection", "6", "--analysis", "5"]
    nr_sources_per_image = run_pyse(files, extra_args, tmpdir)

    for n in nr_sources_per_image:
        assert n == 1

@pytest.mark.parametrize("export_arg", [
    "--skymodel",
    "--csv",
    "--regions",
    "--rmsmap",
    "--sigmap",
    "--residuals",
    "--islands",
])
def test_pyse_export(tmpdir, export_arg):
    files = ["test/data/GRB120422A-120429.fits"]
    extra_args = ["--detection", "6", "--analysis", "5", export_arg]
    nr_sources_per_image = run_pyse(files, extra_args, tmpdir)

    for n in nr_sources_per_image:
        assert n == 1

    # Check CSV
    if export_arg == "--csv":
        assert Path(f"{tmpdir}/GRB120422A-120429.csv").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.csv").exists()

    # Check skymodel
    if export_arg == "--skymodel":
        assert Path(f"{tmpdir}/GRB120422A-120429.skymodel").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.skymodel").exists()

    # Check regions
    if export_arg == "--regions":
        assert Path(f"{tmpdir}/GRB120422A-120429.reg").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.reg").exists()

    # Check rmsmap
    if export_arg == "--rmsmap":
        assert Path(f"{tmpdir}/GRB120422A-120429.rms.fits").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.rms.fits").exists()

    # Check sigmap
    if export_arg == "--sigmap":
        assert Path(f"{tmpdir}/GRB120422A-120429.sig.fits").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.sig.fits").exists()

    # Check residuals
    if export_arg == "--residuals":
        assert Path(f"{tmpdir}/GRB120422A-120429.residuals.fits").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.residuals.fits").exists()

    # Check islands
    if export_arg == "--islands":
        assert Path(f"{tmpdir}/GRB120422A-120429.islands.fits").exists()
    else:
        assert not Path(f"{tmpdir}/GRB120422A-120429.islands.fits").exists()
