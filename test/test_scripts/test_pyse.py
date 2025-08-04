import subprocess
from pathlib import Path


def run_pyse(files, extra_args, out_dir):
    cli_args = ["pyse", *files, "--config-file", "test/data/config.toml"]
    cli_args.extend(extra_args)
    cli_args.extend(["--output-dir", str(out_dir)])

    test = subprocess.Popen(cli_args, stdout=subprocess.PIPE)
    raw_output = test.communicate()[0]
    output = raw_output.decode("utf8").split("\n")

    nr_sources_per_image = []
    for line in output:
        if "Number of sources" in line:
            nr_sources = int(line.split(" = ")[-1].strip())
            nr_sources_per_image.append(nr_sources)

    if len(nr_sources_per_image) != len(files):
        raise ValueError(
            f"Not all images finished correctly. Found {len(nr_sources_per_image)} sets of sources with {len(files)} input images. "
        )

    return nr_sources_per_image


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
    extra_args = ["--detection", "6", "--analysis", "5", *export_args]
    nr_sources_per_image = run_pyse(files, extra_args, tmpdir)

    for n in nr_sources_per_image:
        assert n == 1

    # Check CSV
    assert Path(f"{tmpdir}/GRB120422A-120429.csv").exists()

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
