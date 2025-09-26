#!/usr/bin/env python
"""
Simple interface to TKP source identification & measurement code.
John Sanders & John Swinbank, 2011.

This is a simplified script for running source finding with a minimal set of
arguments. It does not provide a full configuration interface or access to
all features.

Run as:

  $ pyse file ...

For help with command line options:

  $ pyse --help

See chapters 2 & 3 of Spreeuw, PhD Thesis, University of Amsterdam, 2010,
<http://dare.uva.nl/en/record/340633> for details.
"""
import argparse
import ast
import json
import logging
import math
import numbers
import os.path
import pdb
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path

import astropy.io.fits as pyfits
import numpy

from sourcefinder.accessors import open as open_accessor
from sourcefinder.accessors import sourcefinder_image_from_accessor
from sourcefinder.accessors import writefits as tkp_writefits
from sourcefinder.config import read_conf
from sourcefinder.utils import generate_result_maps


def parse_monitoringlist_positions(
    args, str_name="monitor_coords", list_name="monitor_list"
):
    """Load a list of monitoring list (RA, Dec) tuples from command
    line arguments.

    This function processes the flags `--monitor-coords` and `--monitor-list`.
    It does not handle units, which should be matched against the requirements
    of the consuming code.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments object.
    str_name : str, default: "monitor_coords"
        The name of the argument containing the JSON string of coordinates.
    list_name : str, default: "monitor_list"
        The name of the argument containing the file path to a JSON file with
        coordinates.

    Returns
    -------
    list[tuple[float, float]]
        A list of (RA, Dec) tuples parsed from the input arguments.

    Raises
    ------
    json.JSONDecodeError
        If the JSON string or file content cannot be parsed.

    """
    monitor_coords = []
    if hasattr(args, str_name) and getattr(args, str_name):
        try:
            monitor_coords.extend(json.loads(getattr(args, str_name)))
        except json.JSONDecodeError:
            logging.error(
                "Could not parse monitor-coords from command line:"
                "string passed was:\n%s" % (getattr(args, str_name),)
            )
            raise
    if hasattr(args, list_name) and getattr(args, list_name):
        try:
            with open(getattr(args, list_name)) as file:
                mon_list = json.load(file)
            monitor_coords.extend(mon_list)
        except json.JSONDecodeError:
            logging.error(
                "Could not parse monitor-coords from file: "
                + getattr(args, list_name)
            )
            raise
    return monitor_coords


def parse_none(value):
    if isinstance(value, str):
        if value.lower() == "none" or value.lower() == "null":
            return None
    return value


def construct_argument_parser():
    parser = argparse.ArgumentParser(
        description="PySE image configuration options. These can override the values specified in the TOML config file."
    )

    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config-file",
        help="""
        TOML file containing default input arguments to PySE.
        This is especially convenient when swapping between configurations for the same project.
    """,
    )
    general_group.add_argument(
        "--pdb",
        action="store_true",
        help="""
        Enter debug mode when the application crashes. Meant to be used for more comprehensive debugging.
    """,
    )

    image_group = parser.add_argument_group("Image parameters")

    image_group.add_argument(
        "--interpolate-order",
        type=int,
        help="Order of interpolation to use (e.g. 1 for linear).",
    )

    image_group.add_argument(
        "--median-filter",
        type=int,
        help="Size of the median filter to apply to the image. Use 0 to disable.",
    )

    image_group.add_argument(
        "--mf-threshold",
        type=float,
        help="Threshold used with the median filter. Sources below this value are discarded.",
    )

    image_group.add_argument(
        "--rms-filter",
        type=float,
        help="Minimum RMS value to use as filter for the image noise.",
    )

    image_group.add_argument(
        "--deblend-mincont",
        type=float,
        help="Minimum contrast for deblending islands into separate sources (e.g. 0.005).",
    )

    image_group.add_argument(
        "--structuring-element",
        type=ast.literal_eval,
        help="""
        Structuring element for morphological operations, provided as a Python-style nested list (e.g. '[[1,1,1],[1,1,1],[1,1,1]]').
        This is used for defining the connectivity in source detection.
        """,
    )

    image_group.add_argument(
        "--vectorized",
        action="store_true",
        help="Use vectorized operations where applicable.",
    )

    image_group.add_argument(
        "--nr-threads",
        type=int,
        help="""The number of threads used to parallelize Gaussian fits to detected
        sources.
        Note: this does not change numba's 'num threads' for parallel numba operations.
        """,
    )

    image_group.add_argument(
        "--margin",
        type=int,
        help="Margin in pixels to ignore around the edge of the image.",
    )

    image_group.add_argument(
        "--radius",
        type=float,
        help="Radius in pixels around sources to include in analysis.",
    )

    image_group.add_argument(
        "--back-size-x",
        type=int,
        help="Size of the background subimage in the X direction.",
    )

    image_group.add_argument(
        "--back-size-y",
        type=int,
        help="Size of the background subimage in the Y direction.",
    )

    image_group.add_argument(
        "--eps-ra", type=float, help="RA matching tolerance in arcseconds."
    )

    image_group.add_argument(
        "--eps-dec", type=float, help="Dec matching tolerance in arcseconds."
    )
    image_group.add_argument(
        "--detection-thr", type=float, help="Detection threshold"
    )
    image_group.add_argument(
        "--analysis-thr", type=float, help="Analysis threshold"
    )
    image_group.add_argument(
        "--fdr", action="store_true", help="Use False Detection Rate algorithm"
    )
    image_group.add_argument("--alpha", type=float, help="FDR Alpha")
    image_group.add_argument(
        "--deblend_nthresh",
        type=int,
        help="Number of deblending subthresholds; 0 to disable",
    )
    image_group.add_argument(
        "--grid", type=int, help="Background grid segment size"
    )
    image_group.add_argument(
        "--bmaj", type=float, help="Set beam: Major axis of beam (deg)"
    )
    image_group.add_argument(
        "--bmin", type=float, help="Set beam: Minor axis of beam (deg)"
    )
    image_group.add_argument(
        "--bpa", type=float, help="Set beam: Beam position angle (deg)"
    )
    image_group.add_argument(
        "--force-beam",
        action="store_true",
        help="Force fit axis lengths to beam size",
    )
    image_group.add_argument(
        "--detection-image", type=str, help="Find islands on different image"
    )
    image_group.add_argument(
        "--fixed-posns",
        help="List of position coordinates to "
        "force-fit (decimal degrees, JSON, e.g [[123.4,56.7],[359.9,89.9]]) "
        "(Will not perform blind extraction in this mode)",
    )
    image_group.add_argument(
        "--fixed-posns-file",
        help="Path to file containing a list of positions to force-fit "
        "(Will not perform blind extraction in this mode)",
    )
    image_group.add_argument(
        "--ffbox",
        type=float,
        help="Forced fitting positional box size as a multiple of beam width.",
    )
    image_group.add_argument(
        "--ew-sys-err",
        type=float,
        help="Systematic error in east-west direction",
    )
    image_group.add_argument(
        "--ns-sys-err",
        type=float,
        help="Systematic error in north-south direction",
    )

    # Arguments relating to output:
    export_group = parser.add_argument_group("Export parameters")
    export_group.add_argument(
        "--output-dir",
        help="""
        The directory in which to store the output files.
    """,
    )
    export_group.add_argument(
        "--skymodel", action="store_true", help="Generate sky model"
    )
    export_group.add_argument(
        "--csv",
        action="store_true",
        help="Generate csv text file for use in programs such as TopCat",
    )
    export_group.add_argument(
        "--regions", action="store_true", help="Generate DS9 region file(s)"
    )
    export_group.add_argument(
        "--rmsmap", action="store_true", help="Generate RMS map"
    )
    export_group.add_argument(
        "--sigmap", action="store_true", help="Generate significance map"
    )
    export_group.add_argument(
        "--residuals", action="store_true", help="Generate residual maps"
    )
    export_group.add_argument(
        "--islands", action="store_true", help="Generate island maps"
    )

    # Finally, positional arguments- the file list:
    parser.add_argument("files", nargs="+", help="Image files for processing")
    return parser


def regions(sourcelist):
    """
    Return a string containing a DS9-compatible region file describing all the
    sources in sourcelist.
    """
    output = StringIO()
    print("# Region file format: DS9 version 4.1", file=output)
    print(
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        file=output,
    )
    print("image", file=output)
    for source in sourcelist:
        # NB, here we convert from internal 0-origin indexing to DS9 1-origin
        # indexing
        print(
            "ellipse(%f, %f, %f, %f, %f)"
            % (
                source.x.value + 1.0,
                source.y.value + 1.0,
                source.smaj.value * 2,
                source.smin.value * 2,
                math.degrees(source.theta) + 90,
            ),
            file=output,
        )
    return output.getvalue()


def skymodel(sourcelist, ref_freq=73800000):
    """
    Return a string containing a skymodel from the extracted sources for use
    in self-calibration.
    """
    output = StringIO()
    print(
        "#(Name, Type, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis, Orientation, "
        "ReferenceFrequency='60e6', SpectralIndex='[0.0]') = format",
        file=output,
    )
    for source in sourcelist:
        print(
            "%s, GAUSSIAN, %s, %s, %f, 0, 0, 0, %f, %f, %f, %f, [0]"
            % (
                "ra:%fdec:%f" % (source.ra, source.dec),
                "%fdeg" % (source.ra,),
                "%fdeg" % (source.dec,),
                source.flux,
                source.smaj_asec,
                source.smin_asec,
                source.theta_celes,
                ref_freq,
            ),
            file=output,
        )
    return output.getvalue()


def csv(sourcelist, conf):
    """
    Return a string containing a csv from the extracted sources.
    """
    output = StringIO()
    print(", ".join(conf.export.source_params_file), file=output)
    for source in sourcelist:
        values = source.serialize(conf=conf)
        print(", ".join(f"{float(v):.6f}" for v in values), file=output)
    return output.getvalue()


def summary(filename, sourcelist):
    """
    Return a string containing a human-readable summary of all sources in
    sourcelist.
    """
    output = StringIO()
    print("** %s **\n" % (filename), file=output)
    for source in sourcelist:
        print(
            "RA: %s, dec: %s" % (str(source.ra), str(source.dec)), file=output
        )
        print(
            "Error radius (arcsec): %s" % (str(source.error_radius)),
            file=output,
        )
        print(
            "Semi-major axis (arcsec): %s" % (str(source.smaj_asec)),
            file=output,
        )
        print(
            "Semi-minor axis (arcsec): %s" % (str(source.smin_asec)),
            file=output,
        )
        print("Position angle: %s" % (str(source.theta_celes)), file=output)
        print("Flux: %s" % (str(source.flux)), file=output)
        print("Peak: %s\n" % (str(source.peak)), file=output)
    return output.getvalue()


def handle_args(args=None):
    """
    Parses command line options & arguments using OptionParser.
    Options & default values for the script are defined herein.
    """
    parser = construct_argument_parser()
    arguments = parser.parse_args()
    unstructured_args = vars(arguments)

    # Extract file paths, which are only to be supplied via command line,
    # not config
    files = unstructured_args.pop("files")

    # Automatically start the debugger on an unhandled exception if
    # specified
    debug_on_error = unstructured_args.pop("pdb")
    if debug_on_error:

        def excepthook(type, value, traceback):
            pdb.post_mortem(traceback)

        sys.excepthook = excepthook

    # Merge the CLI arguments with the config file parameters
    config_file = unstructured_args.pop("config_file")
    conf = read_conf(config_file)

    # Structure the arguments based on their group
    cli_args: dict = dict()
    for argument in parser._actions:
        section_name = argument.container.title
        if section_name not in cli_args:
            cli_args[section_name] = dict()
        arg_name = argument.dest
        if arg_name in unstructured_args:
            # Default arguments like '--help' and arguments popped from
            # unstructured args like "--pdb" should be skipped
            cli_value = unstructured_args[arg_name]
            if cli_value is not None:
                # For argparse arguments where no value is provided in the
                # command line we get a None value, ignore these
                cli_args[section_name][arg_name] = cli_value

    # Note: Dataclass replace is not recursive for stacked dataclasses.
    #       Replace one by one to avoid arguments being reset to their
    #       defaults.
    conf_image = replace(conf.image, **cli_args["Image parameters"])
    conf_export = replace(conf.export, **cli_args["Export parameters"])
    conf = replace(conf, image=conf_image, export=conf_export)

    # Overwrite 'fixed_coords' with a parsed list of coords
    # collated from both command line and file.
    fixed_coords = parse_monitoringlist_positions(
        conf.image, str_name="fixed_posns", list_name="fixed_posns_file"
    )
    # Quick & dirty check that the position list looks plausible
    if fixed_coords:
        mlist = numpy.array(fixed_coords)
        if not (len(mlist.shape) == 2 and mlist.shape[1] == 2):
            parser.error("Positions for forced-fitting must be [RA,dec] pairs")

    # We have four potential modes, of which we choose only one to run:
    #
    # 1. Blind sourcefinding
    #  1.1 Thresholding, no detection image (no extra cmd line options)
    #  1.2 Thresholding, detection image (--detection-image)
    #  1.3 FDR (--fdr)
    #
    # 2. Fit to fixed points (--fixed-coords and/or --fixed-list)

    if fixed_coords:
        if conf.image.fdr:
            parser.error("--fdr not supported with fixed positions")
        elif conf.image.detection_image:
            parser.error(
                "--detection-image not supported with fixed positions"
            )
        mode = "fixed"  # mode 2 above
    elif conf.image.fdr:
        if conf.image.detection_image:
            parser.error("--detection-image not supported with --fdr")
        mode = "fdr"  # mode 1.3 above
    elif conf.image.detection_image:
        mode = "detimage"  # mode 1.2 above
    else:
        mode = "threshold"  # mode 1.1 above

    return conf, mode, files


def writefits(filename, data, header={}):
    try:
        os.unlink(filename)
    except OSError:
        # Thrown if file didn't exist
        pass
    tkp_writefits(data, filename, header)


def get_detection_labels(filename, det, anl, beam, configuration, plane=0):
    print("Detecting islands in %s" % (filename,))
    print("Thresholding with det = %f sigma, analysis = %f sigma" % (det, anl))
    ff = open_accessor(filename, beam=beam, plane=plane)
    imagedata = sourcefinder_image_from_accessor(ff, conf=configuration)
    labels, labelled_data, *_ = imagedata.label_islands(
        det * imagedata.rmsmap, anl * imagedata.rmsmap
    )
    return labels, labelled_data


def get_beam(bmaj, bmin, bpa):
    if (
        isinstance(bmaj, numbers.Real)
        and isinstance(bmin, numbers.Real)
        and isinstance(bpa, numbers.Real)
    ):
        return (float(bmaj), float(bmin), float(bpa))
    if bmaj or bmin or bpa:
        print("WARNING: partial beam specification ignored")
    return None


def bailout(reason):
    # Exit with error
    print("ERROR: %s" % (reason))
    sys.exit(1)


def run_sourcefinder(files, conf, mode):
    """
    Iterate over the list of files, running a sourcefinding step on each in
    turn. If specified, a DS9-compatible region file and/or a FITS file
    showing the residuals after Gaussian fitting are dumped for each file.
    A string containing a human readable list of sources is returned.
    """
    output = StringIO()

    beam = get_beam(conf.image.bmaj, conf.image.bmin, conf.image.bpa)

    if mode == "detimage":
        labels, labelled_data = get_detection_labels(
            conf.image.detection_image,
            conf.image.detection_thr,
            conf.image.analysis_thr,
            beam,
            conf,
        )
    else:
        labels, labelled_data = [], None

    for counter, filename in enumerate(files):
        print(
            "Processing %s (file %d of %d)."
            % (filename, counter + 1, len(files))
        )
        imagename = os.path.splitext(os.path.basename(filename))[0]
        ff = open_accessor(filename, beam=beam, plane=0)
        imagedata = sourcefinder_image_from_accessor(ff, conf=conf)

        if mode == "fixed":
            # FIXME: conf.image.fixed_coords does not exist
            sr = imagedata.fit_fixed_positions(
                conf.image.fixed_coords,
                conf.image.ffbox * max(imagedata.beam[0:2]),
            )

        else:
            if mode == "fdr":
                print(
                    "Using False Detection Rate algorithm with alpha = %f"
                    % (conf.image.alpha,)
                )
                sr = imagedata.fd_extract(
                    alpha=conf.image.alpha,
                )
            else:
                if labelled_data is None:
                    print(
                        "Thresholding with det = %f sigma, analysis = %f sigma"
                        % (conf.image.detection_thr, conf.image.analysis_thr)
                    )

                sr = imagedata.extract(
                    labelled_data=labelled_data,
                    labels=labels,
                )

        export_dir = Path(conf.export.output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        if conf.export.regions:
            regionfile = export_dir / (imagename + ".reg")
            regionfile = open(regionfile, "w")
            regionfile.write(regions(sr))
            regionfile.close()
        # This applies a slower method for computing the Gaussian islands and
        # residuals than the methods from the extract module - since these
        # methods are called in parallel, from image.py, or vectorized - and
        # extends to pixel positions corresponding to values below the analysis
        # threshold, i.e. well into the background noise. Some users may want to
        # accept the extra compute time to be able to compare the residuals with
        # the background noise.
        # if conf.export.residuals or conf.export.islands:
        #     gaussian_map, residual_map = generate_result_maps(imagedata.data,
        #                                                       sr)
        if conf.export.residuals:
            residualfile = export_dir / (imagename + ".residuals.fits")
            writefits(
                residualfile,
                imagedata.Gaussian_residuals,
                pyfits.getheader(filename),
            )
        if conf.export.islands:
            islandfile = export_dir / (imagename + ".islands.fits")
            writefits(
                islandfile,
                imagedata.Gaussian_islands,
                pyfits.getheader(filename),
            )
        if conf.export.rmsmap:
            rmsfile = export_dir / (imagename + ".rms.fits")
            writefits(
                rmsfile,
                numpy.array(imagedata.rmsmap),
                pyfits.getheader(filename),
            )
        if conf.export.sigmap:
            sigfile = export_dir / (imagename + ".sig.fits")
            writefits(
                sigfile,
                numpy.array(imagedata.data_bgsubbed / imagedata.rmsmap),
                pyfits.getheader(filename),
            )
        if conf.export.skymodel:
            with open(
                export_dir / (imagename + ".skymodel"), "w"
            ) as skymodelfile:
                if ff.freq_eff:
                    skymodelfile.write(skymodel(sr, ff.freq_eff))
                else:
                    print(
                        "WARNING: Using default reference frequency for %s"
                        % (skymodelfile.name,)
                    )
                    skymodelfile.write(skymodel(sr))
        if conf.export.csv:
            with open(export_dir / (imagename + ".csv"), "w") as csvfile:
                csvfile.write(csv(sr, conf))
                print(summary(filename, sr), end=" ", file=output)

    return output.getvalue()


def main():
    logging.basicConfig()
    conf, mode, files = handle_args()
    print(run_sourcefinder(files, conf, mode), end=" ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
