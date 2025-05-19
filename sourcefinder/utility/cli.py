import json
import logging
import ast
import argparse
import sys
import pdb
from pathlib import Path

from sourcefinder.config import _read_conf_as_dict, Conf

def parse_monitoringlist_positions(args, str_name="monitor_coords",
                                   list_name="monitor_list"):
    """Loads a list of monitoringlist (RA,Dec) tuples from cmd line args object.

    Processes the flags "--monitor-coords" and "--monitor-list"
    NB This is just a dumb function that does not care about units,
    those should be matched against whatever uses the resulting values...
    """
    monitor_coords = []
    if hasattr(args, str_name) and getattr(args, str_name):
        try:
            monitor_coords.extend(json.loads(getattr(args, str_name)))
        except ValueError:
            logging.error("Could not parse monitor-coords from command line:"
                          "string passed was:\n%s" % (getattr(args, str_name),)
                          )
            raise
    if hasattr(args, list_name) and getattr(args, list_name):
        try:
            mon_list = json.load(open(getattr(args, list_name)))
            monitor_coords.extend(mon_list)
        except ValueError:
            logging.error("Could not parse monitor-coords from file: "
                          + getattr(args, list_name))
            raise
    return monitor_coords

def parse_none(value):
    if isinstance(value, str):
        if value.lower() == "none" or value.lower() == "null":
            return None
    return value

def read_and_update_config_file(config_file: str | Path, overwrite_data: dict):
    """Read the config file and overwrite the parameters with those found on the command line.

    Parameters
    ----------
    config_file: :class:`str`
        The path to the configuration file. Must be a .toml format.
    params: :class:`dict`
        The parameters as parsed by argparse from the command line.

    Returns
    -------
    :class:`Conf`
        The PySE configuration based on the config file and CLI arguments
    """
    config_data = _read_conf_as_dict(config_file)
    combined_data = dict()
    def overwrite_params_from_nested_dict(to_update: dict, update_with: dict):
        # Recursively copy the fields in 'update_with' and replace the fields
        # if found in 'overwrite_params'.
        for key, value in update_with.items():
            if isinstance(value, dict):
                to_update[key] = dict()
                overwrite_params_from_nested_dict(to_update[key], value)
            else:
                to_update[key] = parse_none(overwrite_data.get(key)) or parse_none(value)

    overwrite_params_from_nested_dict(combined_data, config_data)
    return Conf(**combined_data)

def construct_argument_parser():
    parser = argparse.ArgumentParser(
        description="PySE image configuration options. These can override the values specified in the TOML config file."
    )

    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config_file",
        default="pyse_config.toml",
        help="""
        TOML file containing default input arguments to PySE.
        Default file name: pyse_config.toml
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
        "--interpolate_order",
        type=int,
        help="Order of interpolation to use (e.g. 1 for linear)."
    )

    image_group.add_argument(
        "--median_filter",
        type=int,
        help="Size of the median filter to apply to the image. Use 0 to disable."
    )

    image_group.add_argument(
        "--mf_threshold",
        type=float,
        help="Threshold used with the median filter. Sources below this value are discarded."
    )

    image_group.add_argument(
        "--rms_filter",
        type=float,
        help="Minimum RMS value to use as filter for the image noise."
    )

    image_group.add_argument(
        "--deblend_mincont",
        type=float,
        help="Minimum contrast for deblending islands into separate sources (e.g. 0.005)."
    )

    image_group.add_argument(
        "--structuring_element",
        type=ast.literal_eval,
        help="""
        Structuring element for morphological operations, provided as a Python-style nested list (e.g. '[[1,1,1],[1,1,1],[1,1,1]]').
        This is used for defining the connectivity in source detection.
        """
    )

    image_group.add_argument(
        "--vectorized",
        action="store_true",
        help="Use vectorized operations where applicable."
    )

    image_group.add_argument(
        "--allow_multiprocessing",
        action="store_true",
        help="Allow use of multiprocessing to fit gaussians to islands in parallel."
    )

    image_group.add_argument(
        "--margin",
        type=int,
        help="Margin in pixels to ignore around the edge of the image."
    )

    image_group.add_argument(
        "--radius",
        type=float,
        help="Radius in pixels around sources to include in analysis."
    )

    image_group.add_argument(
        "--back_size_x",
        type=int,
        help="Size of the background estimation box in the X direction."
    )

    image_group.add_argument(
        "--back_size_y",
        type=int,
        help="Size of the background estimation box in the Y direction."
    )

    image_group.add_argument(
        "--eps_ra",
        type=float,
        help="RA matching tolerance in arcseconds."
    )

    image_group.add_argument(
        "--eps_dec",
        type=float,
        help="Dec matching tolerance in arcseconds."
    )
    image_group.add_argument("--detection", default=10, type=float,
                            help="Detection threshold")
    image_group.add_argument("--analysis", default=3, type=float,
                            help="Analysis threshold")
    image_group.add_argument("--fdr", action="store_true",
                            help="Use False Detection Rate algorithm")
    image_group.add_argument("--alpha", default=1e-2, type=float,
                            help="FDR Alpha")
    image_group.add_argument("--deblend-thresholds", default=0, type=int,
                            help="Number of deblending subthresholds; 0 to disable")
    image_group.add_argument("--grid", default=64, type=int,
                            help="Background grid segment size")
    image_group.add_argument("--bmaj", type=float,
                            help="Set beam: Major axis of beam (deg)")
    image_group.add_argument("--bmin", type=float,
                            help="Set beam: Minor axis of beam (deg)")
    image_group.add_argument("--bpa", type=float,
                            help="Set beam: Beam position angle (deg)")
    image_group.add_argument("--force-beam", action="store_true",
                            help="Force fit axis lengths to beam size")
    image_group.add_argument("--detection-image", type=str,
                            help="Find islands on different image")
    image_group.add_argument('--fixed-posns',
                            help="List of position coordinates to "
                                 "force-fit (decimal degrees, JSON, e.g [[123.4,56.7],[359.9,89.9]]) "
                                 "(Will not perform blind extraction in this mode)",
                            default=None)
    image_group.add_argument('--fixed-posns-file',
                            help="Path to file containing a list of positions to force-fit "
                                 "(Will not perform blind extraction in this mode)",
                            default=None)
    image_group.add_argument('--ffbox', type=float, default=3.,
                            help="Forced fitting positional box size as a multiple of beam width.")


    # Arguments relating to output:
    export_group = parser.add_argument_group("export")
    export_group.add_argument("--skymodel", action="store_true",
                        help="Generate sky model")
    export_group.add_argument("--csv", action="store_true",
                        help="Generate csv text file for use in programs such as TopCat")
    export_group.add_argument("--regions", action="store_true",
                        help="Generate DS9 region file(s)")
    export_group.add_argument("--rmsmap", action="store_true",
                        help="Generate RMS map")
    export_group.add_argument("--sigmap", action="store_true",
                        help="Generate significance map")
    export_group.add_argument("--residuals", action="store_true",
                        help="Generate residual maps")
    export_group.add_argument("--islands", action="store_true",
                        help="Generate island maps")

    # Finally, positional arguments- the file list:
    parser.add_argument('files', nargs='+',
                        help="Image files for processing")
    return parser

def parse_arguments():
    parser = construct_argument_parser()
    arguments = parser.parse_args()
    cli_args = vars(arguments)
    config_file = Path(cli_args.pop("config_file"))
    if not config_file.exists() or not config_file.is_file():
        raise ValueError(f"Config file {config_file} does not exist or is not a file. Specify config file location using the --config_file arguement.")

    pdb_on_crash = cli_args.pop("pdb")
    breakpoint()
    if pdb_on_crash:
        # Automatically start the debugger on an unhandled exception
        def excepthook(type, value, traceback):
            pdb.post_mortem(traceback)

        sys.excepthook = excepthook


    conf =  read_and_update_config_file(config_file, cli_args)
    return conf
