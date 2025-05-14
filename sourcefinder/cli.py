import argparse
import ast
from pathlib import Path

from sourcefinder.config import _read_conf_as_dict, Conf

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
                to_update[key] = overwrite_data.get(key) or value

    overwrite_params_from_nested_dict(combined_data, config_data)
    return Conf(**combined_data)

def construct_argument_parser():
    parser = argparse.ArgumentParser(
        description="PySE image configuration options. These can override the values specified in the TOML config file."
    )

    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config_file",
        default="trap_config.toml",
        help="""
        TOML file containing default input arguments to TraP.
        Default file name: trap_config.toml
        This is especially convenient when swapping between configurations for the same project.
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
        "--residuals",
        action="store_true",
        help="If set, residual maps will be generated after source extraction."
    )

    image_group.add_argument(
        "--islands",
        action="store_true",
        help="If set, individual islands of sources will be saved for inspection."
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
    return parser

def parse_arguments():
    parser = construct_argument_parser()
    arguments = parser.parse_args()
    cli_args = vars(arguments)
    config_file = cli_args.pop("config_file")
    conf =  read_and_update_config_file(config_file, cli_args)
    return conf

def main():
    arguments = parse_arguments()
    return run(arguments)


if __name__ == "__main__":
    sys.exit(main())
