import json
import logging


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
