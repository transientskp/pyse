#!/usr/bin/env python
"""
Simple interface to TKP source identification & measurement code.
John Sanders & John Swinbank, 2011.

This is a simplified script for running source finding with a minimal set of
arguments. It does not provide a full configuration interface or access to
all features.

Run as:

  $ python pyse.py file ...

For help with command line options:

  $ python pyse.py --help

See chapters 2 & 3 of Spreeuw, PhD Thesis, University of Amsterdam, 2010,
<http://dare.uva.nl/en/record/340633> for details.
"""
import argparse
import logging
import math
import numbers
import os.path
import sys
import pdb
from io import StringIO

import astropy.io.fits as pyfits
import numpy

from sourcefinder.accessors import open as open_accessor
from sourcefinder.accessors import sourcefinder_image_from_accessor
from sourcefinder.accessors import writefits as tkp_writefits
from sourcefinder.config import ImgConf
from sourcefinder.utility.monitoring import parse_monitoringlist_positions, construct_argument_parser, read_and_update_config_file
from sourcefinder.utils import generate_result_maps
from sourcefinder.config import Conf
from sourcefinder import image

def regions(sourcelist):
    """
    Return a string containing a DS9-compatible region file describing all the
    sources in sourcelist.
    """
    output = StringIO()
    print(u"# Region file format: DS9 version 4.1", file=output)
    print(
        u"global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        file=output)
    print(u"image", file=output)
    for source in sourcelist:
        # NB, here we convert from internal 0-origin indexing to DS9 1-origin indexing
        print(u"ellipse(%f, %f, %f, %f, %f)" % (
            source.x.value + 1.0,
            source.y.value + 1.0,
            source.smaj.value * 2,
            source.smin.value * 2,
            math.degrees(source.theta) + 90
        ), file=output)
    return output.getvalue()


def skymodel(sourcelist, ref_freq=73800000):
    """
    Return a string containing a skymodel from the extracted sources for use in self-calibration.
    """
    output = StringIO()
    print(
        u"#(Name, Type, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis, Orientation, "
        "ReferenceFrequency='60e6', SpectralIndex='[0.0]') = format",
        file=output)
    for source in sourcelist:
        print(u"%s, GAUSSIAN, %s, %s, %f, 0, 0, 0, %f, %f, %f, %f, [0]" % (
            u"ra:%fdec:%f" % (source.ra, source.dec),
            u"%fdeg" % (source.ra,),
            u"%fdeg" % (source.dec,),
            source.flux,
            source.smaj_asec,
            source.smin_asec,
            source.theta_celes,
            ref_freq
        ), file=output)
    return output.getvalue()


def csv(sourcelist):
    """
    Return a string containing a csv from the extracted sources.
    """
    output = StringIO()
    print(
        "ra, ra_err, dec, dec_err, smaj, smaj_err, smin, smin_err, pa, pa_err,"
        " int_flux, int_flux_err, pk_flux, pk_flux_err, x, y, snr, "
        "reduced_chisq", file=output)
    for source in sourcelist:
        values = (
            source.ra, source.ra.error, source.dec, source.dec.error,
            source.smaj_asec, source.smaj_asec.error, source.smin_asec,
            source.smin_asec.error, source.theta_celes,
            source.theta_celes.error, source.flux, source.flux.error,
            source.peak, source.peak.error, source.x, source.y, source.sig,
            source.reduced_chisq)
        print(", ".join(f"{float(v):.6f}" for v in values), file=output)
    return output.getvalue()


def summary(filename, sourcelist):
    """
    Return a string containing a human-readable summary of all sources in
    sourcelist.
    """
    output = StringIO()
    print(u"** %s **\n" % (filename), file=output)
    for source in sourcelist:
        print(u"RA: %s, dec: %s" % (str(source.ra), str(source.dec)),
              file=output)
        print(u"Error radius (arcsec): %s" % (str(source.error_radius)),
              file=output)
        print(u"Semi-major axis (arcsec): %s" % (str(source.smaj_asec)),
              file=output)
        print(u"Semi-minor axis (arcsec): %s" % (str(source.smin_asec)),
              file=output)
        print(u"Position angle: %s" % (str(source.theta_celes)), file=output)
        print(u"Flux: %s" % (str(source.flux)), file=output)
        print(u"Peak: %s\n" % (str(source.peak)), file=output)
    return output.getvalue()





def handle_args(args=None):
    """
    Parses command line options & arguments using OptionParser.
    Options & default values for the script are defined herein.
    """
    parser = construct_argument_parser()
    arguments = parser.parse_args()
    cli_args = vars(arguments)

    # Extract file paths, which are only to be supplied via command line, not config
    files = cli_args.pop("files")

    # Automatically start the debugger on an unhandled exception if specified
    debug_on_error = cli_args.pop("pdb")
    if debug_on_error:
        def excepthook(type, value, traceback):
            pdb.post_mortem(traceback)

        sys.excepthook = excepthook

    # Merge the CLI arguments with the config file parameters
    config_file = cli_args.pop("config_file")
    conf =  read_and_update_config_file(config_file, cli_args)

    # breakpoint()
    # FIXME: nocheckin, add back fixed coord parsing


    # Overwrite 'fixed_coords' with a parsed list of coords
    # collated from both command line and file.
    fixed_coords = parse_monitoringlist_positions(
        conf.extraction, str_name="fixed_posns", list_name="fixed_posns_file"
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
        if conf.extraction.fdr:
            parser.error("--fdr not supported with fixed positions")
        elif conf.extraction.detection_image:
            parser.error("--detection-image not supported with fixed positions")
        mode = "fixed"  # mode 2 above
    elif conf.extraction.fdr:
        if conf.extraction.detection_image:
            parser.error("--detection-image not supported with --fdr")
        mode = "fdr"  # mode 1.3 above
    elif conf.extraction.detection_image:
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
    print(u"Detecting islands in %s" % (filename,))
    print(u"Thresholding with det = %f sigma, analysis = %f sigma" % (det, anl))
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
        print(u"WARNING: partial beam specification ignored")
    return None


def bailout(reason):
    # Exit with error
    print(u"ERROR: %s" % (reason))
    sys.exit(1)


def run_sourcefinder(files, conf, mode):
    """
    Iterate over the list of files, running a sourcefinding step on each in
    turn. If specified, a DS9-compatible region file and/or a FITS file
    showing the residuals after Gaussian fitting are dumped for each file.
    A string containing a human readable list of sources is returned.
    """
    output = StringIO()

    beam = get_beam(conf.extraction.bmaj, conf.extraction.bmin, conf.extraction.bpa)
    configuration = conf.image

    if mode == "detimage":
        labels, labelled_data = get_detection_labels(
            conf.extraction.detection_image, conf.extraction.detection, conf.extraction.analysis, beam,
            configuration
        )
    else:
        labels, labelled_data = [], None

    for counter, filename in enumerate(files):
        print(u"Processing %s (file %d of %d)." % (
            filename, counter + 1, len(files)))
        imagename = os.path.splitext(os.path.basename(filename))[0]
        ff = open_accessor(filename, beam=beam, plane=0)
        imagedata = sourcefinder_image_from_accessor(ff, conf=configuration)

        if mode == "fixed":
            # FIXME: conf.extraction.fixed_coords does not exist
            sr = imagedata.fit_fixed_positions(conf.extraction.fixed_coords,
                                               conf.extraction.ffbox * max(
                                                   imagedata.beam[0:2])
                                               )

        else:
            if mode == "fdr":
                print(u"Using False Detection Rate algorithm with alpha = %f" % (
                    conf.extraction.alpha,))
                sr = imagedata.fd_extract(
                    alpha=conf.extraction.alpha,
                    deblend_nthresh=conf.extraction.deblend_thresholds,
                    force_beam=conf.extraction.force_beam
                )
            else:
                if labelled_data is None:
                    print(
                        u"Thresholding with det = %f sigma, analysis = %f sigma" % (
                         conf.extraction.detection, conf.extraction.analysis))

                sr = imagedata.extract(
                    det=conf.extraction.detection, anl=conf.extraction.analysis,
                    labelled_data=labelled_data, labels=labels,
                    deblend_nthresh=conf.extraction.deblend_thresholds,
                    force_beam=conf.extraction.force_beam
                )

        if conf.export.regions:
            regionfile = imagename + ".reg"
            regionfile = open(regionfile, 'w')
            regionfile.write(regions(sr))
            regionfile.close()
        # This applies a slower method for computing the Gaussian islands and
        # residuals than the methods from the extract module - since these
        # methods are called in parallel, from image.py, or vectorized - and
        # extends to pixel positions corresponding to values below the analysis
        # threshold, i.e. well into the background noise. Some users may want to
        # accept the extra compute time to be able to compare the residuals with
        # the background noise.
        # if conf.extraction.residuals or conf.extraction.islands:
        #     gaussian_map, residual_map = generate_result_maps(imagedata.data,
        #                                                       sr)
        if conf.image.residuals:
            residualfile = imagename + ".residuals.fits"
            writefits(residualfile, imagedata.Gaussian_residuals,
                      pyfits.getheader(filename))
        if conf.image.islands:
            islandfile = imagename + ".islands.fits"
            writefits(islandfile, imagedata.Gaussian_islands,
                      pyfits.getheader(filename))
        if conf.export.rmsmap:
            rmsfile = imagename + ".rms.fits"
            writefits(rmsfile, numpy.array(imagedata.rmsmap),
                      pyfits.getheader(filename))
        if conf.export.sigmap:
            sigfile = imagename + ".sig.fits"
            writefits(sigfile,
                      numpy.array(imagedata.data_bgsubbed / imagedata.rmsmap),
                      pyfits.getheader(filename))
        if conf.export.skymodel:
            with open(imagename + ".skymodel", 'w') as skymodelfile:
                if ff.freq_eff:
                    skymodelfile.write(skymodel(sr, ff.freq_eff))
                else:
                    print(
                        u"WARNING: Using default reference frequency for %s" % (
                        skymodelfile.name,))
                    skymodelfile.write(skymodel(sr))
        if conf.export.csv:
            with open(imagename + ".csv", 'w') as csvfile:
                csvfile.write(csv(sr))
                print(summary(filename, sr), end=u' ', file=output)
    return output.getvalue()


if __name__ == "__main__":
    logging.basicConfig()
    conf, mode, files = handle_args()
    print(run_sourcefinder(files, conf, mode), end=u' ')
