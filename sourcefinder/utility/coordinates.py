#
# LOFAR Transients Key Project
"""
General purpose astronomical coordinate handling routines.
"""
import numpy
import datetime
import logging
import math
import sys
from numba import njit, guvectorize, float64

import pytz
from astropy import wcs as pywcs
from casacore.measures import measures
from casacore.quanta import quantity

logger = logging.getLogger(__name__)

# Note that we take a +ve longitude as WEST.
CORE_LAT = 52.9088
CORE_LON = -6.8689

# ITRF position of CS002
# Should be a good approximation for anything refering to the LOFAR core.
ITRF_X = 3826577.066110000
ITRF_Y = 461022.947639000
ITRF_Z = 5064892.786

# Useful constants
SECONDS_IN_HOUR = 60 ** 2
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR


def julian_date(time=None, modified=False):
    """
    Calculate the Julian date at a given timestamp.

    Parameters
    ----------
    time : datetime.datetime, default: None
        Timestamp to calculate JD for. If not provided, the current UTC time
        will be used.
    modified : bool, default: False
        If True, return the Modified Julian Date, which is the number of days
        (including fractions) that have elapsed between the start of
        17 November 1858 AD and the specified time.

    Returns
    -------
    float
        Julian date value.
    """
    if not time:
        time = datetime.datetime.now(pytz.utc)
    mjdstart = datetime.datetime(1858, 11, 17, tzinfo=pytz.utc)
    mjd = time - mjdstart
    mjd_daynumber = (mjd.days + mjd.seconds / (24. * 60 ** 2) +
                     mjd.microseconds / (24. * 60 ** 2 * 1000 ** 2))
    if modified:
        return mjd_daynumber
    return 2400000.5 + mjd_daynumber


def mjd2datetime(mjd):
    """
    Convert a Modified Julian Date to datetime via 'unix time' representation.

    NB 'unix time' is defined by the casacore/casacore package.

    Parameters
    ----------
    mjd : float
        Modified Julian Date to be converted.

    Returns
    -------
    datetime.datetime
        A datetime object representing the given Modified Julian Date.
    """
    q = quantity("%sd" % mjd)
    return datetime.datetime.fromtimestamp(q.to_unix_time())


def mjd2lst(mjd, position=None):
    """
    Converts a Modified Julian Date into Local Apparent Sidereal Time in
    seconds at a given position. If position is None, we default to the
    reference position of CS002.

    Parameters
    ----------
    mjd : float
        Modified Julian Date in days.
    position : casacore measure, default: None
        Position for the LST calculation.

    Returns
    -------
    float
        Local Apparent Sidereal Time in seconds.
    """
    dm = measures()
    position = position or dm.position(
        "ITRF", "%fm" % ITRF_X, "%fm" % ITRF_Y, "%fm" % ITRF_Z
    )
    dm.do_frame(position)
    last = dm.measure(dm.epoch("UTC", "%fd" % mjd), "LAST")
    fractional_day = last['m0']['value'] % 1
    return fractional_day * 24 * SECONDS_IN_HOUR


def mjds2lst(mjds, position=None):
    """
    As mjd2lst(), but takes an argument in seconds rather than days.

    Parameters
    ----------
    mjds : float
        Modified Julian Date (in seconds).
    position : casacore measure, default: None
        Position for Local Sidereal Time calculations.

    Returns
    -------
    float
        Local Apparent Sidereal Time in seconds.
    """
    return mjd2lst(mjds / SECONDS_IN_DAY, position)


def jd2lst(jd, position=None):
    """
    Converts a Julian Date into Local Apparent Sidereal Time in seconds at a
    given position. If position is None, we default to the reference position
    of CS002.

    Parameters
    ----------
    jd : float
        Julian Date to be converted.
    position : casacore measure, default: None
        Position for Local Sidereal Time calculations.

    Returns
    -------
    float
        Local Apparent Sidereal Time in seconds.
    """
    return mjd2lst(jd - 2400000.5, position)


# NB: datetime is not sensitive to leap seconds.
# However, leap seconds were first introduced in 1972.
# So there are no leap seconds between the start of the
# Modified Julian epoch and the start of the Unix epoch,
# so this calculation is safe.
# julian_epoch = datetime.datetime(1858, 11, 17)
# unix_epoch = datetime.datetime(1970, 1, 1, 0, 0)
# delta = unix_epoch - julian_epoch
# deltaseconds = delta.total_seconds()
# unix_epoch = 3506716800

# The above is equivalent to this:
unix_epoch = quantity("1970-01-01T00:00:00").get_value('s')


def julian2unix(timestamp):
    """
    Convert a modified Julian timestamp (number of seconds since 17 November
    1858) to Unix timestamp (number of seconds since 1 January 1970).

    Parameters
    ----------
    timestamp : numbers.Number
        Number of seconds since the Unix epoch.

    Returns
    -------
    numbers.Number
        Number of seconds since the modified Julian epoch.
    """
    return timestamp - unix_epoch


def unix2julian(timestamp):
    """
    Convert a Unix timestamp (number of seconds since 1 January 1970) to a
    modified Julian timestamp (number of seconds since 17 November 1858).

    Parameters
    ----------
    timestamp : numbers.Number
        Number of seconds since the Unix epoch.

    Returns
    -------
    numbers.Number
        Number of seconds since the modified Julian epoch.
    """
    return timestamp + unix_epoch


def sec2deg(seconds):
    """
    Convert seconds of time to degrees of arc.

    Parameters
    ----------
    seconds : float
        Time in seconds to be converted to degrees of arc.

    Returns
    -------
    float
        Equivalent value in degrees of arc.
    """
    return 15.0 * seconds / 3600.0


def sec2days(seconds):
    """
    Convert seconds to the equivalent number of days.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    float
        Equivalent time duration in days.
    """
    return seconds / (24.0 * 3600)


def sec2hms(seconds):
    """
    Convert seconds to hours, minutes, and seconds.

    Returns
    -------
    hours : int

    minutes : int

    seconds : float

    """
    hours, seconds = divmod(seconds, 60 ** 2)
    minutes, seconds = divmod(seconds, 60)
    return int(hours), int(minutes), seconds


def altaz(mjds, ra, dec, lat=CORE_LAT):
    """
    Calculate the azimuth and elevation of a source from time and position
    on the sky.

    Parameters
    ----------
    mjds : float
        Modified Julian Date in seconds.
    ra : float
        Right Ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    lat : float, default : CORE_LAT
        Latitude of the observer in degrees.

    Returns
    -------
    hrz_altitude: float
        Altitude of the source in degrees.
    hrz_azimuth :float
        Azimuth of the source in degrees.

    """

    # compute hour angle in degrees
    ha = mjds2lst(mjds) - ra
    if (ha < 0):
        ha = ha + 360

    # convert degrees to radians
    ha, dec, lat = [math.radians(value) for value in (ha, dec, lat)]

    # compute altitude in radians
    sin_alt = (math.sin(dec) * math.sin(lat) +
               math.cos(dec) * math.cos(lat) * math.cos(ha))
    alt = math.asin(sin_alt)

    # compute azimuth in radians
    # divide by zero error at poles or if alt = 90 deg
    cos_az = ((math.sin(dec) - math.sin(alt) * math.sin(lat)) /
              (math.cos(alt) * math.cos(lat)))
    az = math.acos(cos_az)
    # convert radians to degrees
    hrz_altitude, hrz_azimuth = [math.degrees(value) for value in (alt, az)]
    # choose hemisphere
    if math.sin(ha) > 0:
        hrz_azimuth = 360 - hrz_azimuth

    return hrz_altitude, hrz_azimuth


def ratohms(radegs):
    """Convert RA in decimal degrees format to hours, minutes,
    seconds format.

    Parameters
    ----------
    radegs : float
        RA in degrees format.

    Returns
    -------
    hours : int

    minutes : int

    seconds : float

    """

    radegs %= 360
    raseconds = radegs * 3600 / 15.0
    return sec2hms(raseconds)


def dectodms(decdegs):
    """Convert Declination in decimal degrees format to hours,
    minutes, seconds format.

    Parameters
    ----------
    decdegs : float
        Declination in decimal degrees format.

    Returns
    -------
    hours : int

    minutes : int

    seconds : float

    """

    sign = -1 if decdegs < 0 else 1
    decdegs = abs(decdegs)
    if decdegs > 90:
        raise ValueError("coordinate out of range")
    decd = int(decdegs)
    decm = int((decdegs - decd) * 60)
    decs = (((decdegs - decd) * 60) - decm) * 60
    # Necessary because of potential roundoff errors
    if decs - 60 > -1e-7:
        decm += 1
        decs = 0
        if decm == 60:
            decd += 1
            decm = 0
            if decd > 90:
                raise ValueError("coordinate out of range")

    if sign == -1:
        if decd == 0:
            if decm == 0:
                decs = -decs
            else:
                decm = -decm
        else:
            decd = -decd
    return decd, decm, decs


def propagate_sign(val1, val2, val3):
    """Determine the sign of the input values and ensure consistency.

    casacore (reasonably enough) demands that a minus sign (if required)
    comes at the start of the quantity. Thus "-0D30M" rather than "0D-30M".
    Python regards "-0" as equal to "0"; we need to split off a separate sign
    field.

    If more than one of our inputs is negative, it's not clear what the user
    meant: we raise an exception.

    Parameters
    ----------
    val1 : float
        First input value (hours or degrees).
    val2 : float
        Second input value (minutes).
    val3 : float
        Third input value (seconds).

    Returns
    -------
    sign : str
        "+" or "-" string denoting the sign.
    val1: float
        Absolute value of ``val1``.
    val2 : float
        Absolute value of ``val2``.
    val3 : float
        Absolute value of ``val3``.

    """
    signs = [x < 0 for x in (val1, val2, val3)]
    if signs.count(True) == 0:
        sign = "+"
    elif signs.count(True) == 1:
        sign, val1, val2, val3 = "-", abs(val1), abs(val2), abs(val3)
    else:
        raise ValueError("Too many negative coordinates")
    return sign, val1, val2, val3


def hmstora(rah, ram, ras):
    """Convert RA in hours, minutes, seconds format to decimal
    degrees format.

    Parameters
    ----------
    rah : float
        Right Ascension hours.
    ram : float
        Right Ascension minutes.
    ras : float
        Right Ascension seconds.

    Returns
    -------
    float
        RA in decimal degrees.

    """
    sign, rah, ram, ras = propagate_sign(rah, ram, ras)
    ra = quantity("%s%dH%dM%f" % (sign, rah, ram, ras)).get_value()
    if abs(ra) >= 360:
        raise ValueError("coordinates out of range")
    return ra


def dmstodec(decd, decm, decs):
    """Convert Dec in degrees, minutes, seconds format to decimal
    degrees format.

    Parameters
    ----------
    decd : int
        Degrees component of the Declination.
    decm : int
        Minutes component of the Declination.
    decs : float
        Seconds component of the Declination.

    Returns
    -------
    float
        Declination in decimal degrees.

    """
    sign, decd, decm, decs = propagate_sign(decd, decm, decs)
    dec = quantity("%s%dD%dM%f" % (sign, decd, decm, decs)).get_value()
    if abs(dec) > 90:
        raise ValueError("coordinates out of range")
    return dec


def cmp(a, b):
    """Compare two values and return an integer indicating their
    relationship.

    Parameters
    ----------
    a : Any
        The first value to compare.
    b : Any
        The second value to compare.

    Returns
    -------
    int
        1 if `a` is greater than `b`, -1 if `a` is less than `b`,
        and 0 if they are equal.

    """
    return bool(a > b) - bool(a < b)


def angsep(ra1, dec1, ra2, dec2):
    """Find the angular separation of two sources, in arcseconds,
    using the proper spherical trigonometry formula.

    Parameters
    ----------
    ra1 : float
        Right Ascension of the first source, in decimal degrees.
    dec1 : float
        Declination of the first source, in decimal degrees.
    ra2 : float
        Right Ascension of the second source, in decimal degrees.
    dec2 : float
        Declination of the second source, in decimal degrees.

    Returns
    -------
    float
        Angular separation between the two sources, in arcseconds.

    """

    b = (math.pi / 2) - math.radians(dec1)
    c = (math.pi / 2) - math.radians(dec2)
    temp = (math.cos(b) * math.cos(c)) + (
    math.sin(b) * math.sin(c) * math.cos(math.radians(ra1 - ra2)))

    # Truncate the value of temp at +- 1: it makes no sense to do math.acos()
    # of a value outside this range, but occasionally we might get one due to
    # rounding errors.
    if abs(temp) > 1.0:
        temp = 1.0 * cmp(temp, 0)

    return 3600 * math.degrees(math.acos(temp))


@njit
def cmp_jitted(a, b):
    """Compare two values and return an integer indicating their
    relationship.

    Parameters
    ----------
    a : Any
        The first value to compare.
    b : Any
        The second value to compare.

    Returns
    -------
    int
        1 if `a` is greater than `b`, -1 if `a` is less than `b`,
        and 0 if they are equal.

    """
    return bool(a > b) - bool(a < b)


@guvectorize([(float64[:], float64[:], float64[:])],
             '(n), (n) -> ()', nopython=True)
def angsep_vectorized(ra_dec1, ra_dec2, angular_separation):
    """Find the angular separation of two sources, in arcseconds,
    using the proper spherical trigonometry formula.

    Parameters
    ----------
    ra_dec1 : ndarray
        RA and Dec of the first source, in decimal degrees.
    ra_dec2 : ndarray
        RA and Dec of the second source, in decimal degrees.
    angular_separation : float
        Angular separation between the two sources, in arcseconds.
        This value is assigned within the function due to the guvectorize
        decorator.

    Returns
    -------
    None
        The result is stored in the `angular_separation` parameter.

    """
    ra1, dec1 = ra_dec1
    ra2, dec2 = ra_dec2

    b = (math.pi / 2) - math.radians(dec1)
    c = (math.pi / 2) - math.radians(dec2)
    temp = (math.cos(b) * math.cos(c)) + (
            math.sin(b) * math.sin(c) * math.cos(math.radians(ra1 - ra2)))

    # Truncate the value of temp at +- 1: it makes no sense to do math.acos()
    # of a value outside this range, but occasionally we might get one due to
    # rounding errors.
    if abs(temp) > 1.0:
        temp = 1.0 * cmp_jitted(temp, 0)

    angular_separation[0] = 3600 * math.degrees(math.acos(temp))


def alphasep(ra1, ra2, dec1, dec2):
    """Find the angular separation of two sources in RA, in
    arcseconds.

    Parameters
    ----------
    ra1 : float
        Right ascension of the first source, in decimal degrees.
    ra2 : float
        Right ascension of the second source, in decimal degrees.
    dec1 : float
        Declination of the first source, in decimal degrees.
    dec2 : float
        Declination of the second source, in decimal degrees.

    Returns
    -------
    float
        Angular separation in RA, in arcseconds.

    """

    return 3600 * (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2.0))


def deltasep(dec1, dec2):
    """Find the angular separation of two sources in declination, in
    arcseconds.

    Parameters
    ----------
    dec1 : float
        Declination of the first source, in decimal degrees.
    dec2 : float
        Declination of the second source, in decimal degrees.

    Returns
    -------
    float
        Angular separation in declination, in arcseconds.

    """

    return 3600 * (dec1 - dec2)


def alpha(l, m, alpha0, delta0):
    """Convert a coordinate in l, m into a coordinate in RA.

    Parameters
    ----------
    l : float
        Direction cosine along the l-axis, given by offset in cells times
        cell size (in radians).
    m : float
        Direction cosine along the m-axis, given by offset in cells times
        cell size (in radians).
    alpha0 : float
        Right ascension of the centre of the field, in decimal degrees.
    delta0 : float
        Declination of the centre of the field, in decimal degrees.

    Returns
    -------
    float
        Right Ascension (RA) in decimal degrees.

    """
    return (alpha0 + (math.degrees(math.atan2(l, (
        (math.sqrt(1 - (l * l) - (m * m)) * math.cos(math.radians(delta0))) -
        (m * math.sin(math.radians(delta0))))))))


def alpha_inflate(theta, decl):
    """Compute the RA expansion for a given theta at a given
    declination.

    Parameters
    ----------
    theta : float
        Angular distance from the center, in decimal degrees.
    decl : float
        Declination of the source, in decimal degrees.

    Returns
    -------
    float
        RA inflation in decimal degrees.

    Notes
    -----
    For a derivation, see MSR TR 2006 52, Section 2.1:
    http://research.microsoft.com/apps/pubs/default.aspx?id=64524

    """
    if abs(decl) + theta > 89.9:
        return 180.0
    else:
        return math.degrees(abs(math.atan(
            math.sin(math.radians(theta)) / math.sqrt(abs(
                math.cos(math.radians(decl - theta)) * math.cos(
                    math.radians(decl + theta)))))))


def delta(l, m, delta0):
    """Convert a coordinate in l, m into a coordinate in declination.

    Parameters
    ----------
    l : float
        Direction cosine along the l-axis, given by offset in cells times cell
        size (in radians).
    m : float
        Direction cosine along the m-axis, given by offset in cells times cell
        size (in radians).
    delta0 : float
        Declination of the center of the field, in decimal degrees.

    Returns
    -------
    float
        Declination in decimal degrees.

    """
    return math.degrees(math.asin(m * math.cos(math.radians(delta0)) +
                                  (math.sqrt(1 - (l * l) - (m * m)) *
                                   math.sin(math.radians(delta0)))))


def l(ra, dec, cra, incr):
    """Convert a coordinate in RA, Dec into a direction cosine l.

    Parameters
    ----------
    ra : float
        Right ascension of the source, in decimal degrees.
    dec : float
        Declination of the source, in decimal degrees.
    cra : float
        Right ascension of the centre of the field, in decimal degrees.
    incr : float
        Number of degrees per pixel (negative in the case of RA).

    Returns
    -------
    float
        Direction cosine l.

    """
    return ((math.cos(math.radians(dec)) * math.sin(math.radians(ra - cra))) /
            (math.radians(incr)))


def m(ra, dec, cra, cdec, incr):
    """Convert a coordinate in RA, Dec into a direction cosine m.

    Parameters
    ----------
    ra : float
        Right ascension of the source, in decimal degrees.
    dec : float
        Declination of the source, in decimal degrees.
    cra : float
        Right ascension of the center of the field, in decimal degrees.
    cdec : float
        Declination of the center of the field, in decimal degrees.
    incr : float
        Number of degrees per pixel.

    Returns
    -------
    float
        Direction cosine m.

    """
    return ((math.sin(math.radians(dec)) * math.cos(math.radians(cdec))) -
            (math.cos(math.radians(dec)) * math.sin(math.radians(cdec)) *
             math.cos(math.radians(ra - cra)))) / math.radians(incr)


def lm_to_radec(ra0, dec0, l, m):
    """Find the l direction cosine of a source in a radio image,
    given the RA and Dec of the field centre.

    Parameters
    ----------
    ra0 : float
        Right ascension of the field center, in decimal degrees.
    dec0 : float
        Declination of the field center, in decimal degrees.
    l : float
        Direction cosine of the source along the l-axis.
    m : float
        Direction cosine of the source along the m-axis.

    Returns
    -------
    ra : float
        Right ascension in decimal degrees.
    dec : float
        Declination in decimal degrees.

    Notes
    -----
    This function should be the inverse of radec_to_lmn, but it is
    not. There is likely an error here.

    """

    sind0 = math.sin(dec0)
    cosd0 = math.cos(dec0)
    dl = l
    dm = m
    d0 = dm * dm * sind0 * sind0 + dl * dl - 2 * dm * cosd0 * sind0
    sind = math.sqrt(abs(sind0 * sind0 - d0))
    cosd = math.sqrt(abs(cosd0 * cosd0 + d0))
    if sind0 > 0:
        sind = abs(sind)
    else:
        sind = -abs(sind)

    dec = math.atan2(sind, cosd)

    if l != 0:
        ra = math.atan2(-dl, (cosd0 - dm * sind0)) + ra0
    else:
        ra = math.atan2(1e-10, (cosd0 - dm * sind0)) + ra0

        # Calculate RA,Dec from l,m and phase center.  Note: As done in
        # Meqtrees, which seems to differ from l, m functions above.  Meqtrees
        # equation may have problems, judging from my difficulty fitting a
        # fringe to L4086 data.  Pandey's equation is now used in radec_to_lmn

    return ra, dec


def radec_to_lmn(ra0, dec0, ra, dec):
    """Convert equatorial coordinates (RA, Dec) to direction cosines
    (l, m, n).

    Parameters
    ----------
    ra0 : float
        Right ascension of the reference point (in decimal degrees).
    dec0 : float
        Declination of the reference point (in decimal degrees).
    ra : float
        Right Ascension of the target point (in decimal degrees).
    dec : float
        Declination of the target point (in decimal degrees).

    Returns
    -------
    l : float
        Direction cosine along the l-axis.
    m : float
        Direction cosine along the m-axis.
    n : float
        Direction cosine along the n-axis.

    """
    l = math.cos(dec) * math.sin(ra - ra0)
    sind0 = math.sin(dec0)
    if sind0 != 0:
        # from pandey;  gives same results for casa and cyga
        m = (math.sin(dec) * math.cos(dec0) -
             math.cos(dec) * math.sin(dec0) * math.cos(ra - ra0))
    else:
        m = 0
    n = math.sqrt(1 - l ** 2 - m ** 2)
    return l, m, n


def eq_to_gal(ra, dec):
    """Find the Galactic coordinates of a source given the equatorial
    coordinates.

    Parameters
    ----------
    ra : float
        Right ascension (RA) in decimal degrees.
    dec : float
        Declination (Dec) in decimal degrees.

    Returns
    -------
    lon_l : float
        Galactic longitude in decimal degrees.
    lat_b : float
        Galactic latitude in decimal degrees.

    """
    dm = measures()

    result = dm.measure(
        dm.direction("J200", "%fdeg" % ra, "%fdeg" % dec),
        "GALACTIC"
    )
    lon_l = math.degrees(result['m0']['value']) % 360  # 0 < ra < 360
    lat_b = math.degrees(result['m1']['value'])

    return lon_l, lat_b


def gal_to_eq(lon_l, lat_b):
    """Find the equatorial coordinates of a source given the Galactic
    coordinates.

    Parameters
    ----------
    lon_l : float
        Galactic longitude in decimal degrees.
    lat_b : float
        Galactic latitude in decimal degrees.

    Returns
    -------
    ra : float
        Right ascension (RA) in decimal degrees.
    dec : float
        Declination (Dec) in decimal degrees.

    """
    dm = measures()

    result = dm.measure(
        dm.direction("GALACTIC", "%fdeg" % lon_l, "%fdeg" % lat_b),
        "J2000"
    )
    ra = math.degrees(result['m0']['value']) % 360  # 0 < ra < 360
    dec = math.degrees(result['m1']['value'])

    return ra, dec


def eq_to_cart(ra, dec):
    """Find the cartesian coordinates on the unit sphere given the
    equatorial coordinates.

    Parameters
    ----------
    ra : float
        Right ascension (RA) in decimal degrees.
    dec : float
        Declination (Dec) in decimal degrees.

    Returns
    -------
    x : float
        Cartesian x-coordinate.
    y : float
        Cartesian y-coordinate.
    z : float
        Cartesian z-coordinate.

    """
    return (
    math.cos(math.radians(dec)) * math.cos(math.radians(ra)),  # Cartesian x
    math.cos(math.radians(dec)) * math.sin(math.radians(ra)),  # Cartesian y
    math.sin(math.radians(dec)))  # Cartesian z


class CoordSystem:
    """A container for constant strings representing different
    coordinate systems.

    """
    FK4 = "B1950 (FK4)"
    FK5 = "J2000 (FK5)"


def coordsystem(name):
    """Given a string, return a constant from class CoordSystem.

    Parameters
    ----------
    name : str
        The name of the coordinate system (e.g., 'j2000', 'fk5', 'b1950',
        'fk4').

    Returns
    -------
    str
        A constant from the CoordSystem class representing the coordinate
        system.

    Raises
    ------
    KeyError
        If the provided name does not match any known coordinate system.

    """
    mappings = {
        'j2000': CoordSystem.FK5,
        'fk5': CoordSystem.FK5,
        CoordSystem.FK5.lower(): CoordSystem.FK5,
        'b1950': CoordSystem.FK4,
        'fk4': CoordSystem.FK4,
        CoordSystem.FK4.lower(): CoordSystem.FK4
    }
    return mappings[name.lower()]


def convert_coordsystem(ra, dec, insys, outsys):
    """Convert RA & dec (given in decimal degrees) between equinoxes.

    This function takes right Ascension (RA) and declination (Dec) coordinates
    in decimal degrees and converts them between different equinoxes, such as
    B1950 and J2000. The input and output equinoxes are specified as parameters.

    Parameters
    ----------
    ra : float
        Right ascension in decimal degrees.
    dec : float
        Declination in decimal degrees.
    insys : str
        Input equinox, e.g., 'B1950' or 'J2000'.
    outsys : str
        Output equinox, e.g., 'B1950' or 'J2000'.

    Returns
    -------
    ra : float
        Converted right ascension in decimal degrees.
    dec : float
        Converted declination in decimal degrees.

    Raises
    ------
    Exception
        If the input or output equinox is unknown.

    """
    dm = measures()

    if insys == CoordSystem.FK4:
        insys = "B1950"
    elif insys == CoordSystem.FK5:
        insys = "J2000"
    else:
        raise Exception("Unknown Coordinate System")

    if outsys == CoordSystem.FK4:
        outsys = "B1950"
    elif outsys == CoordSystem.FK5:
        outsys = "J2000"
    else:
        raise Exception("Unknown Coordinate System")

    result = dm.measure(
        dm.direction(insys, "%fdeg" % ra, "%fdeg" % dec),
        outsys
    )

    ra = math.degrees(result['m0']['value']) % 360  # 0 < ra < 360
    dec = math.degrees(result['m1']['value'])

    return ra, dec


class WCS:
    """Wrapper around pywcs.WCS.

    This is primarily to preserve API compatibility with the earlier,
    home-brewed python-wcslib wrapper. It includes:

      * A fix for the reference pixel lying at the zenith;
      * Raises ValueError if coordinates are invalid.

    """
    # ORIGIN is the upper-left corner of the image. pywcs supports both 0
    # (NumPy, C-style) or 1 (FITS, Fortran-style). The TraP uses 1.
    ORIGIN = 1

    # We can set these attributes on the pywcs.WCS().wcs object to configure
    # the coordinate system.
    WCS_ATTRS = ("crpix", "cdelt", "crval", "ctype", "cunit", "crota")

    def __init__(self):
        # Currently, we only support two dimensional images.
        self.wcs = pywcs.WCS(naxis=2)

    def __setattr__(self, attrname, value):
        if attrname in self.WCS_ATTRS:
            # Account for arbitrary coordinate rotations in images pointing at
            # the North Celestial Pole. We set the reference direction to
            # infinitesimally less than 90 degrees to avoid any ambiguity. See
            # discussion at #4599.
            if attrname == "crval" and (
                    value[1] == 90 or value[1] == math.pi / 2):
                value = (value[0], value[1] * (1 - sys.float_info.epsilon))
            self.wcs.wcs.__setattr__(attrname, value)
        else:
            super().__setattr__(attrname, value)

    def __getattr__(self, attrname):
        if attrname in self.WCS_ATTRS:
            return getattr(self.wcs.wcs, attrname)
        raise AttributeError(f"{type(self)!r} object has no attribute {attrname!r}")

    def p2s(self, pixpos):
        """Convert pixel coordinates to spatial coordinates.

        This function converts a given pixel position (x, y) into spatial
        coordinates (right ascension and declination).

        Parameters
        ----------
        pixpos : list or tuple
            A list or tuple of two floats containing the pixel position as
            [x, y].

        Returns
        -------
        ra : float
            Right ascension corresponding to the pixel position in
            decimal degrees.
        dec : float
            Declination corresponding to the pixel position in decimal
            degrees.

        """
        ra, dec = self.wcs.wcs_pix2world(pixpos[0], pixpos[1], self.ORIGIN)
        if math.isnan(ra) or math.isnan(dec):
            raise RuntimeError("Spatial position is not a number")
        return float(ra), float(dec)

    def s2p(self, spatialpos):
        """Convert spatial coordinates to pixel coordinates.

        This function converts a given spatial position (right ascension and
        declination) into pixel coordinates (x, y).

        Parameters
        ----------
        spatialpos : tuple[float, float]
            A tuple containing:
            - ra (float): Right ascension in decimal degrees.
            - dec (float): Declination in decimal degrees.

        Returns
        -------
        x : float
            X pixel value corresponding to the spatial position.
        y : float
            Y pixel value corresponding to the spatial position.

        """

        x, y = self.wcs.wcs_world2pix(spatialpos[0], spatialpos[1], self.ORIGIN)
        if math.isnan(x) or math.isnan(y):
            raise RuntimeError("Pixel position is not a number")
        return float(x), float(y)

    def all_p2s(self, array_of_pixpos):
        """Vectorized pixel to spatial coordinate conversion, making
        use of all_pix2world from astropy. This will save time when
        thousands of sources are detected.

        Parameters
        ----------
        array_of_pixpos : ndarray
            A (N, 2) array where each row represents [x, y] pixel positions.

        Returns
        -------
        ndarray
            A (N, 2) array where each row contains:
            - Right ascension (float) in decimal degrees.
            - Declination (float) in decimal degrees.

        """
        sky_coordinates = self.wcs.all_pix2world(array_of_pixpos,
                                                 self.ORIGIN)
        if numpy.isnan(sky_coordinates).any():
            raise RuntimeError("Spatial position is not a number")
        # Mimic conditional from extract.Detection._physical_coordinates
        if numpy.any(numpy.abs(abs(sky_coordinates[:, 1] > 90.0))):
            raise ValueError("At least one object falls outside the sky")

        return sky_coordinates
