import os
import unittest


def requires_database():
    """Decorator to skip a test if database functionality is
    disabled.

    This function checks the environment variable `TKP_DISABLEDB`. If it is set
    to a truthy value, the test is skipped with an appropriate message.

    Returns
    -------
    function
        A decorator that either skips the test or allows it to run.

    """
    if os.environ.get("TKP_DISABLEDB", False):
        return unittest.skip("Database functionality disabled in configuration")
    return lambda func: func


def requires_data(*args):
    """Decorator to skip a test if required data files are not
    available.

    This function checks for the existence of the specified data files.
    If any of the files do not exist, the test is skipped with an appropriate
    message.

    Parameters
    ----------
    *args : str
        Variable-length argument list of file paths to check.

    Returns
    -------
    function
        A decorator that either skips the test or allows it to run.

    """
    for filename in args:
        if not os.path.exists(filename):
            return unittest.skip("Test data (%s) not available" % filename)
    return lambda func: func


def requires_module(module_name):
    """Decorator to skip a test if a required module is not
    available.

    This function attempts to import the specified module. If the module
    cannot be imported, the test is skipped with an appropriate message.

    Parameters
    ----------
    module_name : str
        The name of the module to check for availability.

    Returns
    -------
    function
        A decorator that either skips the test or allows it to run.

    """
    try:
        __import__(module_name)
    except ImportError:
        return unittest.skip("Required module (%s) not available" % module_name)
    return lambda func: func


def duration(test_duration):
    """Decorator to skip a test if its duration exceeds the maximum
    allowed duration.

    This function checks the environment variable
    ``TKP_MAXTESTDURATION`` to determine the maximum allowed test
    duration. If the test's duration exceeds this value, the test is
    skipped with an appropriate message.

    Parameters
    ----------
    test_duration : float
        The duration of the test in seconds.

    Returns
    -------
    function
        A decorator that either skips the test or allows it to run.

    """
    max_duration = float(os.environ.get("TKP_MAXTESTDURATION", False))
    if max_duration:
        if max_duration < test_duration:
            return unittest.skip(
             "Tests of duration > %s disabled with TKP_MAXTESTDURATION" %
                max_duration)
    return lambda func: func


def requires_test_db_managed():
    """Decorator to disable tests that perform potentially low-level
    database management operations, such as destroying and creating
    databases.

    This decorator checks the environment variables to determine whether
    such tests should be enabled or skipped:

    - If the ``TKP_DBENGINE`` environment variable is set to
      'monetdb', the test is skipped because database management tests
      are not supported for MonetDB and must be tested manually.

    - If the ``TKP_TESTDBMANAGEMENT`` environment variable is set, the
      test is allowed to run.

    - Otherwise, the test is skipped with an appropriate message.

    Returns
    -------
    function
        A decorator that either skips the test or allows it to run.

    """
    if os.environ.get('TKP_DBENGINE', 'postgresql') == 'monetdb':
        return unittest.skip("DB management tests not supported for Monetdb,"
                             "must be tested manually.")

    if os.environ.get("TKP_TESTDBMANAGEMENT", False):
        return lambda func: func
    return unittest.skip("DB management tests disabled, TKP_TESTDBMANAGEMENT"
                         " not set")


def high_ram_requirements():
    """Decorator to disable tests that break Travis due to
    out-of-memory issues.

    This function checks the `TRAVIS` environment variable to determine if the
    tests are running in a Travis CI environment. If so, tests with high RAM
    requirements are skipped with an appropriate message.

    Returns
    -------
    function
        A decorator that either skips the test or allows it to run.

    """
    if os.environ.get("TRAVIS", False):
        return unittest.skip(("High-ram requirement unit-tests disabled on "
                              "Travis"))
    return lambda func: func
