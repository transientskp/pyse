#!/usr/bin/env python
from setuptools import setup, find_packages

install_requires = """
    numpy
    astropy
    scipy
    pytz
    python-casacore
    python-dateutil
    six
    """.split()


tkp_scripts = [
    "scripts/pyse",
    ]

package_list = find_packages(where='.', exclude=['tests'])

setup(
    name="radio-pyse",
    version="0.2",
    packages=package_list,
    scripts=tkp_scripts,
    description="Python Source Extractor",
    author="TKP Discovery WG",
    author_email="discovery@transientskp.org",
    url="http://docs.transientskp.org/",
    install_requires=install_requires,
)
