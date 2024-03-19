Python Source Extractor
=======================

This is the Python Source Extractor (PySE) for extraction and measurement of sources in radio astronomical images, or,
more generally, images from aperture synthesis. This means that PySE has been tested thoroughly on images with correlated noise.
PySE was primarily designed for the `LOFAR Transients Key Project (TKP) <https://transientskp.org/>`_, i.e. targeting the
detection of transient sources, which will be mostly unresolved.

Recent development - 2020 - 2023 - has been funded by grant 27019G06 from the Netherlands eScience Center
for `PADRE <https://www.esciencecenter.nl/projects/the-petaflop-aartfaac-data-reduction-engine-padre/>`_.

Current development is internally funded by The Netherlands eScience Center through the Software Sustainability
Project "PySE".

The code in this repo was formerly part of the `LOFAR Transients Pipeline (TraP) <https://github.com/transientskp/tkp/>`_
repo.
Compatibility with TraP is maintained, i.e. the output of PySE is suitable for futher processing in TraP.
Also, PySE is still included and regularly updated in Trap.
A reason for starting this repo is to offer PySE as a standalone application.

Documentation
-------------

Currently only offered as `part of the TraP documentation <https://tkp.readthedocs.io/en/latest/tools/pyse.html>`_.

Installation
------------

PySE has an entry on PyPI:

.. code-block:: bash

    $ pip install radio-pyse

but this is outdated. This year (2024), we will update PySE's record on PyPI.

The recommended way of installing PySE is currently:

.. code-block:: bash

    $ git clone git@github.com:transientskp/pyse.git
    $ cd pyse
    $ pip install -e .

Original development of PySE was done in Python 2, but PySE is presently only Python 3 compatible.


License
-------

PySE was released under the BSD-2 license.
This is still the case for the master branch.

However, some of the fast branches of PySE make use of `SEP <https://github.com/kbarbary/sep>`_.
Please have a look at the `SEP license info <https://github.com/kbarbary/sep?tab=readme-ov-file#license>`_.
This means that the license for PySE as a whole, for some of the fast branches, is LGPLv3.
Only sourcefinder/image.py makes use of SEP, so the other Python modules have a BSD-2 license.

We will be including C source code for least squares fitting into one or more of the fast branches.
This fitting code was written decades ago as part of the `Groningen Image Processing System (GIPSY) <https://www.astro.rug.nl/~gipsy/>`_.
GIPSY has an `ASCL record <https://ascl.net/1109.018>`_ and a LGPL2+ license.

Authors
-------

The list of authors, sorted by the number of commits:

- Hanno Spreeuw
- John Swinbank
- Gijs Molenaar
- Tim Staley
- Evert Rol
- John Sanders
- Bart Scheers
- Mark Kuiack


Developer information
---------------------

.. image:: https://github.com/transientskp/pyse/actions/workflows/ci.yaml/badge.svg
   :target: https://github.com/transientskp/pyse/actions
