.. image:: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml/badge.svg
   :target: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml


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

PySE is compatible with Python 3.10 or up.


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

PySE uses |hatch|_ to manage the different environments during development.
So make sure you have ``hatch`` installed *globally*.  You could either use
your system's package manager to install ``hatch``, or use ``pipx`` to
install as a regular user.  Please ensure that you are using a version
``hatch>=1.10``, otherwise you might encounter `this bug
<https://github.com/pypa/hatch/issues/1395>`_.

In general, to run command in a specific environment managed by
``hatch``, you can do this:

.. code-block:: bash

   $ hatch run <environment>:<command> [--options]

You can see a summary of all the evironments managed by ``hatch`` like
this:

.. code-block:: bash

   $ hatch env show
                Standalone
   ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┓
   ┃ Name    ┃ Type    ┃ Dependencies ┃
   ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━┩
   │ default │ virtual │              │
   ├─────────┼─────────┼──────────────┤
   │ test    │ virtual │ pytest       │
   │         │         │ pytest-cov   │
   ├─────────┼─────────┼──────────────┤
   │ lint    │ virtual │ black        │
   │         │         │ flake8       │
   │         │         │ mypy         │
   │         │         │ ruff         │
   └─────────┴─────────┴──────────────┘

Some common tasks using ``hatch`` are summarised below.

Package builds
++++++++++++++

``hatch`` does package builds in an isolated environment.  The package
build setup also uses a dynamic hook (also known as build hooks) to
generate the package version from Git repository release tags.  So to
do a local package build, you need to ensure all Git tags are present.

1. Fetch all Git release tags locally.

   .. code-block:: bash

      $ git fetch --tags

2. You can now build a distribution (a wheel file and a source
   tarball) locally using:

   .. code-block:: bash

      $ hatch build

   This creates the distribution files in the ``dist/`` directory in
   the project root.

   ::

     $ tree dist/
     dist/
     ├── radio_pyse-0.3.2.dev9+gfb04dc7.d20240729-py3-none-any.whl
     └── radio_pyse-0.3.2.dev9+gfb04dc7.d20240729.tar.gz

3. If you want to trigger only the build hooks (like generating the
   package version), you can do:

   .. code-block:: bash

      $ hatch build --hooks-only

   This is necessary to refresh the version information if you update
   any of the build configuration in ``pyproject.toml``, or if you are
   implementing something that depends on the version, e.g. making a
   new capability available only for a newer version.

Running the test suite
++++++++++++++++++++++

.. code-block:: bash

   $ hatch run test:pytest [tests/test_iwanttorun.py] [-k match_string] [--options]
   $ hatch run test:pytest --no-cov  # to disable coverage

Running formatters and static analysis tools
++++++++++++++++++++++++++++++++++++++++++++

You can run supported linters/formatters (see the environment
definition for ``lint``) like this.

.. code-block:: bash

   $ hatch run lint:mypy [--options]
   $ hatch run lint:flake8 [--options]
   $ hatch run lint:ruff check sourcefinder
   $ hatch run lint:black --check sourcefinder

Note that on first run, ``mypy`` might need to install type-stubs.
You can do that with:

.. code-block:: bash

   $ hatch run lint:mypy --install-type --non-interactive

Running scripts that use PySE
+++++++++++++++++++++++++++++

Normally a regular user would install a released version from PyPI,
but to use a development version you may run such scripts like this:

.. code-block:: bash

   $ hatch run scripts/pyse [--options]

Since the development environment is the default, you don't need to
specify the ``<envrironment>:`` prefix in the run command.


.. |hatch| replace:: ``hatch``
.. _hatch: https://hatch.pypa.io/latest/
