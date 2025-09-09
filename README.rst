.. image:: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml/badge.svg
   :target: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml

.. image:: https://app.readthedocs.org/projects/pyse/badge/?version=latest
   :alt: Documentation Status
   :target: https://pyse.readthedocs.io/en/latest/?badge=latest


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

`stable <https://pyse.readthedocs.io/en/stable/>`_ and `latest <https://pyse.readthedocs.io/en/latest/>`_.


Installation
------------

PySE has an entry on PyPI:

.. code-block:: bash

    $ pip install radio-pyse

PySE is compatible with Python 3.10 or up.


License
-------

PySE was released under the BSD-2 license.

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
                        Matrices
   ┏━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
   ┃ Name ┃ Type    ┃ Envs        ┃ Dependencies     ┃
   ┡━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
   │ test │ virtual │ test.py3.10 │ pytest           │
   │      │         │ test.py3.11 │ pytest-cov       │
   │      │         │ test.py3.12 │                  │
   │      │         │ test.py3.13 │                  │
   └──────┴─────────┴─────────────┴──────────────────┘

As shown above, the test environments also define a matrix to cover
multiple Python versions.  So we can run the whole test matrix locally
with:

.. code-block:: bash

   $ hatch run test:pytest

Instead if you want to run only a subset, you can limit the python
versions like this:

.. code-block:: bash

   $ hatch run +py=3.13 test:pytest # +py and +python are equivalent
   $ hatch run +python=3.13 test:pytest
   $ hatch run +python=3.12,3.13 test:pytest

For more options, see ``hatch env run -h``.

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

   $ hatch run pyse [--options]

Since the development environment is the default, you don't need to
specify the ``<envrironment>:`` prefix in the run command.


.. |hatch| replace:: ``hatch``
.. _hatch: https://hatch.pypa.io/latest/
