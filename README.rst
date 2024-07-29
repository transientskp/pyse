Python Source Extractor
=======================

This is the Python Source Extractor (PySE) for radio astronomical images.

This project was formerly part of the Transient Detection Pipeline:

https://github.com/transientskp/tkp/


Installation
------------

PySE available on pypi::

    $ pip install radio-pyse

PySE is compatible with Python 3.10 or up.


License
-------

PySE is released under the BSD-2 license.


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

PySE uses |hatch|_ to manage the different environments while development.
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
build setup also uses a dynamic hook to generate the package version
from Git repository release tags.  So to do a local package build, you
need to ensure all Git tags are present.

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

3. If you want to trigger only the build hooks, you can do:

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
.. image:: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml/badge.svg
   :target: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml
