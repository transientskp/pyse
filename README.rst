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


.. |hatch| replace:: ``hatch``
.. _hatch: https://hatch.pypa.io/latest/
.. image:: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml/badge.svg
   :target: https://github.com/transientskp/pyse/actions/workflows/python-tests.yml
