.. _DevelopersGuideFile:

Developer's Guide
=================

Introduction
------------

Community development of the code is strongly encouraged so please contact the lead developer if you or your team would like to contribute. You are welcome to enquire about the reasoning behind the structure and design of the code with the development team.

Installation
------------

See :ref:`developer_install` for how to install *Bedroc* and the extra dependencies required for the developer tools and documentation.
 
Pre-commit
----------

Run pre-commit before issuing a pull request to the main repository::

    pre-commit run --all-files

Documentation
-------------

.. note::
    Documentation should be consistent with the source code, so any changes to the source code should be reflected in changes to the documentation, if required.

When the doc dependencies have been installed the documentation can be compiled::

    cd docs
    sphinx-apidoc -f -o source ../bedroc

To generate HTML documentation use the appropriate command for your operating system:

Mac/Linux::

    make html

Windows::

    .\make html

To generate PDF documentation, noting that ``latexpdf`` must be available on your system::

    make latexpdf

Documentation is built the appropriately named subdirectory in ``_build``.

Tests
-----

.. note::
    Tests should be consistent with the source code, so any changes to the source code should be reflected in changes to the test suite, if required.

You can confirm that all tests pass by navigating to the root directory of *Bedroc* and running::
    
    pytest
    
Similarly, to test coverage::

    pytest --cov

The percentage test coverage is reported in ``README.md`` but requires manual update.

Add a corresponding test for new features that are developed.