.. _Installation:

########################
Package and Installation
########################

PyGFMC is a pure-Python package; there is nothing to compile.
It needs Python 3.10 or newer.

:red:`Installation from PyPI`
=============================

.. container:: red

   PyGFMC is not yet available on PyPI. Until then, install it from source as described above.



Installation from source
========================


The source is on GitHub at https://github.com/robin-schaefer/PyGFMC.
Clone it and install from the root of the repository:

.. code-block:: console

    git clone https://github.com/robin-schaefer/PyGFMC.git
    cd PyGFMC
    pip install -e "./codes"             # GFMC only
    pip install -e "./codes[netket]"     # + training guides with NetKet

PyGFMC is not published on PyPI. The source directory containing :code:`codes/pyproject.toml` is required for these commands.


For GPU runs, install the appropriate CUDA build of JAX first: `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

The editable install :code:`-e` makes changes to the source code take effect without reinstalling.
Alternatively, without installing, :code:`codes/` can be added to your :code:`PYTHONPATH`.

Tests
=====

To check the installation:

.. code-block:: console

    python -c "import pygfmc; print(pygfmc.__version__)"

To run the tests, install the :code:`test` extra (pytest) and run pytest from the :code:`tests/` folder:

.. code-block:: console

    pip install -e "./codes[test]"
    cd tests
    pytest core -q                       # core suite (about 50 minutes on one CPU)
    RUN_GFMC_BENCH=1 pytest benchmark -q # benchmarks (about 140 minutes)

One section in the appendix of the :ref:`Paper <Paper>` is dedicated to the tests.


Package setup
=============
PyGFMC has four subpackages:

* :code:`pygfmc.walkers`: the walker population, its initializers and reconfiguration,
* :code:`pygfmc.engine`: the GFMC engine (propagation, estimators, HDF5 storage).
  These functions are also available directly from :code:`pygfmc`,
* :code:`pygfmc.hamiltonian`: Hamiltonians, lattices and observables,
* :code:`pygfmc.wavefunction`: guiding wavefunctions and their training.


The module :code:`pygfmc.protocols` defines the interfaces for a Hamiltonian, an observable or a guide (:class:`~pygfmc.protocols.Operator`, :class:`~pygfmc.protocols.DiagonalOperator`, :class:`~pygfmc.protocols.Guide`), which is useful when implementing your own classes.

.. code-block:: python

    from pygfmc.walkers import Walker, prepare_walkers_random, prepare_walkers_sz
    from pygfmc.engine import run, statistics, evaluate_observables, write_to_HDF5, run_from_HDF5
    from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization
    from pygfmc.wavefunction import JastrowSpinHalf, ClusterCPS, ClusterMLP, train_with_netket


Dependencies
============

The core dependencies are installed automatically:

* `JAX <https://jax.readthedocs.io>`_ (with the matching CUDA build for GPU runs)
* `NumPy <https://numpy.org>`_ and `SciPy <https://scipy.org>`_
* `h5py <https://www.h5py.org>`_ for walker checkpoints and results

:code:`pygfmc[netket]` requires additional dependencies (`NetKet <https://www.netket.org>`_, `optax <https://optax.readthedocs.io>`_ and `Flax <https://flax.readthedocs.io>`_), which are needed only to **train** guides with NetKet within PyGFMC.


Contact
=======

Please reach out for feedback, tips and bugs: `robin_schaefer@fas.harvard.edu <mailto:robin_schaefer@fas.harvard.edu>`_

Bug reports and questions are also welcome as `GitHub issues <https://github.com/robin-schaefer/PyGFMC/issues>`__.
