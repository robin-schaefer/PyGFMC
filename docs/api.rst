.. _API:

#############
API reference
#############

Every public function and class in the four subpackages and the pygfmc.protocols module, with its docstring.
The entries are generated from the source code, so they always match the installed version.

.. contents:: Contents
    :local:
    :depth: 1


.. _api-protocols:

:code:`pygfmc.protocols`
========================

The interfaces of a Hamiltonian or off-diagonal observable, a diagonal observable and a guiding wavefunction, as :class:`typing.Protocol` classes.
A class satisfies one by having the members; it does not inherit from it.
They are also available directly from :code:`pygfmc`.

.. currentmodule:: pygfmc.protocols

.. autosummary::

    Operator
    DiagonalOperator
    Guide

.. autoclass:: Operator
    :members:
.. autoclass:: DiagonalOperator
    :members:
.. autoclass:: Guide
    :members:


.. _api-walkers:

:code:`pygfmc.walkers`
======================

The walker population, its initializers and reconfiguration.

.. currentmodule:: pygfmc.walkers

.. autosummary::

    Walker
    prepare_walkers_random
    prepare_walkers_sz
    prepare_walkers_from_configs
    reconfiguration

.. autoclass:: Walker
    :exclude-members: config, log_weight, e_loc
.. autofunction:: prepare_walkers_random
.. autofunction:: prepare_walkers_sz
.. autofunction:: prepare_walkers_from_configs
.. autofunction:: reconfiguration


.. _api-engine:

:code:`pygfmc.engine`
=====================

The GFMC engine: propagation, estimators and HDF5 storage.
These functions are also available directly from :code:`pygfmc`.

.. currentmodule:: pygfmc.engine

.. autosummary::

    run
    run_gfmc
    statistics
    evaluate_observables
    write_to_HDF5
    run_from_HDF5
    load_raw_from_HDF5
    check_run_compatible
    resume_from_HDF5

.. autofunction:: run
.. autofunction:: run_gfmc
.. autofunction:: statistics
.. autofunction:: evaluate_observables
.. autofunction:: write_to_HDF5
.. autofunction:: run_from_HDF5
.. autofunction:: load_raw_from_HDF5
.. autofunction:: check_run_compatible
.. autofunction:: resume_from_HDF5

.. _api-engine-advanced:

Building blocks
---------------

The functions the drivers are made of, for custom production loops and diagnostics.
They are imported from their modules, e.g. :code:`from pygfmc.engine.evolution import evolve_walkers`.

.. currentmodule:: pygfmc.engine.evolution

.. autosummary::

    evolve_walkers
    evolve_carried
    forward_walk_operator
    require_lam
    require_operator_protocol
    require_guide_protocol

.. autofunction:: evolve_walkers
.. autofunction:: evolve_carried
.. autofunction:: forward_walk_operator
.. autofunction:: require_lam
.. autofunction:: require_operator_protocol
.. autofunction:: require_guide_protocol

.. currentmodule:: pygfmc.engine.estimators

.. autosummary::

    estimate_average_log_branch_weight
    check_off_diag_operator_positivity
    count_unique_configs
    autocorrelation

.. autofunction:: estimate_average_log_branch_weight
.. autofunction:: check_off_diag_operator_positivity
.. autofunction:: count_unique_configs
.. autofunction:: autocorrelation

.. currentmodule:: pygfmc.engine.multi_gpu

.. autosummary::

    run_gfmc_pmap
    reconfiguration_global
    shard_walkers

.. autofunction:: run_gfmc_pmap
.. autofunction:: reconfiguration_global
.. autofunction:: shard_walkers


.. _api-hamiltonian:

:code:`pygfmc.hamiltonian`
==========================

Hamiltonians, lattices, measurement operators and exact diagonalization.

.. currentmodule:: pygfmc.hamiltonian

.. autosummary::

    SpinHalf
    Spin
    BoseHubbard
    ToricCode
    PlaquetteModel
    StructureFactor
    SzSzStructureFactor
    is_valid_k
    Lattice
    chain
    pyrochlore
    kagome
    exact_diagonalization

Hamiltonians
------------

.. autoclass:: SpinHalf
.. autoclass:: Spin
.. autoclass:: BoseHubbard
.. autoclass:: ToricCode
.. autoclass:: PlaquetteModel

.. _api-observables:

Measurement operators
---------------------

.. autofunction:: StructureFactor
.. autoclass:: SzSzStructureFactor
.. autofunction:: is_valid_k

.. _api-lattices:

Lattices
--------

.. autoclass:: Lattice
.. autofunction:: chain
.. autofunction:: pyrochlore
.. autofunction:: kagome

Exact diagonalization
---------------------

.. autofunction:: exact_diagonalization


.. _api-wavefunction:

:code:`pygfmc.wavefunction`
===========================

Guiding wavefunctions and their training. :code:`train_with_netket` needs the optional extra netket: :code:`pip install "pygfmc[netket]"`, or from source :code:`pip install -e "./codes[netket]"` in the root of the cloned `repository <https://github.com/robin-schaefer/PyGFMC>`__ (see :ref:`Installation`).

.. currentmodule:: pygfmc.wavefunction

.. autosummary::

    JastrowSpinHalf
    ClusterCPS
    ClusterMLP
    ExactGroundState
    JastrowBoseHubbard
    JastrowSpin
    train_with_netket
    make_hilbert

.. autoclass:: JastrowSpinHalf
.. autoclass:: ClusterCPS
.. autoclass:: ClusterMLP
.. autoclass:: ExactGroundState
.. autoclass:: JastrowBoseHubbard
.. autoclass:: JastrowSpin
.. autofunction:: train_with_netket
.. autofunction:: make_hilbert
