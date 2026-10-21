.. _Examples:

########
Examples
########

The :code:`notebooks/` folder of the `GitHub repository <https://github.com/robin-schaefer/PyGFMC/tree/main/notebooks>`__ contains several Jupyter notebooks that cover the implemented Hamiltonians and wavefunctions and demonstrate different parts of the algorithm.

Bose-Hubbard
============

:download:`bose_hubbard_chain.ipynb <../notebooks/bose_hubbard_chain.ipynb>`

This is the quickstart example of the :ref:`start page <PyGFMC>`.

* **Model:** :class:`~pygfmc.hamiltonian.BoseHubbard` chain
* **Guide:** :class:`~pygfmc.wavefunction.JastrowBoseHubbard`
* **Observables:** none


Transverse-field Ising model
============================

:download:`transverse_field_ising_chain.ipynb <../notebooks/transverse_field_ising_chain.ipynb>`

This is the code discussed in :ref:`Running a simulation <Usage>`.

* **Model:** :class:`~pygfmc.hamiltonian.Spin`-1 transverse-field Ising chain
* **Guide:** :class:`~pygfmc.wavefunction.JastrowSpin`
* **Observables:** :math:`\sum_i S^z_i S^z_{i+1}`, :math:`\sum_i S^x_i`


Spin-1/2 XXZ chain
==================

:download:`xxz_chain.ipynb <../notebooks/xxz_chain.ipynb>`

This is discussed in the :ref:`Paper <Paper>`.

* **Model:** :class:`~pygfmc.hamiltonian.SpinHalf` XXZ chain
* **Guide:** :class:`~pygfmc.wavefunction.JastrowSpinHalf`
* **Observables:** :math:`S^z_\mathrm{tot}`, :math:`S^z_0`, :math:`S^x_\mathrm{tot}`, :math:`S^x_0`, the structure factors :math:`S^{zz}(k)` at all allowed :math:`k` and :math:`S^\pm` at :math:`k=\pi`


n-body model: custom classes
============================

:download:`n_body_model.ipynb <../notebooks/n_body_model.ipynb>`

This is discussed in the appendix of the :ref:`Paper <Paper>`.

* **Model:** user-defined :math:`n`-body Hamiltonian 
* **Guide:** user-defined guide trained with NetKet
* **Observables:** none


XXZ model on the kagome lattice
===============================

:download:`kagome_xxz.ipynb <../notebooks/kagome_xxz.ipynb>`

* **Model:** :class:`~pygfmc.hamiltonian.SpinHalf` XXZ model on the kagome lattice
* **Guide:** :class:`~pygfmc.wavefunction.ClusterCPS` on stars of David
* **Observables:** none


Toric code
==========

:download:`toric_code.ipynb <../notebooks/toric_code.ipynb>`

* **Model:** :class:`~pygfmc.hamiltonian.ToricCode`
* **Guide:** none
* **Observables:** none


Ring exchange on the pyrochlore lattice
=======================================

:download:`ring_model_pyrochlore.ipynb <../notebooks/ring_model_pyrochlore.ipynb>`

Download :download:`pyrochlore.h5 <../notebooks/pyrochlore.h5>` into the same
directory as this notebook. 
It contains the lattice, initial state and references.

* **Model:** :class:`~pygfmc.hamiltonian.PlaquetteModel` on the pyrochlore lattice
* **Guide:** none
* **Observables:** fraction of flippable hexagons
