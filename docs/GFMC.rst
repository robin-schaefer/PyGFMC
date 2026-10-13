.. _GFMC:

###################
About the algorithm
###################

Green's Function Monte Carlo (GFMC) projects a walker ensemble onto the ground state of a sign-problem-free Hamiltonian :math:`H`,

.. math::

    \vert\psi_0\rangle \;\propto\; \lim_{n\to\infty}\,e^{-n\beta H}\,\vert\psi\rangle\,.

A model is sign-problem-free if all off-diagonal matrix elements of :math:`H` in the chosen basis are non-positive,

.. math::

    \langle x'\vert H\vert x\rangle \;\le\; 0 \qquad \text{for all } x' \neq x\,.

The core idea is the following:

* **Walkers:** Within an ensemble of fixed size, each walker represents a basis state :math:`x` and carries a weight.
* **Propagation:** The walker population is propagated by interpreting the Hamiltonian as a stochastic process.
  A walker at :math:`x` moves to :math:`x'` with a probability proportional to :math:`-\langle x'\vert H\vert x\rangle \ge 0`.
  Each walker carries a weight that is updated with each transition.
  PyGFMC propagates either in continuous imaginary time (:code:`"cont_time"`, the default) or in discrete sub-steps of :math:`\lambda - H` (:code:`"single_step"` and :code:`"multi_step"`).
  :math:`\lambda` is a positive constant.
* **Reconfiguration:** At fixed intervals the walkers are resampled according to the weights accumulated during propagation.
  This removes low-weight walkers and duplicates high-weight ones.
* **Importance sampling:** An (optional) guiding wavefunction :math:`\psi_T > 0` modifies the transition probabilities and biases the moves toward the ansatz,

  .. math::

      -\langle x'\vert H\vert x\rangle \;\longrightarrow\;
      -\frac{\psi_T(x')}{\psi_T(x)}\,\langle x'\vert H\vert x\rangle\,.

  This drastically improves convergence for problems that are hard to equilibrate.
* **Estimators:** Besides the energy, which is available through the reconfiguration, the walker ensemble also gives access to observables.
  While diagonal observables are accessible with minimal additional cost, off-diagonal observables require launching auxiliary Markov chains, which increases the computational cost significantly.

A detailed description of the algorithm can be found in the :ref:`paper <Paper>`.
