"""Effective sample size versus projection depth for several step sizes (docs figure).

The step-size scan of the "Tuning the step size" section of
good_simulation.rst: the spin-1 transverse-field Ising chain of "Running a
simulation" and of notebooks/transverse_field_ising_chain.ipynb (12 sites,
J = 1, h = h_c = 1.326, JastrowSpin guide trained with NetKet), equilibrated
with one run, then statistics() at each step size beta, recording the
effective sample size n_eff at projection depths tau = m * beta up to
tau_max.  Mean and standard error over independent seeds.

The part below the setup is the code block shown on the page; keep the two in
sync when editing either.

    python plot_neff_vs_tau.py       # writes neff_vs_tau.pdf and .svg
"""
import os

import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygfmc import run, statistics
from pygfmc.hamiltonian import Spin, chain
from pygfmc.walkers import prepare_walkers_random
from pygfmc.wavefunction import JastrowSpin, train_with_netket, make_hilbert
import netket as nk          # after pygfmc, so that everything stays in float32

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- setup: the running example of "Running a simulation" ----------------
L = 12
lattice = chain(L)
h_c = 1.326
H = Spin.create(lattice.edges, S=1.0, J_z=-1.0, h_pm=-h_c / 2)
walkers = prepare_walkers_random(1024, L, jax.random.PRNGKey(0), states=(-1.0, 0.0, 1.0))
guide = JastrowSpin(L, lattice.edges, beta=-0.1, d=0.0)
sampler = nk.sampler.MetropolisLocal(make_hilbert(H.local_hilbert_space, L))
guide = train_with_netket(H, guide, sampler=sampler)
walkers, energy, obs, stats, raw = run(
    walkers, 1000, H, step_size=0.5, guide=guide,
    p_array=[4, 8, 16, 32, 64], rng_key=jax.random.PRNGKey(1),
    warmup_steps=500, verbosity=0,
)

# ---- the code block shown on the page -----------------------------------

tau_max = 128.0                                      # target projection depth
betas   = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]  # step sizes to compare
seeds   = range(2)                                   # independent repetitions
colors  = plt.cm.viridis(np.linspace(0, 0.9, len(betas)))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for beta, color in zip(betas, colors):
    m_max   = round(tau_max / beta)
    m_array = np.unique(np.linspace(1, m_max, 32).round().astype(int))
    n_eff   = np.array([
        statistics(walkers, H, guide=guide,
                   avg_log_weight=stats["avg_log_weight"], step_size=beta,
                   method="cont_time", rng_key=jax.random.PRNGKey(seed),
                   m_array=m_array, quiet=True)[1]["effective_sample_size"]
        for seed in seeds])
    tau = m_array * beta                             # tau = (p/2) * beta = m * beta
    ax.errorbar(tau, n_eff.mean(axis=0),
                yerr=n_eff.std(axis=0, ddof=1) / np.sqrt(len(seeds)),
                fmt="o-", ms=4, capsize=2, color=color, label=f"{beta:g}")

ax.set_xlabel(r"Projection depth $\tau = p\beta/2$")
ax.set_ylabel(r"Effective sample size $n_\mathrm{eff}$")
ax.set_yscale("log")
ax.set_yticks([2**k for k in range(1, 11)], [str(2**k) for k in range(1, 11)])
ax.minorticks_off()
ax.set_xlim(0, tau_max)
ax.legend(title=r"step size $\beta$", loc="upper right", ncol=5)
fig.tight_layout()
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"neff_vs_tau.{ext}"))
print("wrote neff_vs_tau.pdf/.svg")
