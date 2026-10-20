# Core tests

The core suite checks the building blocks of PyGFMC against exact results:
that every Hamiltonian's rate tensor reproduces the true matrix, that the
estimators, the forward-walking bookkeeping and the HDF5 storage are exact,
that the three propagators agree, and that the guides, the multi-device path
and user-defined operators and guides work end to end. It has 587 tests, runs in
about 50 minutes on one CPU and should be run after every change.

```bash
cd tests
pytest core -q                    # the whole suite
pytest core -q -m "not slow"      # without the 13 cases marked slow (about 20-25 min)
pytest core/test_io_hdf5.py -q    # one file
```

The NetKet training tests (`test_guides.py::test_train_with_netket_generic`,
`test_jastrow_spin.py::test_train_with_netket_keeps_the_form`,
`test_jastrow_bose_hubbard.py::test_train_with_netket_keeps_the_form`) are skipped when NetKet is not installed or does not import.
Everything else needs only the core dependencies and `pytest`, plus the repository's `paper/` and `notebooks/` folders for `test_repository_consistency.py`.

`test_benchmark_contracts.py` also validates all committed external-reference
contracts without running GFMC or external solvers. Negative tests ensure zero
references, missing outputs/windows, nonfinite results, stale metadata,
normalization changes, reference hashes, RK windows and observable signs cannot
silently bypass benchmark assertions.

## Helper modules

| File | Content |
|---|---|
| `reference.py` | Independent per-class Hamiltonian matrices built from Kronecker products of single-site operators (`reference_matrix`, `reference_dense`, `reference_ground_state`, `reference_basis_index`) and a small pyrochlore geometry builder. They share no code with `get_rates`, which is what makes the matrix comparisons meaningful. |
| `rates.py` | `rebuild_matrix`: the operator matrix assembled from `get_rates` alone over a full basis (checking `generate_state` and that no nonzero-rate slot leaves the basis), and `all_configs`. Tests assert Hermiticity and signs on what the propagators sample with it. |
| `guides.py` | The L = 6 XXZ ring and the four spin-1/2 guides shared by `test_guides.py` and `test_guided_methods.py` (`xxz_ring`, `build_guide`), and `log_ratios_from_log_psi`, the reference for a guide's fast ratios used by the Jastrow tests. |
| `conftest.py` | The `h` (XXZ ring) and `e0` (its exact energy) fixtures of `test_guides.py` and `test_guided_methods.py`. |
| `walkers.py` | Deterministic walker initializers used to set up tests (`balanced_walkers`, `neel_walkers`). |
| `../conftest.py` | Puts `../codes` (which holds `pygfmc`) on the import path, scopes `jax_enable_x64` to the test that enables it, provides the `x64` fixture (`with x64(): ...`, the one way tests switch to float64), disables NetKet's experimental sharding so several NetKet tests can share one process, and registers the `slow` marker (tests over about a minute on a CPU). |

## All tests

"Cases" counts the parametrized instances pytest collects.

### `test_benchmark_contracts.py` (61 cases)

Fast negative tests for the benchmark scoring and the committed external-reference contracts, without running GFMC or an external solver.

| Test | Cases | What it checks |
|---|---|---|
| `test_zero_reference_is_asserted` | 1 | A zero reference is still compared: a GFMC value of 1 against 0 fails. |
| `test_missing_observable_fails` | 2 | A reference without an output, or an output without a reference, fails. |
| `test_small_zero_reference_error_passes` | 1 | A small deviation from a zero reference passes the absolute bound. |
| `test_reference_exclusion_must_be_explicit_and_current` | 1 | An unavailable reference needs an explicit reason; an exclusion for a reference that exists fails as obsolete. |
| `test_invalid_observable_results_fail` | 2 | No windows or a NaN estimate fails. |
| `test_missing_projection_window_fails` | 1 | An output missing one of the requested p windows fails. |
| `test_committed_reference_contract` | 41 | Every committed reference file matches its model's couplings, geometry, sector, conventions, measurement-protocol hash and pinned SHA-256. |
| `test_stale_reference_contract_is_rejected` | 3 | Changed couplings, initial sector or edge order are rejected. |
| `test_reference_bytes_and_normalization_are_pinned` | 2 | A changed file hash or normalization string in the manifest is rejected. |
| `test_rk_assertions_check_all_windows_and_signed_observables` | 4 | The RK-point benchmark passes on the reference values and fails on a nonzero energy at the smallest window or a sign-flipped observable. |
| `test_missing_requested_reference_fails_not_skips` | 3 | With the benchmark requested, a missing reference file fails each of the three suites instead of skipping. |

### `test_diag_offdiag_consistency.py` (4 cases)

The same diagonal operator through both estimator pipelines, and the forward walk's edge cases: m = 0, a protocol-only operator and the jit cache of a continuation.

| Test | Cases | What it checks |
|---|---|---|
| `test_diagonal_op_diag_vs_offdiag_consistency` | 1 | Ising energy via the diagonal and off-diagonal estimators matches ED within 0.01 and the two agree within 1e-5 (float32 rounding). |
| `test_forward_walk_at_m_zero_is_the_plain_mean` | 1 | At p = 1 (m = 0) the raw off-diagonal column equals the diagonal pipeline's m = 0 value at each measured step, with a zero log norm; it differs from the p = 4 column, and the p = 1 estimates agree within 0.01. |
| `test_protocol_only_operator_measures_off_diagonally` | 1 | A NamedTuple with only `M_total == 1` and `get_rates` runs as an off-diagonal operator; the identity reads exactly 1. |
| `test_continuing_with_the_measured_shift_compiles_nothing_new` | 1 | A continuation passing `stats['avg_log_weight']` adds no entry to the driver's jit caches. |

### `test_diag_offdiag_window_alignment.py` (1 case)

Diagonal and off-diagonal estimators of the same SzSz agree to float64 precision.

| Test | Cases | What it checks |
|---|---|---|
| `test_diag_offdiag_pipelines_agree_tfim6` | 1 | SzSz(4,5) on the L=6 TFIM: diag and off-diag estimates agree to 1e-9 at every p. |

### `test_estimator_window_reuse.py` (2 cases)

Check shared window selection against explicit scalar weighted sums.

| Test | Cases | What it checks |
|---|---|---|
| `test_multiple_operators_on_irregular_measurement_grid` | 2 | `_off_diagonal_estimates` on three operators measured at irregular steps, with and without a trimmed tail, equals the explicitly summed weighted ratios of every window. |

### `test_exact_diagonalization.py` (8 cases)

Generic `exact_diagonalization` against the independent reference matrices.

| Test | Cases | What it checks |
|---|---|---|
| `test_low_lying_spectrum_is_sorted_and_matches_dense` | 5 | The requested states come back sorted and match the dense reference. |
| `test_observables_match_a_direct_evaluation` | 1 | Diagonal and off-diagonal observables against a dense computation. |
| `test_runs_in_float64_and_restores_the_caller_precision` | 1 | Double precision inside, and `jax_enable_x64` restored even on a raise. |
| `test_input_validation` | 1 | Bad local spaces, sizes, state counts and contradictory N/d raise. |

### `test_exact_ground_state.py` (20 cases)

Tests for the ExactGroundState guiding wavefunction.

| Test | Cases | What it checks |
|---|---|---|
| `test_bijection` | 4 | The guide's basis index maps the full configuration space onto 0..dim-1. |
| `test_log_diff` | 3 | log_diff_from_configs matches the brute-force ratio on rate-carrying moves. |
| `test_exact_guide_local_energy_on_every_supported_configuration` | 4 | Exhaustive local energies and zero variance on the nonzero exact-wavefunction support for SpinHalf (including a fixed sector), Spin and BoseHubbard. |
| `test_zero_variance_integration` | 1 | With psi_T = psi_0 the GFMC energy collapses onto E_0 within float32 noise (100 warm-up steps suffice: e_L is constant). |
| `test_off_diag_observable_is_exact_at_every_p` | 1 | With psi_T = psi_0 the forward-walked estimate is unbiased and flat in p. |
| `test_hdf5_roundtrip` | 1 | save/load reproduces the amplitudes and writes the metadata attributes. |
| `test_hdf5_load_rejects_wrong_family` | 1 | Loading into a different Hamiltonian family raises. |
| `test_plaquette_works_without_config_to_basis_index` | 1 | The index comes from local_hilbert_space, so PlaquetteModel is supported. |
| `test_basis_values_must_be_the_hamiltonians_own` | 1 | A reordered `local_hilbert_space` is accepted; different, repeated or non-finite values are rejected. |
| `test_nonzero_transition_leaving_the_basis_is_an_error` | 1 | A transition of nonzero amplitude to a state outside the basis raises instead of being dropped. |
| `test_non_hermitian_operator_is_rejected` | 1 | The operator matrix is checked for finite entries and Hermiticity before the eigensolve. |
| `test_allocation_guard_refuses_a_huge_dense_solve` | 1 | The estimated sparse or dense allocation is checked before anything is built; `number_of_states` goes through the integer validator. |

### `test_get_rates_all_classes.py` (13 cases)

Every Hamiltonian class: `get_rates` must reproduce the full matrix (rebuilt by `core/rates.py`).

| Test | Cases | What it checks |
|---|---|---|
| `test_get_rates` | 5 | The matrix rebuilt from get_rates equals the reference, over the full basis. |
| `test_diagonal_weight_matches_get_rates` | 5 | The fast `diagonal_weight(c)` must equal `get_rates(c)[0][0]`. |
| `test_toric_per_object_couplings` | 1 | Per-star/plaquette/site couplings: scalar == uniform array; a non-uniform model rebuilds the reference and its rate matrix is symmetric. |
| `test_move_reproduces_generate_state` | 2 | `Spin.move` and `BoseHubbard.move` name the two sites and increments of every transition, and adding them gives exactly `generate_state`. |

### `test_gfmc_multidevice.py` (12 cases, marked `slow`)

run_gfmc_pmap on four fake CPU devices: `return_raw=False` and strides stay on pmap, `off_diag_stride` gating (same measured steps as `run_gfmc`, stride rows a subset of stride 1) and resumability.

| Test | Cases | What it checks |
|---|---|---|
| `test_multidevice` | 12 | One worker check ran on 4 devices and passed. |

### `test_golden_outputs.py` (47 cases)

Golden-output regression: every propagator, for every Hamiltonian class and every guide that accepts it, must reproduce stored results bit for bit (CPU, same JAX version; regenerate with `PYGFMC_WRITE_GOLDEN=1`). The plaquette walkers start in the largest connected sector of a 4 x 3 square-lattice ring exchange, whose exact ground state is the sector's Perron-Frobenius vector; a full-space diagonalization would return an arbitrary vector of the degenerate space.

| Test | Cases | What it checks |
|---|---|---|
| `test_golden_output_is_reproduced_bit_for_bit` | 47 | Configurations, weights, energies, observables, raw series and stats of a short run match `reference_data/golden_cpu.npz` exactly: `cont_time`/`multi_step`/`single_step` for SpinHalf with no guide, JastrowSpinHalf, ClusterCPS, ClusterMLP and ExactGroundState (15), Spin-1 and BoseHubbard with no guide, their Jastrow and ExactGroundState (9 each), ToricCode and PlaquetteModel with no guide and ExactGroundState (6 each), each run measuring the class's diagonal and off-diagonal operators (the structure factors included); plus the gated path (`return_raw=False`, a stride and a batch size) and `statistics()`. |

### `test_guided_methods.py` (11 cases)

Guided GFMC reaches the exact ground state under `multi_step` and `cont_time` (`test_guides.py` covers `single_step`).

| Test | Cases | What it checks |
|---|---|---|
| `test_spin_half_guides_under_every_method` | 8 | JastrowSpinHalf, ClusterCPS, ClusterMLP and ExactGroundState reach E0 within 0.02/site under `multi_step` and `cont_time`, and the deeper window is worse by at most 0.01/site. |
| `test_jastrow_bose_hubbard_matches_ed` | 2 | A guided Bose-Hubbard chain at unit filling reaches E0 within 0.02/site under `cont_time` and `multi_step`. |
| `test_gutzwiller_guide_is_not_vacuous` | 1 | The guided chain differs from the free one with the same key. |

### `test_guides.py` (18 cases)

Guide tests for the four documented wavefunctions.

| Test | Cases | What it checks |
|---|---|---|
| `test_initialization` | 4 | The guide builds and evaluates amplitudes through whatever routines it provides. |
| `test_hdf5_roundtrip` | 3 | save / load reproduces the amplitudes of the trainable guides (`ExactGroundState` in `test_exact_ground_state.py`). |
| `test_train_with_netket_generic` | 3 | The generic trainer returns a usable float32 guide of the same class, with moved parameters (single-site sampler, since h_x breaks S^z_tot). |
| `test_default_sampler_is_refused_when_the_sum_is_not_conserved` | 1 | The default MetropolisExchange conserves S^z_tot; h_x or J_x != J_y breaks it, so it raises. |
| `test_gfmc_energy_convergence` | 4 | A short `single_step` run reproduces E0 within 0.02/site, and the deepest window is worse by at most 0.01/site. |
| `test_log_diff_consistency` | 3 | log_diff_from_operator, log_diff_from_configs and log_psi agree (ExactGroundState has only one). |

### `test_hamiltonian_fixes.py` (32 cases)

Input checks and small cases of the `pygfmc.hamiltonian` subpackage.

| Test | Cases | What it checks |
|---|---|---|
| `test_models_reject_invalid_edges` | 24 | `SpinHalf`, `Spin` and `BoseHubbard` reject malformed edge lists (one-dimensional, three columns, negative, non-integer, NaN, infinite, out of int64 range) with a `ValueError` naming the edges. |
| `test_edge_validation_preserves_valid_inputs` | 4 | `edges_array` accepts empty lists and arrays, float-valued integer pairs and unsigned arrays unchanged. |
| `test_spinhalf_positive_jpm_raises` | 1 | J_pm > 0 raises at construction, uniform and per-edge; sign_check=False builds it. |
| `test_spinhalf_sign_problem_free_models_stay_silent` | 1 | Sign-problem-free models and sign_check=False measurement operators do not warn. |
| `test_exact_ground_state_energy_on_tiny_bases` | 1 | Exact diagonalization works on dim-2 and dim-4 bases. |
| `test_plaquette_rejects_odd_length` | 1 | Odd plaquette lengths raise; even ones are accepted. |

### `test_identity_forward_walk.py` (2 cases)

Identity-operator checks of the forward-walking pure estimator; the forward walk is jitted as in the production scan.

| Test | Cases | What it checks |
|---|---|---|
| `test_identity_shared_keys_is_exactly_one` | 1 | Shared shadow keys: <identity>_p == 1 to 1e-9 (float64) for every p and stride. |
| `test_identity_run_gfmc_correlated_multistep` | 1 | Through the full run_gfmc path, <identity>_p == 1 to 1e-9 at ss=8 multi_step and ss=1. |

### `test_io_hdf5.py` (44 cases, 1 marked `slow`)

write_to_HDF5 / run_from_HDF5 / evaluate_observables — the per-run storage format.

| Test | Cases | What it checks |
|---|---|---|
| `test_run_groups_are_zero_based_and_self_contained` | 1 | Runs land in run_0, run_1, ... each self-contained; /raw is one shared series. |
| `test_all_stats_entries_are_stored` | 1 | Nothing from stats is dropped -- only routed by the parameters/ prefix. |
| `test_guide_is_recorded_with_its_path` | 1 | The guide class and a has_guide flag are stored with the run. |
| `test_resume_advances_the_stream_and_skips_the_warmup` | 1 | Resume skips the warm-up and starts on the key the previous run ended on. |
| `test_external_walker_file` | 1 | walker_filename stores the ensemble in a separate file that resume finds. |
| `test_resume_without_a_stored_ensemble_is_refused` | 1 | A file written without walkers cannot be resumed. |
| `test_must_match_mismatch_is_refused` | 6 | Changing a MUST_MATCH parameter is refused on append. |
| `test_may_differ_is_allowed` | 3 | Changing a MAY_DIFFER parameter is accepted. |
| `test_operators_are_compared_by_name_not_count` | 1 | Same count, different operators -> identical shapes, different meaning. |
| `test_write_is_atomic_on_mismatch` | 1 | A refused append must leave the file exactly as it was. |
| `test_rng_key_replay_is_refused` | 1 | Starting again on a key an earlier run started from (a replay) is refused; continuing is not. |
| `test_no_raw_means_no_rewindowing` | 1 | Without raw there is no /raw group and load_raw_from_HDF5 raises. |
| `test_fingerprint_is_stable_and_content_sensitive` | 1 | Equal contents give equal digests; any content change changes the digest. |
| `test_resume_verifies_the_objects_it_is_given` | 1 | The check belongs at resume time, not after the run has been paid for. |
| `test_fixed_parameters_live_once_at_the_file_level` | 1 | MUST_MATCH parameters are stored once at the top, per-run ones inside each run. |
| `test_run_from_HDF5_continues_the_chain` | 1 | One call: verify, load, run without warmup, same tuple as run_gfmc. |
| `test_run_from_HDF5_refuses_a_different_calculation` | 1 | Changed objects are refused by hash, and the propagator cannot be overridden. |
| `test_raw_is_one_continuous_series` | 1 | Runs append into a single /raw, and each run records its own slice. |
| `test_split_run_matches_contiguous_for_equal_and_unequal_pieces` (marked `slow`) | 1 | 600 steps as one run, four equal and three unequal pieces give bit-identical /raw and estimates. |
| `test_split_run_with_off_diag_stride` | 1 | With a stride, a split run keeps the main chain bit-identical and every row labeled with its true step. |
| `test_relaunch_with_changed_hashes_is_refused` | 1 | Same names, changed contents: each change is refused by its own hash, before any stepping. |
| `test_diag_lead_in_matches_the_coherent_rows` | 1 | raw['additional_diag'] equals the next m_max diag rows of an uninterrupted run, without leaking into the run. |
| `test_stats_is_required` | 1 | Without parameters/ there is nothing to validate or continue from. |
| `test_estimator_never_reads_the_invalid_leading_diag_rows` | 1 | Poisoning the zero-filled leading diag rows (below row p) leaves every estimate unchanged. |
| `test_evaluate_observables_reproduces_the_run_it_came_from` | 1 | evaluate_observables(raw) with no further arguments reproduces the run's numbers exactly. |
| `test_load_raw_from_HDF5_round_trips_the_in_memory_raw` | 1 | load_raw_from_HDF5 returns raw under run_gfmc's keys (plus grid metadata) and estimates identically. |
| `test_evaluate_observables_rejects_a_grid_the_operators_are_not_on` | 1 | A p grid of the wrong length (or none) raises; unnamed dense series fall back to positional names. |
| `test_evaluate_observables_energy_at_windows_the_run_never_used` | 1 | A different grid re-windows the energy and leaves the operators alone. |
| `test_jackknife_two_blocks_equals_the_half_chain_split` | 2 | n_blocks=2 jackknife error equals \|x_first_half - x_second_half\| / 2. |
| `test_jackknife_block_count_is_validated` | 1 | Too few usable blocks raise; a small block count warns. |
| `test_mixed_width_diagonal_operators_share_a_run` | 1 | A scalar and an n_k-valued diagonal operator share a run. |
| `test_malformed_measurement_steps_are_rejected` | 1 | `off_diag_steps` must be present, one-dimensional, one per row, integer-valued, strictly increasing and within the chain. |

### `test_jastrow_bose_hubbard.py` (45 cases)

JastrowBoseHubbard: the fast ratio routine, which never builds successor configurations, against `log_psi` on the successors `BoseHubbard.get_rates` returns, for every parameter form; plus storage and NetKet training.

| Test | Cases | What it checks |
|---|---|---|
| `test_fast_ratio_matches_log_psi` | 16 | Every block of the transition layout (hopping, pair term, single-site creation/annihilation) agrees with `log_psi`, for tied / per-pair / per-site / absent `beta`, `h`, `d`. |
| `test_cache_tables_reproduce_the_inline_ratios` | 16 | `build_cache` holds the walker-independent transition tables and pair weights for every model; the cached ratios equal the on-the-fly ones bit for bit, and the cache is memoised per operator. |
| `test_incremental_state_follows_the_moves` | 4 | The incremental neighbor sums, updated through 200 random moves with `BoseHubbard.move`, stay within float rounding of a rebuild, and the ratios from them match the plain ones. |
| `test_incremental_update_without_pairs_is_a_no_op` | 1 | A guide built with `pairs=[]` keeps a zero state through `update_state`. |
| `test_wrong_coupling_is_detected` | 1 | Non-vacuity: flipping the sign of `beta` breaks the agreement. |
| `test_parameters_keep_the_form_they_were_given` | 1 | A scalar stays one tied parameter and `None` stays absent among the trainable leaves. |
| `test_bad_shapes_raise` | 1 | Wrong-length arrays and `beta=None` raise. |
| `test_save_load_and_pytree` | 4 | save / load and tree_flatten / tree_unflatten reproduce the amplitudes and the parameter form. |
| `test_train_with_netket_keeps_the_form` | 1 | NetKet training with an explicit `MetropolisLocal` sampler lowers the energy, keeps the form and learns `d > 0`. |

### `test_jastrow_spin.py` (25 cases)

JastrowSpin: the fast ratio routine against `log_psi` on the successors `Spin.get_rates` returns, for spin 1 and 3/2 and every parameter form, plus a GFMC run and NetKet training.

| Test | Cases | What it checks |
|---|---|---|
| `test_fast_ratio_matches_log_psi` | 6 | Every block of the transition layout (S+S-, S+S+, single-site S±) agrees with `log_psi`, for S = 1, 3/2 and tied / per-pair / per-site parameters. |
| `test_cache_tables_reproduce_the_inline_ratios` | 3 | `build_cache` holds the walker-independent transition tables and pair couplings; the cached ratios equal the on-the-fly ones bit for bit, and the cache is memoised per operator. |
| `test_incremental_state_follows_the_moves` | 3 | The incremental neighbor sums, updated through 200 random moves with `Spin.move`, stay within float rounding of a rebuild, and the ratios from them match the plain ones; `move` agrees with `generate_state`. |
| `test_incremental_mode_does_not_depend_on_the_output_chunking` | 1 | In the incremental mode the same seed gives bit-identical energies, series and walkers for `number_of_outputs` 3 and 30, and a run split over two HDF5 blocks reproduces the contiguous run across the refresh at chain step 100. |
| `test_incremental_update_without_pairs_is_a_no_op` | 1 | A guide built with `pairs=[]` keeps a zero state through `update_state`. |
| `test_float64_parameters_round_trip_exactly` | 1 | Under `jax_enable_x64` float64 parameters are stored and loaded without a float32 cast. |
| `test_forbidden_transitions_keep_their_zero_rate_under_a_huge_field` | 1 | A forbidden move keeps its exact zero rate when the guide ratio overflows (no `0 * inf`); an overflow on an allowed move is not hidden. |
| `test_malformed_pair_lists_are_rejected` | 1 | Guide pair lists get the edge checks: shape, integer values, range. |
| `test_wrong_coupling_is_detected` | 1 | Non-vacuity: flipping the sign of `d` breaks the agreement. |
| `test_parameters_keep_the_form_they_were_given` | 1 | A scalar stays one tied parameter and `None` stays absent among the trainable leaves. |
| `test_save_load_and_pytree` | 3 | save / load and tree_flatten / tree_unflatten reproduce the amplitudes and the parameter form. |
| `test_gfmc_spin1_tfim_matches_ed` | 2 | A short guided run on the L=6 spin-1 transverse-field Ising chain, with rebuilt and with incremental neighbor sums, reproduces E0 within 0.01/site. |
| `test_train_with_netket_keeps_the_form` | 1 | NetKet training on PyGFMC's spin-1 encoding (-1, 0, 1) keeps the parameter form and moves the parameters. |

### `test_lam_contract.py` (15 cases)

The shift `lam`: a keyword the discrete methods require and `cont_time` ignores.

| Test | Cases | What it checks |
|---|---|---|
| `test_require_lam` | 1 | `cont_time` returns None whatever is passed; the discrete methods need a float; an unknown method raises. |
| `test_discrete_methods_without_lam_raise_before_any_work` | 10 | `run`, `run_gfmc`, `statistics`, `evolve_walkers` and `estimate_average_log_branch_weight` refuse a discrete method without `lam`, with one message. |
| `test_cont_time_ignores_lam_bit_for_bit` | 1 | `lam=None`, `5.0` and `-3.0` give identical continuous-time runs; the stats record the value as passed. |
| `test_discrete_lam_is_recorded_and_changes_the_chain` | 1 | For a discrete method `lam` is stored and a different value gives a different chain. |
| `test_cont_time_file_resumes_through_run_from_HDF5` | 1 | A continuous-time chain stores `lam` as NaN and continues bit-identically. |
| `test_lam_stays_a_keyword` | 1 | A `lam` placed positionally after the Hamiltonian collides with `step_size` and raises `TypeError`. |

### `test_method_equivalence.py` (19 cases)

single_step and multi_step sample the same process, so expectations must agree.

| Test | Cases | What it checks |
|---|---|---|
| `test_methods_are_not_bit_identical` | 1 | The PRNG streams differ, so a shared seed gives different raw series (asserted on purpose). |
| `test_avg_log_weight_matches_log_lam_minus_E0` | 8 | avg_log_weight per sub-step -> log(lam - E_0) for every method and step size. |
| `test_methods_agree_on_the_energy` | 4 | Same expectation value, so the windowed energies must agree. |
| `test_methods_agree_on_a_diagonal_observable` | 4 | A bond-averaged SzSz agrees between methods (at fixed p both are equally unconverged). |
| `test_the_observable_converges_towards_exact_with_step_size` | 1 | Both methods approach exact SzSz as step_size (projection depth) grows, so agreement is not vacuous. |
| `test_cont_time_lam_none_reports_statistics` | 1 | cont_time with `lam=None` prints its end-of-run statistics report without raising. |

### `test_multi_gpu.py` (7 cases)

run_gfmc_pmap should agree with run_gfmc on the same TFIM problem.

| Test | Cases | What it checks |
|---|---|---|
| `test_pmap_energy_matches_serial` | 1 | Windowed energies agree between the serial and pmap backends. |
| `test_pmap_off_diagonal_matches_serial` | 1 | Off-diagonal forward-walking estimator: same formula in both paths. |
| `test_run_dispatcher_matches_run_gfmc` | 1 | `run` on a single device reproduces `run_gfmc` exactly (same path, same RNG). |
| `test_pmap_rejects_diag_operators` | 1 | diag_operators is not supported on the pmap path and must raise. |
| `test_pmap_off_diag_batch_size_is_a_noop` | 1 | off_diag_batch_size is bit-identical on the pmap path's operator application (against the module's reference pmap run). |
| `test_pmap_off_diag_batch_size_must_divide_shard` | 1 | A batch size that does not divide the PER-DEVICE shard is rejected. |
| `test_pmap_return_raw_false_without_off_diag` | 1 | Without off-diagonal operators pmap accepts return_raw=False, returns the 4-tuple and records rng keys. |

### `test_off_diag_stride.py` (2 cases)

run_gfmc's off_diag_stride and off_diag_batch_size: the main chain and every measured row stay bit-identical.

| Test | Cases | What it checks |
|---|---|---|
| `test_off_diag_stride_consistency` | 1 | Strides 1/2/3/6: bit-identical main chain, compact rows equal to stride-1 values. |
| `test_off_diag_batch_size_is_a_noop` | 1 | off_diag_batch_size is BIT-IDENTICAL (per-walker keys from absolute indices), so assert ==. |

### `test_positivity_and_zero_weight.py` (5 cases)

Off-diagonal measurement operators: the positivity check and the zero-weight guard.

| Test | Cases | What it checks |
|---|---|---|
| `test_positivity_check_passes_and_fails_as_it_should` | 1 | The report prints only when verbose, a negative rate raises, a diagonal-only operator may carry any sign. |
| `test_negative_rate_operator_is_refused_before_production` | 2 | Both drivers raise during the positivity probe; the correctly signed operator runs. |
| `test_operator_with_no_allowed_move_on_a_configuration_stays_finite` | 1 | A bond-hop measurement on walkers whose bond is empty gives finite values and an exact 0 at step 0. |
| `test_population_extinction_is_an_error_not_nan` | 1 | A step on which every branch weight is zero raises a `ValueError` naming the step. |

### `test_repository_consistency.py` (8 cases)

The published notebooks and the manuscript assets stay usable from a source checkout.

| Test | Cases | What it checks |
|---|---|---|
| `test_notebook_sources_and_downloads` | 7 | Every notebook parses, has no stored error output, and is downloadable with its required bundled HDF5 data. |
| `test_manuscript_references_and_figure_inventory` | 1 | Manuscript references resolve uniquely and each included figure has a PDF, regeneration script and documentation. |

### `test_review_regressions.py` (19 cases)

Regression tests for numerical, checkpoint and API review findings.

| Test | Cases | What it checks |
|---|---|---|
| `test_tiny_hopping_does_not_overflow_geometric_hold` | 3 | A zero or tiny transverse field keeps the multi_step geometric hold finite and the configuration unchanged. |
| `test_checkpoint_switches_source_and_honors_override` | 1 | Resume reads the walker checkpoint the file names (embedded or external) and checks it by hash, honors an explicit `walker_filename`, and a later embedded checkpoint supersedes an external one. |
| `test_resume_without_raw_handles_diagonal_lead_in` | 1 | `run_from_HDF5(..., return_raw=False)` with diagonal operators returns the 4-tuple with finite estimates. |
| `test_nonergodic_default_training_sampler_rejected_before_netket` | 2 | `train_with_netket` refuses the default exchange sampler for local spaces of more than two values, before importing NetKet. |
| `test_autocorrelation_handles_both_signs` | 2 | The integrated autocorrelation time of an AR(1) series matches (1 + rho) / (2 (1 - rho)) for rho = +-0.8. |
| `test_nested_incremental_state_tracks_configuration` | 3 | A guide state held in a nested pytree follows the configuration through every propagator. |
| `test_float64_constructor_precision_and_roundtrip` | 1 | Under x64, JastrowSpinHalf and ClusterCPS keep float64 parameters exactly through construction, save and load. |
| `test_geometry_rejects_self_edges_and_fractional_indices` | 1 | Self-edges and fractional site indices raise. |
| `test_unsupported_rng_checkpoint_does_not_overwrite` | 1 | Saving a non-threefry key raises and leaves the existing checkpoint unchanged byte for byte. |
| `test_unsupported_rng_fails_before_running_or_writing` | 1 | A non-threefry key is refused by `run_gfmc` and `write_to_HDF5` before any step is taken or any file written. |
| `test_shadow_rates_respect_small_positive_budget` | 1 | A rate-carry budget below one population's rates keeps the main and shadow chains on the uncarried path; `rates_fit_budget` is exact at the boundary. |
| `test_pmap_shadow_budget_counts_whole_population` | 1 | The pmap shadow chain sizes the carry budget by the whole population, not one shard. |
| `test_jackknife_tail_truncates_all_observable_series` | 1 | The trailing-step truncation applies to the diagonal and off-diagonal series too; integer-valued float `off_diag_steps` are accepted. |

### `test_sampler.py` (3 cases)

Inverse-CDF sampling of a move (`sample_index`): only slots with positive weight can be drawn.

| Test | Cases | What it checks |
|---|---|---|
| `test_zero_weight_slot_is_never_drawn` | 1 | A cdf that rises across a zero-weight slot (as XLA's parallel cumsum can, exaggerated here) never returns that slot. |
| `test_index_stays_inside_the_array` | 1 | Trailing zero-weight slots are unreachable, also through the clamp of an out-of-range searchsorted result. |
| `test_exact_cdf_draws_proportionally` | 1 | With an exact cdf, slots are drawn in proportion to their weights. |

### `test_sign_problem_free.py` (6 cases)

Sign-problem-free guards: a wrong-sign coupling must be REJECTED.

| Test | Cases | What it checks |
|---|---|---|
| `test_plaquette_site_count_comes_from_the_plaquettes` | 1 | No `N` argument: the site count comes from the plaquettes; gaps and negative indices raise. |
| `test_plaquette_sign_check_rejects_negative_g` | 1 | `H = -g Sum_p flip_p` needs g >= 0; measurement operators (g <= 0) bypass it. |
| `test_toric_code_sign_check_rejects_negative` | 1 | Negative A_s or h_x is rejected unless sign_check=False. |
| `test_toric_code_sign_problem_free` | 1 | Every off-diagonal `get_rates` entry is <= 0 over the full basis when A_s >= 0 and h_x >= 0. |
| `test_spinhalf_rejects_positive_jpmpm_instead_of_swapping` | 1 | A wrong-sign J_pmpm raises, naming the S^x <-> S^y remedy, instead of being swapped silently. |
| `test_spinhalf_field_sign_convention` | 1 | The `get_rates` matrix of `-(h_z S^z + h_x S^x)` equals the explicit 2x2 matrix and has ToricCode's field spectrum; h_x < 0 raises unless sign_check=False. |

### `test_silence_and_gating.py` (7 cases)

`verbosity=0` prints nothing, and the equal-window gating of `return_raw=False` (the suite's one home for it).

| Test | Cases | What it checks |
|---|---|---|
| `test_verbosity_zero_is_silent_even_with_duplicates_and_operators` | 1 | Duplicate `p` values, an off-diagonal operator and `write_to_HDF5(verbose=False)` print nothing. |
| `test_verbosity_one_reports_positivity_and_statistics` | 1 | At `verbosity=1` the positivity report, the dedup notice and the statistics report appear, without a banner. |
| `test_write_to_HDF5_prints_one_line_by_default` | 1 | The default is one line naming the group. |
| `test_equal_windows_of_return_raw_false` | 2 | `return_raw=False` at strides 1 and 2 measures exactly the stride steps of the band every p can use, and leaves the main chain and the largest window untouched. |
| `test_gated_chain_continues_gated` | 1 | A chain stored without raw series continues through `run_from_HDF5` with equal windows and appends. |
| `test_equal_windows_is_a_must_match_parameter` | 1 | Flipping the gating on an existing file is refused. |

### `test_statistics_contract.py` (8 cases)

The keys and invariants of `statistics()` and of the `stats` a run returns.

| Test | Cases | What it checks |
|---|---|---|
| `test_statistics_keys_and_shapes` | 3 | Each method exposes exactly its own diagnostic family (`hold_steps_*` vs `escape_rate_*`, `hold_time_*`, `hops_per_step_*`), with the documented lengths. |
| `test_offspring_fractions_sum_to_one_and_define_survival` | 1 | `offspring_k0..k3` plus `offspring_k4plus` sum to 1; `survival_rate = 1 - offspring_k0`. |
| `test_effective_sample_size_starts_at_N_and_never_grows_with_depth` | 1 | Kish ESS is N at m = 0, non-increasing in m, and never exceeds the surviving-ancestor count. |
| `test_ancestor_correlation_is_one_at_depth_zero_and_decays_with_the_step` | 1 | `ancestor_correlation` is 1 at m = 0 and for a step too short for any hop, lies in [-1, 1], decays with m, and is gone after one step of ten hops. |
| `test_avg_log_weight_is_optional_and_only_a_numerical_shift` | 1 | Omitting `avg_log_weight` uses the walkers' own estimate bit-identically; a different shift moves every diagnostic only within float noise. |
| `test_run_stats_carry_the_diagnostics_and_the_inputs` | 1 | A driver's stats hold the diagnostics, the inputs under `parameters/`, the autocorrelation times, and no duplicate keys. |

### `test_structure_factor_operator.py` (24 cases)

StructureFactor vs an INDEPENDENT Kronecker construction.

| Test | Cases | What it checks |
|---|---|---|
| `test_matches_kronecker_reference` | 2 | Both channels equal the Kronecker reference over the full 2^N matrix. |
| `test_non_vacuous` | 1 | Perturbing one weight by 1e-4 relative must be caught. |
| `test_rates_non_negative` | 3 | The off-diagonal estimator requires every rate >= 0 (inverse-CDF sampling). |
| `test_k_zero_reduces_to_the_plain_pair_sum` | 1 | At k=0 the weight is 2 on every pair, so J_pm = 2 (J_x = J_y = 4). |
| `test_blocked_and_invalid_combinations` | 1 | YY, S.S, misspelled ZZ and mixed species raise with the route named. |
| `test_input_validation` | 1 | The shape contract on positions and k is enforced. |
| `test_periodic_image_does_not_matter_for_valid_k` | 1 | Shifting sites by periodic vectors leaves the operator unchanged for an allowed k, and changes it for a k off the grid. |
| `test_is_valid_k` | 1 | k.L_j / 2 pi integer for every row; single k gives a bool, many an array. |
| `test_incommensurate_k_is_accepted_without_complaint` | 1 | The constructor does NOT check k, by design (commensurability is the caller's job, `is_valid_k`); the weights are 1 + cos(k.dr) on every pair for any k. |
| `test_which_AB0_plus_ABk_it_actually_is` | 3 | `O == herm(0) + herm(k)` always; `O == Re[AB(0)+AB(k)]` (no factor 2) only for SpSm and SzSz. |
| `test_szsz_channel_is_szsz_structure_factor_at_0_plus_k` | 2 | StructureFactor(channel="SzSz") == SzSzStructureFactor at q = 0 plus q = k, configuration by configuration, on random positions and on a triangular torus with valid k. |
| `test_szsz_channel_matches_kronecker_reference` | 1 | The SzSz channel is sum_{i<j} (1 + cos)(Z_iZ_j + h.c.) = 2(1 + cos) Z_iZ_j. |
| `test_channels_are_additive` | 1 | O("SpSm+SpSp") == O("SpSm") + O("SpSp") exactly; names are case/space-insensitive. |
| `test_scalar_Jx_alongside_array_Jz_builds` | 1 | A scalar J_x next to a per-pair J_z array builds the diagonal operator with the explicit pair sum. |
| `test_any_dimension` | 4 | Nothing is 3D-specific; the weight is a dot product. |

### `test_user_defined_operators.py` (27 cases)

The paper's user-defined operator classes, verified against PyGFMC.

| Test | Cases | What it checks |
|---|---|---|
| `test_nbody_get_rates_rebuilds_the_matrix` | 1 | H[y,x] from get_rates must equal the Kronecker construction. |
| `test_nbody_return_configs_agrees_with_generate_state` | 1 | The materialized tensor and the on-the-fly path must not disagree. |
| `test_nbody_gfmc_matches_exact_diagonalization` | 1 | A 1024-walker, 600-step run with the user-defined Hamiltonian reaches the ED energy within 0.02/site. |
| `test_structure_factor_diagonal_weight_is_exact` | 1 | The appendix structure factor equals an explicit pair sum. |
| `test_protocol_hamiltonian_exact_diagonalization` | 1 | exact_diagonalization needs nothing beyond the protocol. |
| `test_protocol_hamiltonian_observables` | 1 | Diagonal and off-diagonal observables on a protocol-only Hamiltonian. |
| `test_protocol_hamiltonian_exact_guide` | 1 | ExactGroundState on a protocol-only Hamiltonian needs local_hilbert_space, then is exact. |
| `test_protocol_hamiltonian_exact_guide_round_trips` | 1 | The generic index map survives HDF5 and pytree round-trips. |
| `test_generic_index_agrees_with_the_per_class_map` | 1 | Default and reversed local orderings give the same amplitude per configuration. |
| `test_custom_guide_routines_agree` | 1 | The three guide routines must return the same ratios. |
| `test_custom_guide_at_beta_zero_reduces_to_no_guide` | 1 | beta = 0 draws the same moves as no guide; e_L agrees to float32 rounding. |
| `test_custom_guide_gfmc_reaches_the_exact_ground_state` | 1 | GFMC is unbiased whatever the guide; the tolerance covers the MC error. |
| `test_structure_factor_multi_k_matches_the_appendix_pair_sum` | 2 | Same numbers as the pair sum at every allowed k, in both image conventions. |
| `test_structure_factor_takes_the_wavevectors_as_given` | 1 | k is not validated (positions cannot decide it); only shapes are checked. |
| `test_structure_factor_zero_k_is_the_magnetization` | 1 | S(0) = ((sum_i s_i)^2 - N)/4 exactly: the i==j terms are dropped. |
| `test_structure_factor_is_a_pytree_and_jits` | 1 | Registered as a pytree with the phase tables as children; jits. |
| `test_structure_factor_multi_k_in_a_run_equals_separate_operators` | 1 | One vector-valued operator == n_k scalar appendix operators, through run_gfmc. |
| `test_structure_factor_equals_the_appendix_pair_sum_on_any_lattice` | 9 | \|sum_i e^{ik r_i} s_i\|^2 == the O(N^2) pair sum, on symmetric and asymmetric cells. |

### `test_validation_messages.py` (55 cases)

Each bad input raises its own message before any work: one parametrized table over the guide constructors, the driver, the Hamiltonian constructors, `ExactGroundState`, the spin-1/2 guides' operator-layout check and `evaluate_observables`, plus a protocol smoke test.

| Test | Cases | What it checks |
|---|---|---|
| `test_engine_rejects_bad_input_before_walking` | 21 | `run_gfmc` rejects nonfinite steps/shifts, zero walkers, an operator or guide missing its routines, a non-dividing batch size, max(p) >= n_steps, a too-short off-diagonal run, an empty equal-window band and odd p, with the warm-up, probe and production replaced by a failing stub. |
| `test_constructor_and_layout_messages` | 33 | ClusterCPS / ClusterMLP group specs, amplitudes and parameters, Spin `S`, BoseHubbard `n_max`, the `ExactGroundState` `log_amp` length, the SpinHalf-layout `TypeError` of JastrowSpinHalf / ClusterCPS / ClusterMLP on a ToricCode, and `evaluate_observables` without `'energy'`, each with its message. |
| `test_built_in_classes_satisfy_the_protocols` | 1 | Built-in Hamiltonians carry every `pygfmc.Operator` member and pass the engine check, built-in guides have a ratio routine, `__version__` equals `codes/pyproject.toml`'s, and `kagome(2, 2)` has its hexagons. |

### `test_walker_checkpoint_dtypes.py` (7 cases)

Walker checkpoints for every configuration encoding, and the load dtype.

| Test | Cases | What it checks |
|---|---|---|
| `test_save_compacts_when_lossless_and_loads_back_exactly` | 5 | Integer-valued encodings are stored as `int8`, half-integer ones as given; arrays, key and attributes round-trip exactly. |
| `test_load_without_key_or_alw_returns_none` | 1 | Missing optional items come back as None. |
| `test_load_dtype_follows_jax_default_float` | 1 | The default load dtype follows `jax_enable_x64`; explicit dtypes are honored. |

### `test_walker_init.py` (22 cases)

Walker ensembles: each initializer must produce what it promises.

| Test | Cases | What it checks |
|---|---|---|
| `test_prepare_walkers_random_is_iid` | 1 | Every site an independent coin flip: +-1 values, magnetization varies. |
| `test_prepare_walkers_random_draws_only_from_states` | 4 | Every entry is one of `states` -- and all of them do occur. |
| `test_prepare_walkers_sz_sums_exactly` | 8 | sz is the SUM over sites and must be exact for EVERY walker (it pins a sector). |
| `test_prepare_walkers_sz_rejects_unreachable` | 4 | An impossible request must raise, not return a wrong-sum ensemble. |
| `test_prepare_walkers_dtype_follows_states` | 2 | The config dtype comes from `states`; weights are always floating point. |
| `test_prepare_walkers_from_configs_keeps_the_configurations` | 1 | The given (N_walkers, N_sites) array is used as is, dtype included. |
| `test_prepare_walkers_from_configs_rejects_non_2d` | 2 | A single (N_sites,) configuration or a 3-D array raises. |

### `test_walker_io.py` (3 cases)

Walker population round-trips through HDF5, and a resumed run is exact.

| Test | Cases | What it checks |
|---|---|---|
| `test_walker_save_load_resume` | 1 | Checkpointed walkers + RNG key reload bit-exactly and the resumed run matches. |
| `test_systematic_resampling_never_selects_a_zero_weight_walker` | 1 | The parent lookup never returns a zero-weight walker: leading zero at `psi = 0`, trailing zero, a cdf that rounds short of the largest `u`, and dead (`-inf`) walkers through `reconfiguration`. |
| `test_typed_prng_keys_are_checkpointed_as_key_data` | 1 | `jax.random.key` keys are accepted by `Walker.save` and the driver and stored as uint32 key data, equal to the legacy key's. |

### `test_wavefunction_pytree.py` (9 cases)

Pytree hygiene for every guide in `pygfmc/wavefunction/`.

| Test | Cases | What it checks |
|---|---|---|
| `test_aux_is_hashable_and_self_equal` | 5 | Two identical instances have equal, hashable treedefs and share one jit trace. |
| `test_exact_ground_state_E0_survives_roundtrip` | 1 | E0 (NaN when log_amp is supplied, finite otherwise) survives tree_map. |
| `test_jastrow_beta_after_tree_roundtrip` | 1 | JastrowSpinHalf.beta survives tree_flatten / tree_unflatten. |
| `test_jastrow_beta_after_jit` | 1 | JastrowSpinHalf.beta is rebuilt from a jitted tree_map and readable inside jit. |
| `test_jastrow_beta_equals_minus_J_everywhere` | 1 | Every per-site coupling entry of a pair is -beta of that pair (input order, unsorted pairs included), and log_psi is -sum beta_ij x_i x_j. |
