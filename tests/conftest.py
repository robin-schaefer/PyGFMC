"""pytest configuration for PyGFMC test suite.

Fixtures defined here apply to every test file automatically.

JAX/NetKet global state
-----------------------
NetKet changes process-wide JAX state, and the NetKet tests import it from
inside a test body, so the suite guards against two leaks:

* ``jax_enable_x64``: the ``reset_jax_x64`` fixture restores the flag after
  every test, so a test that switches it cannot change the precision of the
  tests that follow.
* NetKet's experimental sharding installs a named mesh in JAX's thread-local
  config.  ``NETKET_SHARDING=0`` and its deprecated older name
  ``NETKET_EXPERIMENTAL_SHARDING=0`` (set below, before NetKet is imported)
  keep it from doing so, and the ``reset_jax_mesh_state`` fixture
  clears any mesh a test leaves behind, which would break later ``jax.pmap``
  calls.

With both in place any number of NetKet tests run in one process.
"""
import contextlib
import os
import sys

import pytest

# ── NetKet sharding off, before NetKet is ever imported ────────────────────
# NetKet's experimental sharding path goes through jax.experimental.shard_map,
# which installs a named mesh in JAX's thread-local config and never pops it.
# The reset_jax_mesh_state fixture below wipes that mesh after each test (it has
# to: a leaked mesh breaks every later jax.pmap call), but NetKet caches the
# fact that it configured sharding -- so a SECOND NetKet test in the same
# process then builds NamedSharding(mesh=AbstractMesh((), ...)) and dies with
# "One of shardings leaf specifications got an empty NamedSharding".
#
# The two mechanisms are individually necessary and jointly incompatible, so we
# remove the cause instead: with sharding disabled NetKet never installs a mesh,
# nothing leaks, and any number of NetKet tests coexist in one process.  It
# must be in place before the first `import netket`, hence module scope in
# conftest.  Newer NetKet releases read NETKET_SHARDING instead of the
# deprecated experimental name, so both are set.
os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "0")
os.environ.setdefault("NETKET_SHARDING", "0")

# The `pygfmc` package lives in ../codes.  Put it first on the path so the suite runs
# against that tree without installation (after `pip install -e ./codes` from the
# repository root this resolves to the same files), and put this folder on the path so the
# benchmark helpers import as `benchmark.helpers`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "codes"))
sys.path.insert(0, _HERE)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: takes more than a minute on a CPU; `-m 'not slow'` runs the "
        "quicker subset of the suite (what CI runs).",
    )


@pytest.fixture(autouse=True)
def reset_jax_x64():
    """Keep one test's ``jax_enable_x64`` from becoming every later test's.

    A safety net: ``pygfmc`` sets ``NETKET_ENABLE_X64=0`` before NetKet is
    imported and ``train_with_netket`` restores the caller's flag, but any
    test that switches x64 from inside its body would otherwise leave the rest
    of the process in float64.  The walker initializers take their width from
    this flag while a Hamiltonian's dtype is an explicit argument, so the two
    would disagree and a production scan would fail with a carry-type
    mismatch.

    Snapshotting per test rather than per module leaves deliberate
    module-scope precision choices intact: those run at import time, before
    this fixture ever sees the flag, so it restores the value the module
    asked for.  What it undoes is a change made from inside a test body.
    """
    import jax
    prev = jax.config.jax_enable_x64
    try:
        yield
    finally:
        if jax.config.jax_enable_x64 != prev:
            jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def x64():
    """Scoped double precision: ``with x64(): ...`` turns ``jax_enable_x64`` on
    inside the block and restores the previous value on exit, also on a raise.

    The one way tests switch precision, so a float64 check never leaks into
    the float32 tests collected after it.  A module whose every test needs
    float64 wraps its tests in an autouse fixture that enters this scope.
    """
    import jax

    @contextlib.contextmanager
    def scope(enabled=True):
        prev = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", enabled)
        try:
            yield
        finally:
            jax.config.update("jax_enable_x64", prev)

    return scope


@pytest.fixture(autouse=True)
def reset_jax_mesh_state():
    """Clear JAX's thread-local mesh stack after every test.

    NetKet internally uses ``jax.experimental.shard_map`` which pushes a
    named mesh (e.g. ``Mesh('S': 1)``) onto JAX's thread-local resource
    stack.  If that mesh is not popped before the next test (e.g. due to
    an exception in NetKet), subsequent ``jax.pmap`` calls fail with a
    mesh-mismatch ``ValueError``.

    This fixture wipes the stack after each test so no residual mesh state
    bleeds between tests.
    """
    try:
        import jax._src.mesh as jm
        from jax._src import config as jconfig
        # Snapshot the thread-local mesh sentinels before the test runs.
        # These are plain object() instances — the "not set locally" state.
        _saved_abstract = jconfig.abstract_mesh_context_manager.get_local()
        _saved_device   = jconfig.device_context.get_local()
    except Exception:
        _saved_abstract = _saved_device = None

    yield

    try:
        # NetKet (via jax.experimental.shard_map) sets a named mesh
        # (e.g. 'S': 1) in JAX's thread-local config.  It never restores
        # the original sentinels, so subsequent jax.pmap calls fail with
        # a mesh-mismatch ValueError.  Restore the saved sentinels to put
        # the thread-local state back exactly as it was before the test.
        import jax._src.mesh as jm
        from jax._src import config as jconfig
        if _saved_abstract is not None:
            jconfig.abstract_mesh_context_manager.set_local(_saved_abstract)
        if _saved_device is not None:
            jconfig.device_context.set_local(_saved_device)
        jm.thread_resources.stack.clear()
    except Exception:
        pass
