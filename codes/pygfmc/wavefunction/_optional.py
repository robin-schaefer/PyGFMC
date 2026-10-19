"""Optional dependencies.

NetKet (and optax) are needed only to TRAIN a guide; running GFMC with an
already-trained guide never touches them.  They are therefore an optional
extra, ``pip install "pygfmc[netket]"`` (``pip install -e "./codes[netket]"``
in a clone of the repository), and
every NetKet entry point imports through :func:`import_netket` so a missing
install fails with the remedy rather than a bare ``ModuleNotFoundError`` from
deep inside a training call.
"""

_INSTALL_HINT = ('pip install "pygfmc[netket]" (or pip install -e "./codes[netket]" '
                 'from the root of a clone of https://github.com/robin-schaefer/PyGFMC)')


def import_netket():
    """Return the ``netket`` module, or raise ``ImportError`` naming the fix.

    Two failure modes are both reported as ``ImportError``: NetKet absent, and
    NetKet present but failing on import.  Treating the second as an
    ImportError is what lets callers and tests handle "NetKet unusable here"
    in one place.
    """
    try:
        import netket
    except ImportError as e:
        raise ImportError(
            f"This feature requires NetKet, which is an optional dependency "
            f"of PyGFMC (only needed for training guides). Install it with "
            f"{_INSTALL_HINT}.") from e
    except Exception as e:
        raise ImportError(
            f"NetKet is installed but failed to import ({type(e).__name__}: "
            f"{e}). Training a guide needs a working NetKet installation; "
            f"running GFMC with a stored guide does not.") from e
    return netket
