"""Seed handling shared by the guides' random initialisers."""
import os

from .._output import emit


def resolve_seed(seed, *, announce=False, context="guide"):
    """Return a concrete integer seed, drawing one from the OS if ``seed is None``.

    ``seed=None`` means "pick a fresh seed", so independent initialisations are
    not silently correlated.  The guides keep the drawn seed as ``guide.seed``,
    the record needed to reproduce the run; ``announce=True`` also prints it.
    """
    if seed is not None:
        return int(seed)
    seed = int.from_bytes(os.urandom(4), "little")
    emit(f"[{context}] drew random seed {seed} from os.urandom "
         f"(pass seed=... to reproduce this run)", int(bool(announce)))
    return seed
