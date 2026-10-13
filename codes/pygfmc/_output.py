"""Printed output of the package.

Everything PyGFMC prints passes through :func:`emit`, gated on a caller's
``verbosity``: a message is printed only when ``verbosity >= level``.  The
drivers pass their ``verbosity`` argument through, so ``verbosity=0`` is
silent; functions with a boolean switch (``quiet``, ``verbose``) pass it as
``verbosity=int(flag)``.  Set :data:`stream` to redirect every message to a
file-like object instead of standard output.
"""
import sys

#: Destination of every message; ``None`` means ``sys.stdout`` at call time.
stream = None


def emit(message, verbosity, level=1, flush=False):
    """Print ``message`` when ``verbosity >= level``."""
    if verbosity >= level:
        print(message, file=stream if stream is not None else sys.stdout,
              flush=flush)
