"""HDF5 attribute helper for the checkpoint files of walkers and guides.

``Walker.save`` and the guides' ``save`` methods each write an HDF5 group
holding a few arrays plus scalar metadata as group attributes;
:func:`write_attrs` serialises that metadata, so all of them store attributes
the same way.  Reading needs no helper: the ``load`` methods read
``group.attrs`` directly.
"""
import numpy as np


def write_attrs(group, attrs: dict) -> None:
    """Write a dict of scalar metadata as HDF5 group attributes.

    None values are skipped so callers can pass through unset optional fields.
    Floats are coerced to Python ``float`` for portable storage.
    """
    for key, val in attrs.items():
        if val is None:
            continue
        if isinstance(val, float) or (hasattr(val, 'dtype')
                                       and np.issubdtype(val.dtype, np.floating)
                                       and val.shape == ()):
            group.attrs[key] = float(val)
        else:
            group.attrs[key] = val

