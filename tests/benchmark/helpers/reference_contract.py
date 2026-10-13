"""Compatibility checks for committed references, without external solvers.

The manifest pins reference bytes and the benchmark's geometry, initialization
sector and measurement conventions. It records an audited pairing, not a new
ED/DMRG calculation. After regenerating and reviewing references, explicitly
refresh it with ``python -m benchmark.helpers.reference_contract`` from tests/.
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "reference_data"
MANIFEST = ROOT / "contracts.json"


def contract(cfg):
    from .gfmc_bench import measurement_operators
    h, edges = cfg["build"]()
    diag, off_diag = measurement_operators(cfg)
    geometry = {}
    for name in ("edges", "plaquettes", "stars"):
        value = edges if name == "edges" else getattr(h, name, None)
        if value is not None:
            encoded = json.dumps(np.asarray(value).tolist(), separators=(",", ":"))
            geometry[name] = hashlib.sha256(encoded.encode()).hexdigest()
    return dict(
        id=cfg["id"], family=cfg["family"], phase=cfg["phase"], N=cfg["N"],
        params=cfg["params"], geometry=geometry,
        initial_charge=cfg.get("target_charge"),
        measured_observables=sorted(set(diag) | set(off_diag)),
        reference_bond=("middle rung" if "_ladderL" in cfg["id"] else "first edge"),
        measurement_protocol_sha256=hashlib.sha256(
            inspect.getsource(measurement_operators).encode()).hexdigest(),
        sector=("zero_flux_ice" if cfg["family"] == "PlaquetteModel" else
                "Bp=+1" if cfg["family"] == "ToricCode" else "see_initial_charge"),
        normalization="E0=total; *_tot=total; *_avg/*_frac=average; "
                      "SpinHalf z_tot=sum(Sz), x_tot=sum(PauliX); "
                      "ToricCode=Pauli; ring_flip=positive stoquastic",
    )


def validate_reference(f, cfg):
    """Reject stale couplings, geometry, sectors, conventions or reference bytes."""
    for key in ("id", "family", "phase", "N"):
        assert f.attrs[key] == cfg[key], (cfg["id"], key)
    params = {k[6:]: v for k, v in f.attrs.items() if k.startswith("param_")}
    assert set(params) == set(cfg["params"]), (cfg["id"], "parameter keys")
    for name, value in cfg["params"].items():
        np.testing.assert_allclose(params[name], value, rtol=0, atol=1e-12,
                                   err_msg=f"{cfg['id']}: {name}")
    current = contract(cfg)
    if "edges" in f:
        np.testing.assert_array_equal(f["edges"][()], cfg["build"]()[1])
    if "graph_seed" in cfg:
        assert f.attrs["graph_seed"] == cfg["graph_seed"]
    if "L" in f.attrs:
        assert 2 * f.attrs["L"] == cfg["N"] and f.attrs["bc"] == "open_legs"
        np.testing.assert_allclose(f.attrs["e_per_site"] * cfg["N"], f.attrs["E0"])
        assert not any(k.endswith("_bond") for k in current["measured_observables"]), (
            "Align the GFMC measurement bond with the DMRG middle rung before enabling ladder bonds")
    # Where number conservation pins the Bose-Hubbard walk, verify the
    # reference actually describes that sector, not merely the same couplings.
    if (cfg["family"] == "BoseHubbard" and "target_charge" in cfg
            and not cfg["params"].get("eta", 0) and not cfg["params"].get("h_pm", 0)):
        np.testing.assert_allclose(f.attrs["n_tot"], cfg["target_charge"], rtol=0, atol=1e-3)
    saved = json.loads(MANIFEST.read_text())[cfg["id"]]
    assert saved["contract"] == current, f"{cfg['id']}: reference contract changed; review/regenerate"
    digest = hashlib.sha256(Path(f.filename).read_bytes()).hexdigest()
    assert saved["sha256"] == digest, f"{cfg['id']}: reference bytes changed; review/regenerate"


def collect_contracts():
    from .models import QUSPIN_MODELS, TENPY_MODELS, PYROCHLORE_MODELS
    result = {}
    for group, models in (("quspin", QUSPIN_MODELS), ("tenpy", TENPY_MODELS),
                          ("pyrochlore", PYROCHLORE_MODELS)):
        for cfg in models:
            path = ROOT / group / (cfg["id"] + ".h5")
            result[cfg["id"]] = dict(contract=contract(cfg),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


if __name__ == "__main__":
    MANIFEST.write_text(json.dumps(collect_contracts(), sort_keys=True, indent=2) + "\n")
