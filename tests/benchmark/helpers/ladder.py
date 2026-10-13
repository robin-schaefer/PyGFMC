"""Two-leg ladder: PyGFMC edge list + matching TeNPy DMRG models.

Two open chains of ``L`` sites (the legs) plus ``L`` rungs, ``N = 2*L``.  The
TeNPy models take the same coupling names as the matching PyGFMC class's
``create(...)``, so one parameter dict drives both sides:

  - SpinHalf: (J_x, J_y, J_z) multiply the physical spin operators
    Sx Sx + Sy Sy + Sz Sz and (h_z, h_x) multiply -Sz and -Sx (the fields
    enter H with a minus sign, as in ``SpinHalf.create``), i.e. TeNPy's
    SpinHalfSite Sx/Sy/Sz.
  - Spin (general S): (J_z, J_pm, J_pmpm, h_z, h_pm) map onto SpinSite's
    Sz/Sp/Sm.
  - BoseHubbard: (t, eta, U, mu, h_pm) map onto BosonSite's B/Bd/N, with the
    U term built as 0.5*U*(NN - N) since TeNPy's 'NN' is N^2.

TeNPy's stock BoseHubbardModel has no pairing (eta) term, so all three
families use a custom ``CouplingMPOModel``.
"""

from __future__ import annotations


def ladder_edges(L: int = 50, periodic_legs: bool = False):
    """Two chains of L sites (0..L-1 top leg, L..2L-1 bottom leg) + L rungs.

    Returns (edges, N) with N = 2*L.
    """
    leg = [(i, i + 1) for i in range(L - 1)] + [(L + i, L + i + 1) for i in range(L - 1)]
    if periodic_legs and L > 2:
        leg += [(L - 1, 0), (2 * L - 1, L)]
    rung = [(i, L + i) for i in range(L)]
    return leg + rung, 2 * L


def _ladder_pairs(lat):
    """(u1, u2, dx) triples for every leg and rung bond of a TeNPy Ladder."""
    return list(lat.pairs["leg_NN"]) + list(lat.pairs["rung_NN"])


def _make_spinhalf_ladder_model():
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    class SpinHalfLadderModel(CouplingMPOModel):
        default_lattice = "Ladder"
        force_default_lattice = True

        def init_sites(self, model_params):
            return SpinHalfSite(conserve=None)

        def init_terms(self, model_params):
            Jx = model_params.get("J_x", 0.0)
            Jy = model_params.get("J_y", 0.0)
            Jz = model_params.get("J_z", 0.0)
            hx = model_params.get("h_x", 0.0)
            hz = model_params.get("h_z", 0.0)

            for u1, u2, dx in _ladder_pairs(self.lat):
                if Jx:
                    self.add_coupling(Jx, u1, "Sx", u2, "Sx", dx)
                if Jy:
                    self.add_coupling(Jy, u1, "Sy", u2, "Sy", dx)
                if Jz:
                    self.add_coupling(Jz, u1, "Sz", u2, "Sz", dx)
            # SpinHalf convention: H contains -h_x Sum Sx - h_z Sum Sz
            if hx:
                for u in range(2):
                    self.add_onsite(-hx, u, "Sx")
            if hz:
                for u in range(2):
                    self.add_onsite(-hz, u, "Sz")

    return SpinHalfLadderModel


def _make_spin_ladder_model():
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinSite

    class SpinLadderModel(CouplingMPOModel):
        default_lattice = "Ladder"
        force_default_lattice = True

        def init_sites(self, model_params):
            S = model_params.get("S", 1.0)
            conserve = model_params.get("conserve", None)
            return SpinSite(S=S, conserve=conserve)

        def init_terms(self, model_params):
            Jz    = model_params.get("J_z", 0.0)
            Jpm   = model_params.get("J_pm", 0.0)
            Jpmpm = model_params.get("J_pmpm", 0.0)
            hz    = model_params.get("h_z", 0.0)
            hp = hm = model_params.get("h_pm", 0.0)   # one amplitude for S+ and S-

            for u1, u2, dx in _ladder_pairs(self.lat):
                if Jz:
                    self.add_coupling(Jz, u1, "Sz", u2, "Sz", dx)
                if Jpm:
                    self.add_coupling(Jpm, u1, "Sp", u2, "Sm", dx)
                    self.add_coupling(Jpm, u1, "Sm", u2, "Sp", dx)
                if Jpmpm:
                    self.add_coupling(Jpmpm, u1, "Sp", u2, "Sp", dx)
                    self.add_coupling(Jpmpm, u1, "Sm", u2, "Sm", dx)
            if hz:
                for u in range(2):
                    self.add_onsite(hz, u, "Sz")
            if hp:
                for u in range(2):
                    self.add_onsite(hp, u, "Sp")
            if hm:
                for u in range(2):
                    self.add_onsite(hm, u, "Sm")

    return SpinLadderModel


def _make_bose_hubbard_ladder_model():
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import BosonSite

    class BoseHubbardLadderModel(CouplingMPOModel):
        default_lattice = "Ladder"
        force_default_lattice = True

        def init_sites(self, model_params):
            n_max = model_params.get("n_max", 2)
            conserve = model_params.get("conserve", None)
            return BosonSite(Nmax=n_max, conserve=conserve)

        def init_terms(self, model_params):
            t   = model_params.get("t", 0.0)
            eta = model_params.get("eta", 0.0)
            U   = model_params.get("U", 0.0)
            mu  = model_params.get("mu", 0.0)
            hp = hm = model_params.get("h_pm", 0.0)   # one amplitude for b+ and b

            for u1, u2, dx in _ladder_pairs(self.lat):
                if t:
                    self.add_coupling(-t, u1, "Bd", u2, "B", dx)
                    self.add_coupling(-t, u1, "B", u2, "Bd", dx)
                if eta:
                    self.add_coupling(-eta, u1, "Bd", u2, "Bd", dx)
                    self.add_coupling(-eta, u1, "B", u2, "B", dx)
            if U:
                for u in range(2):
                    self.add_onsite(0.5 * U, u, "NN")
                    self.add_onsite(-0.5 * U, u, "N")
            if mu:
                for u in range(2):
                    self.add_onsite(-mu, u, "N")
            if hp:
                for u in range(2):
                    self.add_onsite(hp, u, "Bd")
            if hm:
                for u in range(2):
                    self.add_onsite(hm, u, "B")

    return BoseHubbardLadderModel


_MODEL_FACTORY = {
    "SpinHalf": _make_spinhalf_ladder_model,
    "Spin": _make_spin_ladder_model,
    "BoseHubbard": _make_bose_hubbard_ladder_model,
}


def build_ladder_model(family: str, L: int, params: dict, bc_MPS: str = "finite"):
    """Construct a TeNPy CouplingMPOModel instance for the given family."""
    ModelClass = _MODEL_FACTORY[family]()
    model_params = dict(params)
    model_params.update(L=L, bc_MPS=bc_MPS)
    return ModelClass(model_params)
