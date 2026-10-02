"""Small, defensive CRYSTAL output parser for the teaching laboratories.

The parser is deliberately conservative.  It accepts a total energy only when
the output contains both an explicit SCF convergence line and normal CRYSTAL
termination.  This prevents an interrupted SCF calculation from silently
becoming a plausible-looking scientific result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.constants import Avogadro, physical_constants


_HARTREE_TO_KJMOL = (
    physical_constants["Hartree energy"][0] * Avogadro / 1000.0
)

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"

_RE_SAMPLING = re.compile(
    r"SHRINK\.\s*FACT\.\(MONKH\.\)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+"
    r"NUMBER\s+OF\s+K\s+POINTS\s+IN\s+THE\s+IBZ\s+(\d+)",
    re.IGNORECASE,
)

_RE_CONVERGED = re.compile(
    rf"=+\s*SCF\s+ENDED\s*-\s*CONVERGENCE\s+ON\s+ENERGY"
    rf"\s+E\(AU\)\s+({_FLOAT})\s+CYCLES\s+(\d+)",
    re.IGNORECASE,
)

_RE_END_TIME = re.compile(
    rf"^\s*T+\s+END\s+TELAPSE\s+({_FLOAT})\s+TCPU\s+({_FLOAT})",
    re.IGNORECASE | re.MULTILINE,
)

_RE_NORMAL_TERMINATION = re.compile(
    r"^\s*E+\s+TERMINATION\b",
    re.IGNORECASE | re.MULTILINE,
)

_RE_ATOMS_PER_CELL = re.compile(
    r"N\.\s*OF\s*ATOMS\s*PER\s*CELL\s+(\d+)",
    re.IGNORECASE,
)


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class CrystalSCFResult:
    """Parsed quantities used in the MgO k-point convergence exercise."""

    path: Path
    shrink: tuple[int, int, int]
    kpoints_ibz: int
    energy_ha: float
    scf_cycles: int
    elapsed_s: float
    cpu_s: float
    atoms_per_cell: int | None
    converged: bool
    normal_termination: bool

    @property
    def isotropic_shrink(self) -> int | None:
        if self.shrink[0] == self.shrink[1] == self.shrink[2]:
            return self.shrink[0]
        return None


def parse_crystal_scf_output(path: str | Path) -> CrystalSCFResult:
    """Parse one completed CRYSTAL single-point SCF output.

    Raises
    ------
    FileNotFoundError
        If the output file does not exist.
    ValueError
        If required sampling information is absent, the SCF did not explicitly
        converge on energy, or CRYSTAL did not terminate normally.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CRYSTAL output not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    sampling = _RE_SAMPLING.search(text)
    if sampling is None:
        raise ValueError(
            f"{path.name}: could not find the SHRINK / IBZ k-point summary."
        )

    converged_match = _RE_CONVERGED.search(text)
    if converged_match is None:
        raise ValueError(
            f"{path.name}: no explicit 'SCF ENDED - CONVERGENCE ON ENERGY' "
            "marker was found. The energy will not be accepted."
        )

    normal_termination = _RE_NORMAL_TERMINATION.search(text) is not None
    if not normal_termination:
        raise ValueError(
            f"{path.name}: SCF convergence was found, but normal CRYSTAL "
            "termination was not. Inspect the output before using the result."
        )

    timing_matches = list(_RE_END_TIME.finditer(text))
    if not timing_matches:
        raise ValueError(
            f"{path.name}: could not find the final END/TELAPSE/TCPU timing line."
        )
    timing = timing_matches[-1]

    atoms = _RE_ATOMS_PER_CELL.search(text)

    return CrystalSCFResult(
        path=path,
        shrink=(
            int(sampling.group(1)),
            int(sampling.group(2)),
            int(sampling.group(3)),
        ),
        kpoints_ibz=int(sampling.group(4)),
        energy_ha=_as_float(converged_match.group(1)),
        scf_cycles=int(converged_match.group(2)),
        elapsed_s=_as_float(timing.group(1)),
        cpu_s=_as_float(timing.group(2)),
        atoms_per_cell=int(atoms.group(1)) if atoms else None,
        converged=True,
        normal_termination=True,
    )


def analyse_crystal_outputs(
    outputs: Sequence[str | Path],
    *,
    formula_units_per_cell: int = 1,
) -> pd.DataFrame:
    """Parse and compare a sequence of converged CRYSTAL outputs.

    The rows are sorted by isotropic SHRINK value.  Energy differences are
    calculated between successive samplings.  ``delta_E_kJ_mol`` is normalized
    per mole of formula units according to ``formula_units_per_cell``.
    """
    if formula_units_per_cell < 1:
        raise ValueError("formula_units_per_cell must be a positive integer.")

    if not outputs:
        raise ValueError("No CRYSTAL output files were supplied.")

    parsed = [parse_crystal_scf_output(p) for p in outputs]

    for result in parsed:
        if result.isotropic_shrink is None:
            raise ValueError(
                f"{result.path.name}: this teaching analysis expects an "
                f"isotropic SHRINK grid, found {result.shrink}."
            )

    parsed.sort(key=lambda r: r.isotropic_shrink)

    shrink_values = [r.isotropic_shrink for r in parsed]
    if len(shrink_values) != len(set(shrink_values)):
        raise ValueError(
            "Duplicate SHRINK values were found. Supply one output per sampling."
        )

    energies = np.asarray([r.energy_ha for r in parsed], dtype=np.float64)
    delta = np.empty_like(energies)
    delta[:] = np.nan
    if len(energies) > 1:
        delta[1:] = np.abs(np.diff(energies))

    rows = []
    for result, d_e in zip(parsed, delta):
        rows.append(
            {
                "file": result.path.name,
                "SHRINK": result.isotropic_shrink,
                "k-points IBZ": result.kpoints_ibz,
                "energy (Ha)": result.energy_ha,
                "|ΔE| vs previous (Ha)": d_e,
                "|ΔE| vs previous (kJ mol⁻¹)": (
                    d_e * _HARTREE_TO_KJMOL / formula_units_per_cell
                    if np.isfinite(d_e)
                    else np.nan
                ),
                "SCF cycles": result.scf_cycles,
                "elapsed time (s)": result.elapsed_s,
                "CPU time (s)": result.cpu_s,
            }
        )

    return pd.DataFrame(rows)


def plot_energy_convergence(results: pd.DataFrame):
    """Plot successive energy changes in kJ mol-1 versus SHRINK."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(
        results["SHRINK"],
        results["|ΔE| vs previous (kJ mol⁻¹)"],
        marker="o",
    )
    ax.set_xlabel("SHRINK")
    ax.set_ylabel(r"$|\Delta E|$ (kJ mol$^{-1}$)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_computational_cost(results: pd.DataFrame):
    """Plot elapsed wall time versus SHRINK."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(
        results["SHRINK"],
        results["elapsed time (s)"],
        marker="o",
    )
    ax.set_xlabel("SHRINK")
    ax.set_ylabel("Elapsed time (s)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax
