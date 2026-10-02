"""Crystal-structure helpers built on Gemmi and spglib.

The functions in this module deliberately expose crystallographic concepts
rather than programming details.  They are intended for teaching notebooks,
not as a replacement for Gemmi or spglib.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
import spglib


def read_cif(cif_path: str | Path) -> gemmi.SmallStructure:
    """Read a small-molecule/inorganic CIF file with Gemmi."""
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")

    structure = gemmi.read_small_structure(str(path))

    if structure.spacegroup is None:
        structure.determine_and_set_spacegroup("S.H2")

    return structure


def _fractional_xyz(site: gemmi.SmallStructure.Site) -> tuple[float, float, float]:
    xyz = np.mod(
        np.array([site.fract.x, site.fract.y, site.fract.z], dtype=float),
        1.0,
    )
    xyz[np.isclose(xyz, 1.0, atol=1e-12)] = 0.0
    xyz[np.isclose(xyz, 0.0, atol=1e-12)] = 0.0
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def asymmetric_unit_table(cif_path: str | Path) -> pd.DataFrame:
    """Return the sites explicitly listed in the CIF asymmetric unit."""
    structure = read_cif(cif_path)

    rows: list[dict[str, Any]] = []
    for site in structure.sites:
        x, y, z = _fractional_xyz(site)
        rows.append(
            {
                "label": site.label,
                "element": site.element.name,
                "x": x,
                "y": y,
                "z": z,
                "occupancy": float(site.occ),
            }
        )

    return pd.DataFrame(rows)


def unit_cell_table(cif_path: str | Path) -> pd.DataFrame:
    """Expand the asymmetric unit and return all sites in one unit cell."""
    structure = read_cif(cif_path)

    rows: list[dict[str, Any]] = []
    for site in structure.get_all_unit_cell_sites():
        x, y, z = _fractional_xyz(site)
        rows.append(
            {
                "label": site.label,
                "element": site.element.name,
                "x": x,
                "y": y,
                "z": z,
                "occupancy": float(site.occ),
            }
        )

    return pd.DataFrame(rows)


def structure_summary(cif_path: str | Path) -> pd.Series:
    """Return a compact crystallographic summary of a CIF structure."""
    structure = read_cif(cif_path)
    cell = structure.cell
    expanded = list(structure.get_all_unit_cell_sites())
    composition = Counter(site.element.name for site in expanded)

    spacegroup = structure.spacegroup
    if spacegroup is None:
        raise ValueError("No space group could be determined from the CIF.")

    check = structure.check_spacegroup()

    return pd.Series(
        {
            "a (Å)": cell.a,
            "b (Å)": cell.b,
            "c (Å)": cell.c,
            "alpha (°)": cell.alpha,
            "beta (°)": cell.beta,
            "gamma (°)": cell.gamma,
            "cell volume (Å³)": cell.volume,
            "crystal system": spacegroup.crystal_system_str(),
            "space group (H-M)": spacegroup.xhm(),
            "space-group number": spacegroup.number,
            "point group (H-M)": spacegroup.point_group_hm(),
            "centring": spacegroup.centring_type(),
            "sites in asymmetric unit": len(structure.sites),
            "atoms in expanded unit cell": len(expanded),
            "unit-cell composition": dict(composition),
            "CIF space-group consistency": "OK" if not check else check,
        },
        name="value",
    )


def _lattice_vectors(structure: gemmi.SmallStructure) -> np.ndarray:
    """Return direct-lattice vectors as rows, in ångström."""
    basis = []
    for xyz in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        p = structure.cell.orthogonalize(gemmi.Fractional(*xyz))
        basis.append([p.x, p.y, p.z])
    return np.asarray(basis, dtype=np.float64)


def _spglib_cell(cif_path: str | Path):
    """Build the (lattice, positions, species) tuple expected by spglib.

    Partial occupancies are intentionally rejected. spglib represents discrete
    atomic species at definite sites; a partially occupied/disordered CIF
    requires an explicit modelling decision rather than silent conversion.
    """
    structure = read_cif(cif_path)

    partial = [
        site for site in structure.sites
        if not np.isclose(float(site.occ), 1.0, atol=1e-8)
    ]
    if partial:
        labels = ", ".join(site.label for site in partial)
        raise ValueError(
            "spglib analysis is disabled for this CIF because partial "
            f"occupancies were found at: {labels}. "
            "Inspect the CIF first and discuss how disorder should be modelled."
        )

    expanded = list(structure.get_all_unit_cell_sites())
    lattice = _lattice_vectors(structure)
    positions = np.asarray([_fractional_xyz(site) for site in expanded], dtype=np.float64)
    numbers = np.asarray([site.element.atomic_number for site in expanded], dtype=np.int32)

    return lattice, positions, numbers, expanded


def spglib_dataset(cif_path: str | Path, symprec: float = 1e-5):
    """Return the spglib symmetry dataset and the expanded Gemmi sites."""
    lattice, positions, numbers, expanded = _spglib_cell(cif_path)
    dataset = spglib.get_symmetry_dataset(
        (lattice, positions, numbers),
        symprec=symprec,
    )
    if dataset is None:
        raise RuntimeError(
            "spglib could not determine the symmetry of this structure. "
            "Check the CIF and the chosen symmetry tolerance."
        )
    return dataset, expanded
