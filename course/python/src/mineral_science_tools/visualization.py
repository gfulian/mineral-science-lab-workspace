"""Interactive crystal-structure visualization with Gemmi + py3Dmol.

Gemmi is responsible for crystallographic symmetry expansion. py3Dmol is used
only as the interactive renderer. This separation is deliberate: a CIF may
declare a space group without explicitly listing every symmetry operation, and
the 3Dmol CIF parser does not necessarily reconstruct the complete unit cell
from the Hermann–Mauguin symbol alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py3Dmol

from .structure import read_cif


def _wrap_fractional(value: float, atol: float = 1e-10) -> float:
    """Map a fractional coordinate to [0, 1), cleaning numerical boundaries."""
    value = float(value) % 1.0
    if np.isclose(value, 0.0, atol=atol) or np.isclose(value, 1.0, atol=atol):
        return 0.0
    return value


def _expanded_p1_cif(cif_path: str | Path) -> str:
    """Return an in-memory P1 CIF containing the complete conventional cell.

    The original CIF is read and symmetry-expanded with Gemmi. The expanded
    positions are then written explicitly into a temporary P1 representation
    used only for visualization.

    This does *not* change the crystallographic interpretation used elsewhere
    in the laboratory: it simply gives the viewer every atom that must be drawn.
    """
    structure = read_cif(cif_path)
    cell = structure.cell

    expanded = list(structure.get_all_unit_cell_sites())

    lines = [
        "data_visualization",
        "_audit_creation_method 'Generated in memory by mineral_science_tools'",
        f"_cell_length_a {cell.a:.10f}",
        f"_cell_length_b {cell.b:.10f}",
        f"_cell_length_c {cell.c:.10f}",
        f"_cell_angle_alpha {cell.alpha:.10f}",
        f"_cell_angle_beta {cell.beta:.10f}",
        f"_cell_angle_gamma {cell.gamma:.10f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x,y,z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    for i, site in enumerate(expanded, start=1):
        x = _wrap_fractional(site.fract.x)
        y = _wrap_fractional(site.fract.y)
        z = _wrap_fractional(site.fract.z)

        label = f"{site.element.name}{i}"
        lines.append(
            f"{label} {site.element.name} "
            f"{x:.10f} {y:.10f} {z:.10f} {float(site.occ):.6f}"
        )

    return "\n".join(lines) + "\n"


def show_structure(
    cif_path: str | Path,
    *,
    supercell: tuple[int, int, int] = (1, 1, 1),
    bonds: bool = False,
    width: int = 720,
    height: int = 520,
):
    """Display an interactive crystal structure inside a Jupyter notebook.

    Parameters
    ----------
    cif_path
        Path to the original crystallographic CIF.
    supercell
        Replication along a, b, and c. ``(1, 1, 1)`` shows one conventional
        unit cell.
    bonds
        If False (default), atoms are shown as spheres only. Automatically
        inferred sticks are a visualization convention, not crystallographic
        evidence of bonding.
    width, height
        Viewer size in pixels.

    Notes
    -----
    Gemmi expands the asymmetric unit before anything is passed to py3Dmol.
    The viewer therefore receives an explicit P1 representation of the complete
    conventional cell and does not need to infer crystallographic symmetry.
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")

    a, b, c = (int(v) for v in supercell)
    if min(a, b, c) < 1:
        raise ValueError("supercell values must be positive integers")

    render_cif = _expanded_p1_cif(path)

    view = py3Dmol.view(width=width, height=height)

    view.addModel(
        render_cif,
        "cif",
        {
            "assignBonds": bonds,
            "wrapAtoms": True,
        },
    )

    if bonds:
        view.setStyle(
            {},
            {
                "sphere": {"scale": 0.28},
                "stick": {"radius": 0.10},
            },
        )
    else:
        view.setStyle({}, {"sphere": {"scale": 0.36}})

    # The in-memory CIF preserves the original unit-cell parameters.
    view.addUnitCell()

    if (a, b, c) != (1, 1, 1):
        view.replicateUnitCell(a, b, c)

    view.zoomTo()
    return view
