"""Readable symmetry reports for the Mineral Science teaching notebooks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd

from .structure import read_cif, spglib_dataset


def symmetry_summary(cif_path: str | Path, symprec: float = 1e-5) -> pd.DataFrame:
    """Compare the space group declared in the CIF with spglib's numerical result."""
    structure = read_cif(cif_path)
    sg = structure.spacegroup
    if sg is None:
        raise ValueError("No space group could be determined from the CIF.")

    dataset, _ = spglib_dataset(cif_path, symprec=symprec)

    rows = [
        {
            "source": "CIF / Gemmi",
            "space group (H-M)": sg.xhm(),
            "number": sg.number,
            "point group": sg.point_group_hm(),
            "symmetry operations": len(sg.operations()),
        },
        {
            "source": f"spglib (symprec={symprec:g} Å)",
            "space group (H-M)": dataset.international,
            "number": int(dataset.number),
            "point group": dataset.pointgroup,
            "symmetry operations": len(dataset.rotations),
        },
    ]
    return pd.DataFrame(rows)


def _fraction_string(value: int, denominator: int = gemmi.Op.DEN) -> str:
    f = Fraction(int(value), int(denominator))
    return str(f)


def symmetry_generators(cif_path: str | Path) -> dict[str, pd.DataFrame]:
    """Return Hall-derived generators and lattice-centring translations.

    Gemmi stores the generator operations and centring vectors separately.
    Showing these is usually more useful pedagogically than printing every
    operation of a high-symmetry space group.
    """
    structure = read_cif(cif_path)
    sg = structure.spacegroup
    if sg is None:
        raise ValueError("No space group could be determined from the CIF.")

    generators = gemmi.generators_from_hall(sg.hall)

    op_rows = []
    for i, op in enumerate(generators.sym_ops):
        op_rows.append(
            {
                "index": i,
                "operation": op.triplet(),
                "linear-part type": _rotation_label(op.rot_type(), op),
            }
        )

    cen_rows = []
    for i, vec in enumerate(generators.cen_ops):
        cen_rows.append(
            {
                "index": i,
                "centring translation": (
                    f"({_fraction_string(vec[0])}, "
                    f"{_fraction_string(vec[1])}, "
                    f"{_fraction_string(vec[2])})"
                ),
            }
        )

    return {
        "operations": pd.DataFrame(op_rows),
        "centring": pd.DataFrame(cen_rows),
    }


def _has_translation(op: gemmi.Op) -> bool:
    return any(int(v) % gemmi.Op.DEN != 0 for v in op.tran)


def _rotation_label(rot_type: int, op: gemmi.Op | None = None) -> str:
    has_t = _has_translation(op) if op is not None else False

    if rot_type == 1:
        return "translation" if has_t else "identity"
    if rot_type == -1:
        return "inversion + translation" if has_t else "inversion"
    if rot_type == -2:
        return "mirror/glide-type operation" if has_t else "mirror-type operation"
    if rot_type > 1:
        base = f"{rot_type}-fold rotational part"
    else:
        base = f"{abs(rot_type)}-fold rotoinversion part"

    if has_t:
        base += " + translation"
    return base


def symmetry_operations(
    cif_path: str | Path,
    limit: int | None = 24,
) -> pd.DataFrame:
    """Return space-group operations in coordinate-triplet notation.

    The label describes the linear part and whether a translational component
    is present. It intentionally does not try to assign every operation to a
    crystallographic element (e.g. a particular screw or glide plane), because
    that requires geometrical interpretation beyond the matrix alone.
    """
    structure = read_cif(cif_path)
    sg = structure.spacegroup
    if sg is None:
        raise ValueError("No space group could be determined from the CIF.")

    ops = list(sg.operations())
    if limit is not None:
        ops = ops[:limit]

    rows: list[dict[str, Any]] = []
    for i, op in enumerate(ops):
        rows.append(
            {
                "index": i,
                "operation": op.triplet(),
                "type": _rotation_label(op.rot_type(), op),
            }
        )
    return pd.DataFrame(rows)


def wyckoff_table(cif_path: str | Path, symprec: float = 1e-5) -> pd.DataFrame:
    """Return one row for each symmetrically independent atomic orbit."""
    dataset, expanded = spglib_dataset(cif_path, symprec=symprec)
    equivalent = np.asarray(dataset.equivalent_atoms)

    rows = []
    representatives = []
    for idx in equivalent:
        idx = int(idx)
        if idx not in representatives:
            representatives.append(idx)

    for rep in representatives:
        members = np.where(equivalent == rep)[0]
        i = int(members[0])
        site = expanded[i]
        xyz = np.mod(
            np.asarray([site.fract.x, site.fract.y, site.fract.z], dtype=float),
            1.0,
        )

        rows.append(
            {
                "element": site.element.name,
                "representative label": site.label,
                "multiplicity": len(members),
                "Wyckoff letter": dataset.wyckoffs[i],
                "site symmetry": dataset.site_symmetry_symbols[i],
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }
        )

    return pd.DataFrame(rows)
