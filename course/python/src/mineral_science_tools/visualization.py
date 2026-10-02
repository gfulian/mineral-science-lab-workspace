"""Interactive CIF visualization with py3Dmol."""

from __future__ import annotations

from pathlib import Path

import py3Dmol


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
        Path to the CIF file.
    supercell
        Replication along a, b, and c.  ``(1, 1, 1)`` shows one unit cell.
    bonds
        If False (default), atoms are shown as spheres only. This is a useful
        conservative default for minerals because automatically inferred
        "bonds" are a visualization convention, not a crystallographic fact.
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")

    cif_text = path.read_text(encoding="utf-8", errors="replace")

    view = py3Dmol.view(width=width, height=height)

    model = view.addModel(
        cif_text,
        "cif",
        {
            "doAssembly": True,
            "duplicateAssemblyAtoms": True,
            "normalizeAssembly": True,
            "wrapAtoms": True,
            "assignBonds": bonds,
            "dontConnectDuplicatedAtoms": not bonds,
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

    view.addUnitCell(model)

    a, b, c = (int(v) for v in supercell)
    if (a, b, c) != (1, 1, 1):
        view.replicateUnitCell(a, b, c, model, bonds)

    view.zoomTo()
    return view
