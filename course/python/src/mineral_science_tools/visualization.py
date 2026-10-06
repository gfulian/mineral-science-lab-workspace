"""Interactive crystal-structure visualization with Gemmi + py3Dmol.

Gemmi is responsible for crystallographic symmetry expansion and periodic
geometry. py3Dmol is used as the interactive renderer.

Bond and coordination-polyhedron graphics are constructed explicitly by the
course helper instead of relying on py3Dmol's automatic molecular bond
assignment. This is deliberate: automatic bond perception is often unreliable
for inorganic crystals and periodic structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import gemmi
import numpy as np
import py3Dmol
from scipy.spatial import ConvexHull, QhullError

from .structure import read_cif


_DEFAULT_POLYHEDRON_COLORS = {
    "Si": "#F2C94C",
    "Al": "#B7C3D0",
    "Mg": "#6FCF97",
    "Ca": "#56CCF2",
    "Fe": "#EB5757",
    "Ti": "#9B51E0",
    "W":  "#5B6C8F",
    "P":  "#F2994A",
    "S":  "#E2B93B",
    "Zr": "#8E9AAF",
}


def _wrap_fractional(value: float, atol: float = 1e-10) -> float:
    """Map a fractional coordinate to [0, 1), cleaning numerical boundaries."""
    value = float(value) % 1.0
    if np.isclose(value, 0.0, atol=atol) or np.isclose(value, 1.0, atol=atol):
        return 0.0
    return value


def _expanded_sites(cif_path: str | Path):
    structure = read_cif(cif_path)
    sites = list(structure.get_all_unit_cell_sites())
    return structure, sites


def _expanded_p1_cif(cif_path: str | Path) -> str:
    """Return an in-memory P1 CIF containing the complete conventional cell."""
    structure, expanded = _expanded_sites(cif_path)
    cell = structure.cell

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


def _cartesian(cell: gemmi.UnitCell, frac: Sequence[float]) -> np.ndarray:
    p = cell.orthogonalize(
        gemmi.Fractional(float(frac[0]), float(frac[1]), float(frac[2]))
    )
    return np.asarray([p.x, p.y, p.z], dtype=np.float64)


def _display_atoms(
    cif_path: str | Path,
    supercell: tuple[int, int, int],
):
    """Return explicit atoms in the requested conventional-cell supercell."""
    structure, sites = _expanded_sites(cif_path)
    a, b, c = supercell

    atoms = []
    for ia in range(a):
        for ib in range(b):
            for ic in range(c):
                shift = np.asarray([ia, ib, ic], dtype=np.float64)
                for site_index, site in enumerate(sites):
                    frac = np.asarray(
                        [site.fract.x, site.fract.y, site.fract.z],
                        dtype=np.float64,
                    )
                    frac = np.mod(frac, 1.0) + shift
                    atoms.append(
                        {
                            "site_index": site_index,
                            "element": site.element.name,
                            "frac": frac,
                            "cart": _cartesian(structure.cell, frac),
                            "translation": (ia, ib, ic),
                        }
                    )
    return structure, sites, atoms


def _extended_atoms(
    structure,
    sites,
    supercell: tuple[int, int, int],
    shell: int = 1,
):
    """Atoms in the displayed region plus one periodic image shell."""
    a, b, c = supercell
    atoms = []

    for ia in range(-shell, a + shell):
        for ib in range(-shell, b + shell):
            for ic in range(-shell, c + shell):
                shift = np.asarray([ia, ib, ic], dtype=np.float64)
                for site_index, site in enumerate(sites):
                    frac = np.asarray(
                        [site.fract.x, site.fract.y, site.fract.z],
                        dtype=np.float64,
                    )
                    frac = np.mod(frac, 1.0) + shift
                    atoms.append(
                        {
                            "site_index": site_index,
                            "element": site.element.name,
                            "frac": frac,
                            "cart": _cartesian(structure.cell, frac),
                            "translation": (ia, ib, ic),
                        }
                    )
    return atoms


def _pair_key(e1: str, e2: str) -> tuple[str, str]:
    return tuple(sorted((str(e1), str(e2))))


def _distance_cutoff(
    element1: str,
    element2: str,
    *,
    scale: float,
    pair_cutoffs: Mapping[tuple[str, str], float] | None,
) -> float:
    """Distance criterion used only for visualization."""
    key = _pair_key(element1, element2)

    if pair_cutoffs is not None:
        normalized = {
            _pair_key(a, b): float(value)
            for (a, b), value in pair_cutoffs.items()
        }
        if key in normalized:
            return normalized[key]

    r1 = float(gemmi.Element(element1).covalent_r)
    r2 = float(gemmi.Element(element2).covalent_r)
    return float(scale) * (r1 + r2)


def _xyz_dict(xyz: Sequence[float]) -> dict[str, float]:
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
    }


def _draw_distance_bonds(
    view,
    cif_path: str | Path,
    *,
    supercell: tuple[int, int, int],
    scale: float = 1.20,
    pair_cutoffs: Mapping[tuple[str, str], float] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    radius: float = 0.09,
    color: str = "#8A8A8A",
):
    """Draw explicit periodic distance-based bonds as cylinders."""
    structure, sites, displayed = _display_atoms(cif_path, supercell)
    extended = _extended_atoms(structure, sites, supercell, shell=1)

    allowed = None
    if pairs is not None:
        allowed = {_pair_key(a, b) for a, b in pairs}

    seen: set[tuple] = set()

    for atom in displayed:
        p1 = atom["cart"]

        for other in extended:
            if (
                atom["site_index"] == other["site_index"]
                and atom["translation"] == other["translation"]
            ):
                continue

            if allowed is not None and _pair_key(
                atom["element"], other["element"]
            ) not in allowed:
                continue

            p2 = other["cart"]
            distance = float(np.linalg.norm(p2 - p1))
            if distance < 1e-6:
                continue

            cutoff = _distance_cutoff(
                atom["element"],
                other["element"],
                scale=scale,
                pair_cutoffs=pair_cutoffs,
            )
            if distance > cutoff:
                continue

            # Cartesian endpoint key avoids drawing the same periodic bond twice.
            akey = tuple(np.round(p1, 6))
            bkey = tuple(np.round(p2, 6))
            key = tuple(sorted((akey, bkey)))
            if key in seen:
                continue
            seen.add(key)

            view.addCylinder(
                {
                    "start": _xyz_dict(p1),
                    "end": _xyz_dict(p2),
                    "radius": float(radius),
                    "color": color,
                    "fromCap": 1,
                    "toCap": 1,
                }
            )


def _polyhedron_spec(
    value,
) -> dict:
    """Normalize a compact or explicit polyhedron specification."""
    if value is None:
        return {}

    if isinstance(value, str):
        return {"ligands": (value,)}

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(
        "Each polyhedron specification must be a ligand element string "
        "or a mapping."
    )


def _draw_coordination_polyhedra(
    view,
    cif_path: str | Path,
    *,
    supercell: tuple[int, int, int],
    polyhedra: Mapping[str, str | Mapping],
    default_scale: float = 1.25,
    opacity: float = 0.35,
):
    """Draw coordination polyhedra as custom triangular meshes.

    Polyhedra are defined from a central element and its neighbouring ligand
    atoms. The faces are generated with scipy.spatial.ConvexHull.
    """
    structure, sites, displayed = _display_atoms(cif_path, supercell)
    extended = _extended_atoms(structure, sites, supercell, shell=1)

    for center_element, raw_spec in polyhedra.items():
        spec = _polyhedron_spec(raw_spec)
        ligand_elements = tuple(spec.get("ligands", ("O",)))
        cutoff = spec.get("cutoff")
        scale = float(spec.get("scale", default_scale))
        color = spec.get(
            "color",
            _DEFAULT_POLYHEDRON_COLORS.get(center_element, "#80A4ED"),
        )
        poly_opacity = float(spec.get("opacity", opacity))

        for center in displayed:
            if center["element"] != center_element:
                continue

            center_xyz = center["cart"]
            vertices = []

            for ligand in extended:
                if ligand["element"] not in ligand_elements:
                    continue

                distance = float(np.linalg.norm(ligand["cart"] - center_xyz))
                if distance < 1e-6:
                    continue

                if cutoff is None:
                    local_cutoff = min(
                        _distance_cutoff(
                            center_element,
                            ligand["element"],
                            scale=scale,
                            pair_cutoffs=None,
                        ),
                        4.0,
                    )
                else:
                    local_cutoff = float(cutoff)

                if distance <= local_cutoff:
                    vertices.append(ligand["cart"])

            if len(vertices) < 4:
                continue

            # Remove numerical duplicates from periodic enumeration.
            unique = []
            keys = set()
            for vertex in vertices:
                key = tuple(np.round(vertex, 6))
                if key not in keys:
                    keys.add(key)
                    unique.append(vertex)

            if len(unique) < 4:
                continue

            coords = np.asarray(unique, dtype=np.float64)

            try:
                hull = ConvexHull(coords)
            except QhullError:
                # Coplanar or otherwise degenerate coordination geometry:
                # no closed 3D polyhedron can be generated.
                continue

            face_arr = hull.simplices.astype(int).ravel().tolist()
            vertex_arr = [_xyz_dict(v) for v in coords]

            view.addCustom(
                {
                    "vertexArr": vertex_arr,
                    "faceArr": face_arr,
                    "color": color,
                    "opacity": poly_opacity,
                }
            )


def show_structure(
    cif_path: str | Path,
    *,
    supercell: tuple[int, int, int] = (1, 1, 1),
    representation: str = "atoms",
    bonds: bool | None = None,
    bond_scale: float = 1.20,
    bond_cutoffs: Mapping[tuple[str, str], float] | None = None,
    bond_pairs: Sequence[tuple[str, str]] | None = None,
    polyhedra: Mapping[str, str | Mapping] | None = None,
    width: int = 720,
    height: int = 520,
):
    """Display an interactive crystal structure.

    Parameters
    ----------
    cif_path
        Path to the crystallographic CIF.
    supercell
        Replication along a, b, and c.
    representation
        One of:

        ``"atoms"``
            Small spheres only.

        ``"ball_and_stick"``
            Small spheres plus explicitly generated distance-based bonds.

        ``"stick"``
            Very small spheres plus distance-based bond cylinders.

        ``"spacefill"``
            Large spheres, no bonds.

        ``"polyhedra"``
            Small spheres plus coordination polyhedra. Supply the central
            elements with ``polyhedra=...``.

    bonds
        Backward-compatible override. ``True`` maps to ``ball_and_stick`` and
        ``False`` maps to ``atoms``.
    bond_scale
        Automatic bond cutoff multiplier applied to the sum of Gemmi covalent
        radii. This is a *visualization convention*, not a bond-order model.
    bond_cutoffs
        Optional explicit pair-specific cutoffs in ångström, e.g.
        ``{("W", "O"): 2.2}``.
    bond_pairs
        Optional element-pair filter, e.g. ``[("W", "O"), ("Ca", "O")]``.
    polyhedra
        Mapping from central element to ligand specification. Examples:

        ``{"W": "O"}``

        or

        ``{"W": {"ligands": ("O",), "cutoff": 2.2, "color": "#5B6C8F"}}``.

    Notes
    -----
    CIF files do not contain a unique chemical-bond network. Distance bonds
    and coordination polyhedra are therefore visual interpretations. Their
    criteria should be stated whenever they matter scientifically.
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")

    a, b, c = (int(v) for v in supercell)
    if min(a, b, c) < 1:
        raise ValueError("supercell values must be positive integers")

    if bonds is not None:
        representation = "ball_and_stick" if bonds else "atoms"

    representation = representation.lower().replace("-", "_")
    allowed = {"atoms", "ball_and_stick", "stick", "spacefill", "polyhedra"}
    if representation not in allowed:
        raise ValueError(
            f"representation must be one of {sorted(allowed)}; "
            f"got {representation!r}."
        )

    if representation == "polyhedra" and not polyhedra:
        raise ValueError(
            "representation='polyhedra' requires a polyhedra mapping, "
            "for example polyhedra={'W': 'O'}."
        )

    render_cif = _expanded_p1_cif(path)
    view = py3Dmol.view(width=width, height=height)

    # We deliberately disable py3Dmol bond perception. Periodic mineral bonds
    # are generated explicitly below when requested.
    view.addModel(
        render_cif,
        "cif",
        {
            "assignBonds": False,
            "wrapAtoms": True,
        },
    )

    if representation == "spacefill":
        view.setStyle({}, {"sphere": {"scale": 0.95}})
    elif representation == "stick":
        view.setStyle({}, {"sphere": {"scale": 0.12}})
    elif representation == "ball_and_stick":
        view.setStyle({}, {"sphere": {"scale": 0.28}})
    elif representation == "polyhedra":
        view.setStyle({}, {"sphere": {"scale": 0.20}})
    else:
        view.setStyle({}, {"sphere": {"scale": 0.36}})

    view.addUnitCell()

    if (a, b, c) != (1, 1, 1):
        view.replicateUnitCell(a, b, c)

    if representation in {"ball_and_stick", "stick"}:
        _draw_distance_bonds(
            view,
            path,
            supercell=(a, b, c),
            scale=bond_scale,
            pair_cutoffs=bond_cutoffs,
            pairs=bond_pairs,
            radius=0.10 if representation == "ball_and_stick" else 0.13,
        )

    if representation == "polyhedra":
        _draw_coordination_polyhedra(
            view,
            path,
            supercell=(a, b, c),
            polyhedra=polyhedra or {},
        )

    view.zoomTo()
    return view
