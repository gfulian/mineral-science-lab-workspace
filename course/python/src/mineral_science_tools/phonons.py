"""Teaching helpers for CRYSTAL phonons and Quantas harmonic thermodynamics.

The helpers deliberately keep two scientific ideas separate:

1. Quantas can integrate a complete uniform q mesh directly using equal weights.
2. A complete mesh can optionally be compressed to symmetry-irreducible
   representatives, but the multiplicities must be derived from crystal
   symmetry rather than guessed from the frequencies.

The symmetry reduction functions require q-point coordinates and primitive
structural metadata in the normalized Quantas YAML.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


def prefer_student_output(
    student_path: str | Path,
    reference_path: str | Path,
) -> Path:
    """Use a student's output when present, otherwise use the reference file."""
    student = Path(student_path)
    reference = Path(reference_path)

    if student.exists():
        return student
    if reference.exists():
        return reference

    raise FileNotFoundError(
        "Neither the student calculation nor the reference calculation exists:\n"
        f"  student:   {student}\n"
        f"  reference: {reference}"
    )


def create_ha_yaml(
    crystal_output: str | Path,
    destination: str | Path,
    *,
    jobname: str,
    formula_units: int = 1,
) -> Path:
    """Convert one CRYSTAL phonon output to Quantas' normalized HA YAML."""
    from quantas.api import ha

    return Path(
        ha.create_input(
            Path(crystal_output),
            Path(destination),
            interface="crystal",
            jobname=jobname,
            formula_units=formula_units,
        )
    )


def _load_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return data


def phonon_input_overview(path: str | Path) -> pd.Series:
    """Return a compact overview of a normalized Quantas phonon input."""
    data = _load_yaml(path)
    phonon = data.get("phonon", [])
    weights = np.asarray(
        [float(item.get("weight", np.nan)) for item in phonon],
        dtype=np.float64,
    )

    supercell = data.get("supercell")
    if supercell is None and len(phonon) == 1:
        supercell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    return pd.Series(
        {
            "job": data.get("job", ""),
            "natom": int(data.get("natom", 0)),
            "formula units": int(data.get("formula_units", 1)),
            "q-points stored": int(data.get("qpoints", len(phonon))),
            "sum of q weights": float(np.nansum(weights)),
            "modes per q-point": (
                len(phonon[0].get("band", [])) if phonon else 0
            ),
            "supercell": supercell,
        },
        name="value",
    )


def phonon_frequency_table(
    path: str | Path,
    *,
    q_index: int = 0,
) -> pd.DataFrame:
    """Return frequencies for one q-point from a single-volume HA input."""
    data = _load_yaml(path)
    phonon = data.get("phonon", [])

    if not 0 <= q_index < len(phonon):
        raise IndexError(f"q_index={q_index} is outside the available q-point range.")

    q = phonon[q_index]
    rows = []

    for i, band in enumerate(q.get("band", []), start=1):
        value = band.get("frequency")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 1:
                raise ValueError(
                    "This helper expects a single-volume HA input."
                )
            value = value[0]

        rows.append(
            {
                "branch": i,
                "frequency (cm⁻¹)": float(value),
            }
        )

    table = pd.DataFrame(rows)
    qpos = q.get("q-position")
    table.attrs["q-position"] = qpos
    table.attrs["weight"] = q.get("weight")
    return table


def full_qpoint_table(path: str | Path) -> pd.DataFrame:
    """List the full q mesh stored in a normalized Quantas input."""
    data = _load_yaml(path)
    rows = []

    for i, item in enumerate(data.get("phonon", [])):
        q = item.get("q-position")
        if q is None:
            q = (np.nan, np.nan, np.nan)

        rows.append(
            {
                "index": i,
                "q1": float(q[0]),
                "q2": float(q[1]),
                "q3": float(q[2]),
                "weight": float(item.get("weight", 1.0)),
            }
        )

    return pd.DataFrame(rows)


def _mesh_from_yaml(data: Mapping) -> tuple[int, int, int]:
    phonon = data.get("phonon", [])
    if len(phonon) == 1:
        return (1, 1, 1)

    matrix = data.get("supercell")
    if matrix is None:
        raise ValueError(
            "The normalized YAML does not contain a supercell matrix, "
            "so the uniform q mesh cannot be reconstructed."
        )

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Expected a 3x3 supercell matrix.")

    off_diagonal = matrix - np.diag(np.diag(matrix))
    if not np.allclose(off_diagonal, 0.0, atol=1e-12):
        raise ValueError(
            "The teaching reducer currently supports diagonal supercells only."
        )

    diagonal = np.rint(np.abs(np.diag(matrix))).astype(int)
    if np.any(diagonal < 1):
        raise ValueError("Invalid diagonal supercell matrix.")

    return tuple(int(v) for v in diagonal)


def _primitive_spglib_cell(data: Mapping):
    structure = data.get("structure")
    if not isinstance(structure, Mapping):
        raise ValueError(
            "The normalized YAML has no structural metadata. "
            "Use the full q mesh; Quantas can integrate it directly."
        )

    reference = structure.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError(
            "The normalized YAML has no primitive reference structure."
        )

    lattice = reference.get("lattice")
    positions = reference.get("fractional_positions")
    numbers = reference.get("atomic_numbers")
    if numbers is None:
        numbers = structure.get("atomic_numbers")

    if lattice is None or positions is None or numbers is None:
        raise ValueError(
            "Primitive lattice, fractional positions, or atomic numbers are "
            "missing from the normalized YAML."
        )

    return (
        np.asarray(lattice, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(numbers, dtype=np.int32),
    )


def _qpoint_key(q, mesh: tuple[int, int, int], atol: float = 1e-8):
    q = np.mod(np.asarray(q, dtype=np.float64), 1.0)
    mesh_array = np.asarray(mesh, dtype=np.int64)
    scaled = q * mesh_array
    rounded = np.rint(scaled).astype(np.int64)

    if not np.allclose(scaled, rounded, atol=atol, rtol=0.0):
        raise ValueError(
            f"q-point {q.tolist()} is not commensurate with mesh {mesh}."
        )

    return tuple((rounded % mesh_array).tolist())


def _single_volume_spectrum(item: Mapping) -> np.ndarray:
    frequencies = []
    for band in item.get("band", []):
        value = band.get("frequency")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 1:
                raise ValueError(
                    "Symmetry grouping in this laboratory expects a "
                    "single-volume HA input."
                )
            value = value[0]
        frequencies.append(float(value))
    return np.sort(np.asarray(frequencies, dtype=np.float64))


def irreducible_qpoint_table(
    path: str | Path,
    *,
    symprec: float = 1e-5,
    frequency_tolerance_cm1: float = 0.5,
) -> pd.DataFrame:
    """Group a full uniform q mesh into symmetry-equivalent representatives.

    spglib determines the reciprocal-space mapping from the primitive structure.
    Frequencies are *not* used to decide the weights. They are compared only
    afterwards as a numerical consistency check.
    """
    import spglib

    data = _load_yaml(path)
    phonon = data.get("phonon", [])
    mesh = _mesh_from_yaml(data)

    expected = int(np.prod(mesh))
    if len(phonon) != expected:
        raise ValueError(
            f"Expected the full {mesh} mesh ({expected} q-points), "
            f"but the YAML stores {len(phonon)}."
        )

    for item in phonon:
        if item.get("q-position") is None:
            raise ValueError(
                "At least one q-point has no coordinates; symmetry grouping "
                "cannot be performed safely."
            )

    cell = _primitive_spglib_cell(data)
    mapping, grid_address = spglib.get_ir_reciprocal_mesh(
        mesh,
        cell,
        is_shift=[0, 0, 0],
        is_time_reversal=True,
        symprec=symprec,
    )

    mesh_array = np.asarray(mesh, dtype=np.int64)
    grid_lookup = {}
    for grid_index, address in enumerate(np.asarray(grid_address, dtype=np.int64)):
        key = tuple((address % mesh_array).tolist())
        if key in grid_lookup:
            raise RuntimeError("Duplicate reciprocal-grid address encountered.")
        grid_lookup[key] = grid_index

    groups: dict[int, list[int]] = {}
    for q_index, item in enumerate(phonon):
        key = _qpoint_key(item["q-position"], mesh)
        if key not in grid_lookup:
            raise ValueError(
                f"Could not match q-point {item['q-position']} to the spglib mesh."
            )
        grid_index = grid_lookup[key]
        representative_grid_index = int(mapping[grid_index])
        groups.setdefault(representative_grid_index, []).append(q_index)

    rows = []
    for group_number, (_, members) in enumerate(
        sorted(groups.items(), key=lambda pair: min(pair[1])),
        start=1,
    ):
        representative = members[0]
        rep_item = phonon[representative]
        rep_spectrum = _single_volume_spectrum(rep_item)

        mismatch = 0.0
        for member in members[1:]:
            spectrum = _single_volume_spectrum(phonon[member])
            if spectrum.shape != rep_spectrum.shape:
                raise ValueError("Symmetry-related spectra have different sizes.")
            mismatch = max(
                mismatch,
                float(np.max(np.abs(spectrum - rep_spectrum))),
            )

        q = rep_item["q-position"]
        original_weights = [float(phonon[i].get("weight", 1.0)) for i in members]

        rows.append(
            {
                "irreducible index": group_number,
                "representative full-mesh index": representative,
                "q1": float(q[0]),
                "q2": float(q[1]),
                "q3": float(q[2]),
                "multiplicity": len(members),
                "combined weight": float(np.sum(original_weights)),
                "max spectrum mismatch (cm⁻¹)": mismatch,
                "spectra consistent": mismatch <= frequency_tolerance_cm1,
            }
        )

    table = pd.DataFrame(rows)
    table.attrs["mesh"] = mesh
    table.attrs["full_qpoints"] = len(phonon)
    table.attrs["irreducible_qpoints"] = len(table)
    table.attrs["frequency_tolerance_cm1"] = frequency_tolerance_cm1
    return table


def write_irreducible_phonon_yaml(
    source: str | Path,
    destination: str | Path,
    *,
    symprec: float = 1e-5,
    frequency_tolerance_cm1: float = 0.5,
) -> Path:
    """Write a symmetry-compressed single-volume phonon YAML.

    The operation is intentionally strict.  It refuses to write a reduced file
    if symmetry-related spectra differ beyond the requested tolerance.
    """
    data = _load_yaml(source)
    table = irreducible_qpoint_table(
        source,
        symprec=symprec,
        frequency_tolerance_cm1=frequency_tolerance_cm1,
    )

    if not bool(table["spectra consistent"].all()):
        bad = table.loc[~table["spectra consistent"]]
        raise ValueError(
            "At least one symmetry group has inconsistent spectra. "
            "The reduced YAML will not be written.\n"
            + bad.to_string(index=False)
        )

    original = data["phonon"]
    reduced = []

    for _, row in table.iterrows():
        index = int(row["representative full-mesh index"])
        item = deepcopy(original[index])
        item["weight"] = float(row["combined weight"])
        reduced.append(item)

    data["phonon"] = reduced
    data["qpoints"] = len(reduced)

    provenance = data.setdefault("provenance", {})
    provenance["course_symmetry_reduction"] = {
        "method": "spglib-ir-reciprocal-mesh",
        "source_qpoints": int(table.attrs["full_qpoints"]),
        "irreducible_qpoints": int(table.attrs["irreducible_qpoints"]),
        "mesh": list(table.attrs["mesh"]),
        "symprec": float(symprec),
        "frequency_tolerance_cm-1": float(frequency_tolerance_cm1),
        "weight_rule": "sum of original full-mesh weights",
    }

    destination = Path(destination)
    destination.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def run_ha(
    path: str | Path,
    *,
    temperature_min: float = 0.0,
    temperature_max: float = 1200.0,
    temperature_step: float = 10.0,
):
    """Run Quantas HA and return the public result envelope."""
    from quantas.api import ha

    options = ha.Options(
        temperature_min=float(temperature_min),
        temperature_max=float(temperature_max),
        temperature_step=float(temperature_step),
    )
    return ha.run(Path(path), options=options)


def _find_property_key(inventory, kind: str) -> str:
    kind = kind.lower()

    for item in inventory.properties:
        name = str(item.name).lower()
        symbol_plain = str(item.symbol_plain).lower().replace("_", "")
        symbol_math = str(item.symbol_math).lower().replace("_", "")

        if kind == "cv":
            if "isochoric" in name or symbol_plain == "cv" or symbol_math == "cv":
                return item.key
        elif kind == "entropy":
            if "entropy" in name or symbol_plain == "s":
                return item.key
        else:
            if kind in name or kind == str(item.key).lower():
                return item.key

    available = ", ".join(
        f"{item.key} ({item.name})" for item in inventory.properties
    )
    raise KeyError(f"Could not find HA property '{kind}'. Available: {available}")


def ha_property_curve(
    result_envelope,
    *,
    kind: str = "cv",
    energy_unit: str = "J/mol",
) -> pd.DataFrame:
    """Extract a molar temperature curve through Quantas' public plot contract.

    Using Quantas' plot builder is deliberate: it applies the package's own
    normalization and unit conversion instead of inferring units from raw array
    magnitudes.
    """
    from quantas.api import ha

    inventory = ha.describe_plots(result_envelope)
    key = _find_property_key(inventory, kind)

    plots = ha.build_plots(
        result_envelope,
        properties=[key],
        unit=energy_unit,
    )

    candidates = []
    for plot in plots.plots:
        series = getattr(plot, "series", None)
        if series:
            candidates.extend(series)

    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one HA series for a single-volume calculation; "
            f"found {len(candidates)}."
        )

    series = candidates[0]
    x = np.asarray(series.x, dtype=np.float64)
    y = np.asarray(series.y, dtype=np.float64)

    if kind.lower() == "cv":
        y_name = "Cv (J mol⁻¹ K⁻¹)"
    elif kind.lower() == "entropy":
        y_name = "S (J mol⁻¹ K⁻¹)"
    else:
        y_name = str(getattr(plots.plots[0].y_axis, "label", kind))

    return pd.DataFrame(
        {
            "T (K)": x,
            y_name: y,
        }
    )


def ha_cv_curve(
    path: str | Path,
    *,
    temperature_min: float = 0.0,
    temperature_max: float = 1200.0,
    temperature_step: float = 10.0,
) -> pd.DataFrame:
    """Convenience wrapper returning Quantas harmonic Cv in J mol-1 K-1."""
    result = run_ha(
        path,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        temperature_step=temperature_step,
    )
    return ha_property_curve(result, kind="cv", energy_unit="J/mol")


def load_experimental_heat_capacity(path: str | Path) -> pd.DataFrame:
    """Load the course MgO experimental heat-capacity CSV."""
    return pd.read_csv(path)


def plot_cv_comparison(
    curves: Mapping[str, pd.DataFrame],
    experimental: pd.DataFrame | None = None,
    *,
    dulong_petit: bool = True,
):
    """Plot calculated harmonic Cv curves and optional experimental Cp data."""
    import matplotlib.pyplot as plt
    from scipy.constants import R

    fig, ax = plt.subplots()

    for label, table in curves.items():
        ax.plot(
            table["T (K)"],
            table["Cv (J mol⁻¹ K⁻¹)"],
            label=label,
        )

    if experimental is not None:
        ax.scatter(
            experimental["T (K)"],
            experimental["Cp experimental (J mol⁻¹ K⁻¹)"],
            marker="o",
            label="experiment (approximately Cp)",
        )

    if dulong_petit:
        ax.axhline(
            6.0 * R,
            linestyle="--",
            linewidth=1.0,
            label="Dulong–Petit: 6R",
        )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat capacity (J mol⁻¹ K⁻¹)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def convergence_table(
    curves: Mapping[str, pd.DataFrame],
    *,
    temperatures: Sequence[float] = (100.0, 300.0, 1000.0),
    reference_label: str | None = None,
) -> pd.DataFrame:
    """Sample calculated Cv curves at selected temperatures."""
    rows = []

    if reference_label is None:
        reference_label = list(curves)[-1]

    ref = curves[reference_label]
    ref_t = ref["T (K)"].to_numpy(dtype=np.float64)
    ref_cv = ref["Cv (J mol⁻¹ K⁻¹)"].to_numpy(dtype=np.float64)

    for label, table in curves.items():
        t = table["T (K)"].to_numpy(dtype=np.float64)
        cv = table["Cv (J mol⁻¹ K⁻¹)"].to_numpy(dtype=np.float64)

        for target in temperatures:
            value = float(np.interp(target, t, cv))
            reference = float(np.interp(target, ref_t, ref_cv))
            rows.append(
                {
                    "sampling": label,
                    "T (K)": float(target),
                    "Cv (J mol⁻¹ K⁻¹)": value,
                    f"ΔCv vs {reference_label} (J mol⁻¹ K⁻¹)": value - reference,
                }
            )

    return pd.DataFrame(rows)

def ha_entropy_curve(
    path: str | Path,
    *,
    temperature_min: float = 0.0,
    temperature_max: float = 1200.0,
    temperature_step: float = 10.0,
) -> pd.DataFrame:
    """Convenience wrapper returning Quantas harmonic entropy in J mol-1 K-1."""
    result = run_ha(
        path,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        temperature_step=temperature_step,
    )
    return ha_property_curve(result, kind="entropy", energy_unit="J/mol")


def load_experimental_entropy(path: str | Path) -> pd.DataFrame:
    """Load the course MgO experimental/reference entropy CSV."""
    return pd.read_csv(path)


def plot_entropy_comparison(
    curves: Mapping[str, pd.DataFrame],
    experimental: pd.DataFrame | None = None,
):
    """Plot calculated harmonic entropy curves and optional experimental S° data."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for label, table in curves.items():
        ax.plot(
            table["T (K)"],
            table["S (J mol⁻¹ K⁻¹)"],
            label=label,
        )

    if experimental is not None:
        ax.scatter(
            experimental["T (K)"],
            experimental["S experimental/reference (J mol⁻¹ K⁻¹)"],
            marker="o",
            label="experiment/reference S° (1 bar)",
        )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Entropy (J mol⁻¹ K⁻¹)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def entropy_convergence_table(
    curves: Mapping[str, pd.DataFrame],
    *,
    temperatures: Sequence[float] = (300.0, 600.0, 1000.0),
    reference_label: str | None = None,
) -> pd.DataFrame:
    """Sample harmonic entropy curves at selected temperatures."""
    rows = []

    if reference_label is None:
        reference_label = list(curves)[-1]

    ref = curves[reference_label]
    ref_t = ref["T (K)"].to_numpy(dtype=np.float64)
    ref_s = ref["S (J mol⁻¹ K⁻¹)"].to_numpy(dtype=np.float64)

    for label, table in curves.items():
        t = table["T (K)"].to_numpy(dtype=np.float64)
        entropy = table["S (J mol⁻¹ K⁻¹)"].to_numpy(dtype=np.float64)

        for target in temperatures:
            value = float(np.interp(target, t, entropy))
            reference = float(np.interp(target, ref_t, ref_s))
            rows.append(
                {
                    "sampling": label,
                    "T (K)": float(target),
                    "S (J mol⁻¹ K⁻¹)": value,
                    f"ΔS vs {reference_label} (J mol⁻¹ K⁻¹)": value - reference,
                }
            )

    return pd.DataFrame(rows)

