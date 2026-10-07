"""Teaching helpers for Mineral Science Laboratory 5.

The module uses public Quantas APIs for native results. Experimental/reference
points are rendered in black so that they remain visually distinct from
computational curves.

The helpers intentionally keep *inspection* and *extraction* separate:
students should first inspect representative CRYSTAL files manually, then use
the parser-generated summaries as a consistency check.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
from functools import lru_cache
import re

import numpy as np
import pandas as pd


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def validate_reference_list(path: str | Path) -> pd.DataFrame:
    """Validate every non-empty path in the distributed Quantas list file."""
    path = Path(path)
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = []
    for entry in entries:
        p = Path(entry)
        rows.append(
            {
                "listed file": entry,
                "exists": p.exists(),
                "size (MB)": p.stat().st_size / 1024**2 if p.exists() else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    missing = table.loc[~table["exists"], "listed file"].tolist()
    if missing:
        raise FileNotFoundError(
            "The QHA list contains paths that do not exist from the laboratory "
            "working directory:\n  - " + "\n  - ".join(missing)
        )
    return table


def inspect_reference_input(path: str | Path) -> dict:
    """Extract selected settings from one distributed CRYSTAL input."""
    path = Path(path)
    text = _read(path)
    lines = text.splitlines()

    a = float(lines[4].strip())
    basis = re.search(r"BASISSET\s*\n\s*(\S+)", text, re.I)
    functional = re.search(r"\nDFT\s*\n\s*(\S+)", text, re.I)
    shrink = re.search(r"SHRINK\s*\n\s*(\d+)\s+(\d+)", text, re.I)
    sc = re.search(
        r"SCELPHONO\s*\n"
        r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\n"
        r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\n"
        r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)",
        text,
        re.I,
    )
    matrix = (
        np.asarray([int(x) for x in sc.groups()], dtype=int).reshape(3, 3)
        if sc
        else None
    )

    m = re.search(r"_v(\d{3})_", path.name)
    vf = float(int(m.group(1)) / 100) if m else np.nan

    return {
        "file": path.name,
        "V/Vref label": vf,
        "a conventional (A)": a,
        "V conventional (A^3)": a**3,
        "SCELPHONO": matrix.tolist() if matrix is not None else None,
        "primitive repetitions": (
            int(round(abs(np.linalg.det(matrix)))) if matrix is not None else np.nan
        ),
        "SHRINK": f"{shrink.group(1)} {shrink.group(2)}" if shrink else "",
        "functional": functional.group(1) if functional else "",
        "basis": basis.group(1) if basis else "",
    }


def inspect_reference_output(path: str | Path) -> dict:
    """Extract a few size/sampling diagnostics from one CRYSTAL output."""
    path = Path(path)
    text = _read(path)

    def integer(pattern: str):
        m = re.search(pattern, text, re.I)
        return int(m.group(1)) if m else np.nan

    return {
        "output": path.name,
        "atoms in phonon supercell": integer(
            r"NUMBER OF ATOMS PER SUPERCELL\s+(\d+)"
        ),
        "q-points": integer(
            r"PERMITS THE CALCULATION OF MODES AT\s+(\d+)\s+K POINTS"
        ),
        "atomic orbitals": integer(r"NUMBER OF AO\s+(\d+)"),
        "shells": integer(r"NUMBER OF SHELLS\s+(\d+)"),
        "normal termination": "EEEEEEEEEE TERMINATION" in text,
    }


def reference_series_table(
    folder: str | Path = "reference-results/raw",
) -> pd.DataFrame:
    """Summarize the reference series after students have inspected raw files."""
    folder = Path(folder)
    rows = []
    for inp in sorted(folder.glob("mgo_v*_phonons.d12")):
        row = inspect_reference_input(inp)
        out = inp.with_suffix(".out")
        if out.exists():
            row.update(inspect_reference_output(out))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        ref = df.loc[
            (df["V/Vref label"] - 1).abs().idxmin(),
            "V conventional (A^3)",
        ]
        df["V/Vref from cell"] = df["V conventional (A^3)"] / ref
    return df


def load_qha(path: str | Path):
    from quantas.api import qha

    envelope = qha.read_result(Path(path))
    return envelope, qha.get_result(envelope)


def _field(payload, *names):
    for name in names:
        if hasattr(payload, name):
            return np.asarray(getattr(payload, name), dtype=np.float64)
    available = [n for n in dir(payload) if not n.startswith("_")]
    raise AttributeError(
        f"None of {names!r} found in QHA payload. "
        f"Available fields include: {available}"
    )


def nearest_index(values, target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.nanargmin(np.abs(values - float(target))))


_QHA_GRID_ALIASES = {
    "volume": ("equilibrium_volume",),
    "alpha": ("thermal_expansion",),
    "KT": ("isothermal_bulk_modulus",),
    "KS": ("adiabatic_bulk_modulus",),
    "Kprime": (
        "bulk_modulus_pressure_derivative",
        "isothermal_bulk_modulus_pressure_derivative",
    ),
    "Cv": ("isochoric_heat_capacity",),
    "Cp": ("isobaric_heat_capacity",),
    "entropy": ("entropy",),
    "enthalpy": ("enthalpy",),
}


@lru_cache(maxsize=12)
def _cached_qha_grid_arrays(
    resolved_path: str,
    mtime_ns: int,
) -> dict[str, np.ndarray]:
    """Load one Quantas result once and cache its common grid arrays."""
    _, payload = load_qha(resolved_path)

    arrays: dict[str, np.ndarray] = {
        "temperature": _field(payload, "temperature"),
        "pressure": _field(payload, "pressure"),
    }

    for key, aliases in _QHA_GRID_ALIASES.items():
        try:
            arrays[key] = _field(payload, *aliases)
        except AttributeError:
            pass

    return arrays


def _qha_grid_arrays(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_qha_grid_arrays(str(path), stat.st_mtime_ns)


def qha_grid_field(
    path: str | Path,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native QHA grid arrays without repeatedly reopening the HDF5 file."""
    if quantity not in _QHA_GRID_ALIASES:
        raise KeyError(quantity)

    arrays = _qha_grid_arrays(path)
    if quantity not in arrays:
        raise AttributeError(
            f"QHA result does not expose the requested property {quantity!r}."
        )

    return arrays["temperature"], arrays["pressure"], arrays[quantity]


def qha_section(
    path: str | Path,
    quantity: str,
    *,
    pressure: float = 0.0,
) -> pd.DataFrame:
    temperature, pressures, values = qha_grid_field(path, quantity)
    j = nearest_index(pressures, pressure)
    return pd.DataFrame(
        {
            "T (K)": temperature,
            quantity: values[:, j],
            "P (GPa)": pressures[j],
        }
    )


def qha_isotherm(
    path: str | Path,
    quantity: str,
    *,
    temperature: float = 300.0,
) -> pd.DataFrame:
    temperatures, pressure, values = qha_grid_field(path, quantity)
    i = nearest_index(temperatures, temperature)
    return pd.DataFrame(
        {
            "P (GPa)": pressure,
            quantity: values[i, :],
            "T (K)": temperatures[i],
        }
    )


def qha_inventory(path: str | Path) -> pd.DataFrame:
    """List plot properties advertised by the installed Quantas version."""
    from quantas.api import qha

    result = qha.read_result(Path(path))
    inventory = qha.describe_plots(result)
    return pd.DataFrame(
        [
            {
                "key": item.key,
                "name": item.name,
                "symbol": item.symbol_plain,
                "unit": item.unit,
                "category": item.category,
                "components": ", ".join(item.components or ()),
            }
            for item in inventory.properties
        ]
    )


def _find_plot_key(inventory, words: Sequence[str]) -> str:
    words = [w.lower() for w in words]
    for descriptor in inventory.properties:
        haystack = " ".join(
            [
                str(descriptor.key),
                str(descriptor.name),
                str(descriptor.symbol_plain),
                str(descriptor.symbol_math),
            ]
        ).lower()
        if all(word in haystack for word in words):
            return descriptor.key
    raise KeyError(
        f"Could not find a QHA plot property containing {words}. "
        f"Available keys: {[d.key for d in inventory.properties]}"
    )


_SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "ₚ": "p",
        "ᵥ": "v",
        "ₜ": "t",
        "ₛ": "s",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
    }
)


def _normalise_label(value) -> str:
    text = str(value).translate(_SUBSCRIPT_TRANSLATION).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _flatten_plot_series(collection):
    series = []
    for plot in collection.plots:
        direct = getattr(plot, "series", None) or []
        for item in direct:
            series.append((plot, item))

        # Panel plots may contain their series inside panels.
        for panel in getattr(plot, "panels", None) or []:
            for item in getattr(panel, "series", None) or []:
                series.append((plot, item))
    return series


def _single_pressure_plot_series(
    path: str | Path,
    words: Sequence[str],
    *,
    energy_unit: str = "J/mol",
):
    """Extract one display-ready scalar property line from a 0-GPa result."""
    from quantas.api import qha

    result = qha.read_result(Path(path))
    inventory = qha.describe_plots(result)
    key = _find_plot_key(inventory, words)
    plots = qha.build_plots(
        result,
        properties=[key],
        options=qha.PlotOptions(energy_unit=energy_unit),
    )
    series = _flatten_plot_series(plots)

    if len(series) != 1:
        labels = [
            f"{getattr(s, 'key', '')!s}: {getattr(s, 'label', '')!s}"
            for _, s in series
        ]
        raise ValueError(
            f"Expected one series for property {key!r}; found {len(series)}. "
            f"Series: {labels}"
        )

    plot, item = series[0]
    return pd.DataFrame(
        {
            "T (K)": np.asarray(item.x, dtype=float),
            "value": np.asarray(item.y, dtype=float),
        }
    ), plot.y_axis.label


def _heat_capacity_series(
    path: str | Path,
    component: str,
) -> pd.DataFrame:
    """Extract Cp or Cv from Quantas' stable `heat_capacities` plot property.

    Quantas exposes Cp and Cv together.  Series names can contain underscores,
    Unicode subscripts, or words such as 'isobaric'; selection is therefore
    normalized rather than matched literally against 'Cp'.
    """
    from quantas.api import qha

    component = component.strip().lower()
    if component not in {"cp", "cv"}:
        raise ValueError("component must be 'Cp' or 'Cv'")

    result = qha.read_result(Path(path))
    inventory = qha.describe_plots(result)
    keys = {item.key for item in inventory.properties}
    if "heat_capacities" not in keys:
        raise KeyError(
            "The installed Quantas result does not advertise the documented "
            "'heat_capacities' plot property. Available keys: "
            + ", ".join(sorted(keys))
        )

    plots = qha.build_plots(
        result,
        properties=["heat_capacities"],
        options=qha.PlotOptions(energy_unit="J/mol"),
    )
    series = _flatten_plot_series(plots)

    wanted_tokens = (
        ("cp", "isobaric", "constantpressure")
        if component == "cp"
        else ("cv", "isochoric", "constantvolume")
    )

    matches = []
    diagnostics = []
    for plot, item in series:
        raw = " ".join(
            [
                str(getattr(item, "key", "")),
                str(getattr(item, "label", "")),
                str(getattr(item, "name", "")),
            ]
        )
        norm = _normalise_label(raw)
        diagnostics.append(raw)
        if any(token in norm for token in wanted_tokens):
            matches.append((plot, item))

    if len(matches) != 1:
        raise ValueError(
            f"Could not identify a unique {component.upper()} series inside "
            f"'heat_capacities'. Found {len(matches)} matches. "
            f"Available series metadata: {diagnostics}"
        )

    _, item = matches[0]
    col = "Cp (J mol^-1 K^-1)" if component == "cp" else "Cv (J mol^-1 K^-1)"
    return pd.DataFrame(
        {
            "T (K)": np.asarray(item.x, dtype=float),
            col: np.asarray(item.y, dtype=float),
        }
    )


def entropy_curve(path: str | Path) -> pd.DataFrame:
    table, _ = _single_pressure_plot_series(
        path,
        ["entropy"],
        energy_unit="J/mol",
    )
    return table.rename(columns={"value": "S (J mol^-1 K^-1)"})


def heat_capacity_curve(
    path: str | Path,
    component: str = "Cp",
) -> pd.DataFrame:
    return _heat_capacity_series(path, component)


def enthalpy_increment_curve(
    path: str | Path,
    *,
    reference_temperature: float = 298.15,
) -> pd.DataFrame:
    table, _ = _single_pressure_plot_series(
        path,
        ["enthalpy"],
        energy_unit="kJ/mol",
    )
    table = table.rename(columns={"value": "H (kJ mol^-1)"})
    href = float(
        np.interp(
            float(reference_temperature),
            table["T (K)"].to_numpy(dtype=float),
            table["H (kJ mol^-1)"].to_numpy(dtype=float),
        )
    )
    table["H-Href (kJ mol^-1)"] = table["H (kJ mol^-1)"] - href
    table.attrs["reference_temperature"] = float(reference_temperature)
    return table


def _black_experiment(ax, x, y, label, **kwargs):
    defaults = dict(
        marker="o",
        s=26,
        color="black",
        edgecolors="black",
        zorder=5,
    )
    defaults.update(kwargs)
    ax.scatter(x, y, label=label, **defaults)


def method_comparison(
    poly_path,
    eos_path,
    *,
    temperatures=(300.0, 1000.0, 1500.0),
    pressure=0.0,
) -> pd.DataFrame:
    """Compare polynomial and BM3 results while reading each HDF5 only once."""
    rows = []

    datasets = [
        ("polynomial", _qha_grid_arrays(poly_path)),
        ("BM3 EOS", _qha_grid_arrays(eos_path)),
    ]

    required = {
        "volume": "V (A^3 primitive)",
        "KT": "KT (GPa)",
        "KS": "KS (GPa)",
        "alpha": "alphaV (K^-1)",
    }

    for method, arrays in datasets:
        t = arrays["temperature"]
        p = arrays["pressure"]
        j = nearest_index(p, pressure)

        missing = [key for key in required if key not in arrays]
        if missing:
            raise AttributeError(
                f"{method} result is missing required properties: {missing}"
            )

        for temperature in temperatures:
            i = nearest_index(t, temperature)
            row = {
                "method": method,
                "T requested (K)": float(temperature),
                "T actual (K)": float(t[i]),
                "P actual (GPa)": float(p[j]),
            }
            for key, out in required.items():
                row[out] = float(arrays[key][i, j])
            rows.append(row)

    return pd.DataFrame(rows)


def plot_method_comparison(
    poly_path,
    eos_path,
    quantity: str,
    *,
    pressure=0.0,
):
    """Plot polynomial and BM3 QHA sections for one property."""
    import matplotlib.pyplot as plt

    labels = {
        "volume": ("Volume (Å³, primitive cell)", "V"),
        "KT": ("Isothermal bulk modulus (GPa)", r"$K_T$"),
        "alpha": ("Volumetric thermal expansion (K⁻¹)", r"$\alpha_V$"),
    }
    if quantity not in labels:
        raise KeyError(quantity)

    fig, ax = plt.subplots()
    for label, path in [("polynomial", poly_path), ("BM3 EOS", eos_path)]:
        table = qha_section(path, quantity, pressure=pressure)
        ax.plot(table["T (K)"], table[quantity], label=label)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(labels[quantity][0])
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_absolute_volume(
    qha_path,
    experimental,
    *,
    pressure=0.0,
    volume_multiplier=4.0,
    calculation_label="QHA BM3",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    table = qha_section(qha_path, "volume", pressure=pressure)
    ax.plot(
        table["T (K)"],
        table["volume"] * volume_multiplier,
        label=calculation_label,
    )
    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental["conventional cell volume (A^3)"],
        "experiment",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Conventional cell volume (Å³)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_normalized_volume(
    qha_path,
    experimental,
    *,
    pressure=0.0,
    volume_multiplier=4.0,
    reference_temperature=300.0,
    calculation_label="QHA BM3",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    table = qha_section(qha_path, "volume", pressure=pressure)
    calculated = table["volume"].to_numpy() * volume_multiplier
    i = nearest_index(table["T (K)"], reference_temperature)
    ax.plot(
        table["T (K)"],
        calculated / calculated[i],
        label=calculation_label,
    )

    ex = experimental.copy()
    j = nearest_index(ex["T (K)"], reference_temperature)
    ref = float(ex.iloc[j]["conventional cell volume (A^3)"])
    _black_experiment(
        ax,
        ex["T (K)"],
        ex["conventional cell volume (A^3)"] / ref,
        "experiment",
    )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("V / V(300 K)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_alpha(
    qha_path,
    experimental,
    *,
    pressure=0.0,
    calculation_label="QHA BM3",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    table = qha_section(qha_path, "alpha", pressure=pressure)
    ax.plot(table["T (K)"], table["alpha"], label=calculation_label)
    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental["volumetric alpha derived (K^-1)"],
        "experiment: 3 × linear alpha",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Volumetric thermal expansion (K⁻¹)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_bulk_moduli(
    qha_path,
    experimental,
    *,
    pressure=0.0,
    calculation_label="QHA BM3",
):
    """Compare theoretical KT/KS with matched experimental KT/KS series."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    kt = qha_section(qha_path, "KT", pressure=pressure)
    ks = qha_section(qha_path, "KS", pressure=pressure)
    ax.plot(kt["T (K)"], kt["KT"], label=f"{calculation_label} $K_T$")
    ax.plot(ks["T (K)"], ks["KS"], label=f"{calculation_label} $K_S$")

    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental["KT experimental (GPa)"],
        "experimental $K_T$",
        marker="o",
    )
    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental["KS from adiabatic Cij (GPa)"],
        "experimental $K_S$",
        marker="s",
        facecolors="none",
        linewidths=1.1,
    )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Bulk modulus (GPa)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_thermochemistry(
    qha_path,
    experimental,
    quantity: str,
    *,
    calculation_label="QHA BM3",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if quantity == "S":
        th = entropy_curve(qha_path)
        x = th["T (K)"]
        y = th["S (J mol^-1 K^-1)"]
        excol = "S experimental/reference (J mol^-1 K^-1)"
        ylabel = "Entropy (J mol⁻¹ K⁻¹)"
        label = f"{calculation_label} S"

    elif quantity == "Cp":
        th = heat_capacity_curve(qha_path, "Cp")
        x = th["T (K)"]
        y = th["Cp (J mol^-1 K^-1)"]
        excol = "Cp experimental/reference (J mol^-1 K^-1)"
        ylabel = "Heat capacity (J mol⁻¹ K⁻¹)"
        label = f"{calculation_label} $C_P$"

    elif quantity == "H":
        th = enthalpy_increment_curve(qha_path)
        x = th["T (K)"]
        y = th["H-Href (kJ mol^-1)"]
        excol = "H-H298.15 experimental/reference (kJ mol^-1)"
        ylabel = "H(T) − H(298 K) (kJ mol⁻¹)"
        label = f"{calculation_label} ΔH"

    else:
        raise KeyError(quantity)

    ax.plot(x, y, label=label)
    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental[excol],
        "NIST/JANAF",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_room_temperature_compression(
    grid_path,
    experimental_eos,
    *,
    temperature=300.0,
    volume_multiplier=4.0,
    calculation_label="QHA BM3",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    table = qha_isotherm(grid_path, "volume", temperature=temperature)
    ax.plot(
        table["P (GPa)"],
        table["volume"] * volume_multiplier,
        label=calculation_label,
    )
    _black_experiment(
        ax,
        experimental_eos["P (GPa)"],
        experimental_eos[
            "conventional cell volume from experimental BM3 fit (A^3)"
        ],
        "Speziale et al. experimental BM3 fit",
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Conventional cell volume (Å³)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_pt_map(
    path: str | Path,
    quantity: str,
    *,
    volume_multiplier: float = 4.0,
):
    import matplotlib.pyplot as plt

    temperature, pressure, values = qha_grid_field(path, quantity)

    if quantity == "volume":
        values = values * volume_multiplier
        label = "Conventional cell volume (Å³)"
    elif quantity == "alpha":
        label = "Volumetric thermal expansion (K⁻¹)"
    elif quantity == "KT":
        label = "Isothermal bulk modulus (GPa)"
    else:
        label = quantity

    fig, ax = plt.subplots()
    contour = ax.contourf(temperature, pressure, values.T, levels=14)
    fig.colorbar(contour, ax=ax, label=label)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (GPa)")
    fig.tight_layout()
    return fig, ax

def compare_qha_models_table(
    pbe_path: str | Path,
    b3lyp_path: str | Path,
    *,
    temperatures=(300.0, 1000.0, 1500.0),
    pressure: float = 0.0,
) -> pd.DataFrame:
    """Compact numerical comparison of two QHA electronic-structure models."""
    rows = []
    for model, path in [("PBE", pbe_path), ("B3LYP", b3lyp_path)]:
        arrays = _qha_grid_arrays(path)
        t = arrays["temperature"]
        p = arrays["pressure"]
        j = nearest_index(p, pressure)

        for target in temperatures:
            i = nearest_index(t, target)
            rows.append(
                {
                    "model": model,
                    "T (K)": float(t[i]),
                    "P (GPa)": float(p[j]),
                    "V conventional (A^3)": float(arrays["volume"][i, j]) * 4.0,
                    "alphaV (K^-1)": float(arrays["alpha"][i, j]),
                    "KT (GPa)": float(arrays["KT"][i, j]),
                    "KS (GPa)": float(arrays["KS"][i, j]),
                }
            )
    return pd.DataFrame(rows)


def plot_model_thermochemistry(
    pbe_path: str | Path,
    b3lyp_path: str | Path,
    experimental: pd.DataFrame,
    quantity: str,
):
    """Plot PBE and B3LYP QHA thermochemistry with black reference points."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if quantity == "S":
        pbe = entropy_curve(pbe_path)
        b3 = entropy_curve(b3lyp_path)
        xcol = "T (K)"
        ycol = "S (J mol^-1 K^-1)"
        excol = "S experimental/reference (J mol^-1 K^-1)"
        ylabel = "Entropy (J mol⁻¹ K⁻¹)"
    elif quantity == "Cp":
        pbe = heat_capacity_curve(pbe_path, "Cp")
        b3 = heat_capacity_curve(b3lyp_path, "Cp")
        xcol = "T (K)"
        ycol = "Cp (J mol^-1 K^-1)"
        excol = "Cp experimental/reference (J mol^-1 K^-1)"
        ylabel = "Heat capacity (J mol⁻¹ K⁻¹)"
    elif quantity == "H":
        pbe = enthalpy_increment_curve(pbe_path)
        b3 = enthalpy_increment_curve(b3lyp_path)
        xcol = "T (K)"
        ycol = "H-Href (kJ mol^-1)"
        excol = "H-H298.15 experimental/reference (kJ mol^-1)"
        ylabel = "H(T) − H(298.15 K) (kJ mol⁻¹)"
    else:
        raise KeyError(quantity)

    ax.plot(pbe[xcol], pbe[ycol], label=f"PBE {quantity}")
    ax.plot(b3[xcol], b3[ycol], label=f"B3LYP {quantity}")
    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental[excol],
        "experiment/reference",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_model_volume(
    pbe_path: str | Path,
    b3lyp_path: str | Path,
    experimental: pd.DataFrame,
    *,
    pressure: float = 0.0,
    normalized: bool = False,
    reference_temperature: float = 300.0,
):
    """Compare absolute or normalized PBE/B3LYP volumes with experiment."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for label, path in [("PBE", pbe_path), ("B3LYP", b3lyp_path)]:
        table = qha_section(path, "volume", pressure=pressure)
        volume = table["volume"].to_numpy(dtype=float) * 4.0
        if normalized:
            ref = float(
                np.interp(
                    reference_temperature,
                    table["T (K)"].to_numpy(dtype=float),
                    volume,
                )
            )
            volume = volume / ref
        ax.plot(table["T (K)"], volume, label=label)

    ex = experimental.copy()
    ex_volume = ex["conventional cell volume (A^3)"].to_numpy(dtype=float)
    if normalized:
        ex_ref = float(
            np.interp(
                reference_temperature,
                ex["T (K)"].to_numpy(dtype=float),
                ex_volume,
            )
        )
        ex_volume = ex_volume / ex_ref

    _black_experiment(
        ax,
        ex["T (K)"],
        ex_volume,
        "experiment",
    )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("V / V(300 K)" if normalized else "Conventional cell volume (Å³)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_model_alpha(
    pbe_path: str | Path,
    b3lyp_path: str | Path,
    experimental: pd.DataFrame,
    *,
    pressure: float = 0.0,
):
    """Compare PBE/B3LYP volumetric thermal expansion with experiment."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for label, path in [("PBE", pbe_path), ("B3LYP", b3lyp_path)]:
        table = qha_section(path, "alpha", pressure=pressure)
        ax.plot(table["T (K)"], table["alpha"], label=label)

    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental["volumetric alpha derived (K^-1)"],
        "experiment",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Volumetric thermal expansion (K⁻¹)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_model_bulk_modulus(
    pbe_path: str | Path,
    b3lyp_path: str | Path,
    experimental: pd.DataFrame,
    component: str,
    *,
    pressure: float = 0.0,
):
    """Compare either KT or KS for PBE/B3LYP with the matched experiment."""
    import matplotlib.pyplot as plt

    component = component.upper()
    if component == "KT":
        key = "KT"
        excol = "KT experimental (GPa)"
        ylabel = "Isothermal bulk modulus, $K_T$ (GPa)"
        exlabel = "experimental $K_T$"
    elif component == "KS":
        key = "KS"
        excol = "KS from adiabatic Cij (GPa)"
        ylabel = "Adiabatic bulk modulus, $K_S$ (GPa)"
        exlabel = "experimental $K_S$"
    else:
        raise ValueError("component must be 'KT' or 'KS'")

    fig, ax = plt.subplots()

    for label, path in [("PBE", pbe_path), ("B3LYP", b3lyp_path)]:
        table = qha_section(path, key, pressure=pressure)
        ax.plot(table["T (K)"], table[key], label=f"{label} ${component[0]}_{component[1]}$")

    _black_experiment(
        ax,
        experimental["T (K)"],
        experimental[excol],
        exlabel,
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax

