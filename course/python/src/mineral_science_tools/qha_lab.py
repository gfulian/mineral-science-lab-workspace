"""Teaching helpers for Mineral Science Laboratory 5.

The module intentionally uses only public Quantas APIs for native results.
Experimental points are rendered in black by design so they remain visually
separate from computational series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence
import re
import numpy as np
import pandas as pd


def _read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def inspect_reference_input(path: str | Path) -> dict:
    """Extract the few CRYSTAL settings students are expected to identify."""
    path=Path(path); text=_read(path); lines=text.splitlines()
    a=float(lines[4].strip())
    basis=re.search(r"BASISSET\s*\n\s*(\S+)",text,re.I)
    functional=re.search(r"\nDFT\s*\n\s*(\S+)",text,re.I)
    shrink=re.search(r"SHRINK\s*\n\s*(\d+)\s+(\d+)",text,re.I)
    sc=re.search(r"SCELPHONO\s*\n\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\n\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\n\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)",text,re.I)
    matrix=np.asarray([int(x) for x in sc.groups()],dtype=int).reshape(3,3) if sc else None
    m=re.search(r"_v(\d{3})_",path.name)
    vf=float(int(m.group(1))/100) if m else np.nan
    return {
        "file": path.name,
        "V/Vref label": vf,
        "a conventional (A)": a,
        "V conventional (A^3)": a**3,
        "SCELPHONO": matrix.tolist() if matrix is not None else None,
        "primitive repetitions": int(round(abs(np.linalg.det(matrix)))) if matrix is not None else np.nan,
        "SHRINK": f"{shrink.group(1)} {shrink.group(2)}" if shrink else "",
        "functional": functional.group(1) if functional else "",
        "basis": basis.group(1) if basis else "",
    }


def inspect_reference_output(path: str | Path) -> dict:
    path=Path(path); text=_read(path)
    def integer(pattern):
        m=re.search(pattern,text,re.I); return int(m.group(1)) if m else np.nan
    return {
        "output": path.name,
        "atoms in phonon supercell": integer(r"NUMBER OF ATOMS PER SUPERCELL\s+(\d+)"),
        "q-points": integer(r"PERMITS THE CALCULATION OF MODES AT\s+(\d+)\s+K POINTS"),
        "atomic orbitals": integer(r"NUMBER OF AO\s+(\d+)"),
        "shells": integer(r"NUMBER OF SHELLS\s+(\d+)"),
        "normal termination": "EEEEEEEEEE TERMINATION" in text,
    }


def reference_series_table(folder: str | Path = "reference-results/raw") -> pd.DataFrame:
    folder=Path(folder)
    rows=[]
    for inp in sorted(folder.glob("mgo_v*_4x4x4.d12")):
        row=inspect_reference_input(inp)
        out=inp.with_suffix(".out")
        if out.exists(): row.update(inspect_reference_output(out))
        rows.append(row)
    df=pd.DataFrame(rows)
    if not df.empty:
        ref=df.loc[(df["V/Vref label"]-1).abs().idxmin(),"V conventional (A^3)"]
        df["V/Vref from cell"] = df["V conventional (A^3)"]/ref
    return df


def load_qha(path: str | Path):
    from quantas.api import qha
    envelope=qha.read_result(Path(path))
    return envelope, qha.get_result(envelope)


def _field(payload, *names):
    for name in names:
        if hasattr(payload,name): return np.asarray(getattr(payload,name),dtype=np.float64)
    available=[n for n in dir(payload) if not n.startswith("_")]
    raise AttributeError(f"None of {names!r} found in QHA payload. Available fields include: {available}")


def nearest_index(values, target: float) -> int:
    values=np.asarray(values,dtype=float)
    return int(np.nanargmin(np.abs(values-float(target))))


def qha_grid_field(path: str | Path, quantity: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    _,p=load_qha(path)
    T=_field(p,"temperature"); P=_field(p,"pressure")
    aliases={
        "volume": ("equilibrium_volume",),
        "alpha": ("thermal_expansion",),
        "KT": ("isothermal_bulk_modulus",),
        "KS": ("adiabatic_bulk_modulus",),
        "Kprime": ("bulk_modulus_pressure_derivative","isothermal_bulk_modulus_pressure_derivative"),
        "Cv": ("isochoric_heat_capacity",),
        "Cp": ("isobaric_heat_capacity",),
        "entropy": ("entropy",),
        "enthalpy": ("enthalpy",),
    }
    if quantity not in aliases: raise KeyError(quantity)
    return T,P,_field(p,*aliases[quantity])


def qha_section(path: str | Path, quantity: str, *, pressure: float=0.0) -> pd.DataFrame:
    T,P,Z=qha_grid_field(path,quantity)
    j=nearest_index(P,pressure)
    return pd.DataFrame({"T (K)":T, quantity:Z[:,j],"P (GPa)":P[j]})


def qha_isotherm(path: str | Path, quantity: str, *, temperature: float=300.0) -> pd.DataFrame:
    T,P,Z=qha_grid_field(path,quantity)
    i=nearest_index(T,temperature)
    return pd.DataFrame({"P (GPa)":P, quantity:Z[i,:],"T (K)":T[i]})


def qha_inventory(path: str | Path) -> pd.DataFrame:
    """List plot properties advertised by the installed Quantas version."""
    from quantas.api import qha
    result=qha.read_result(Path(path)); inv=qha.describe_plots(result)
    return pd.DataFrame([{"key":x.key,"name":x.name,"symbol":x.symbol_plain,"unit":x.unit,"category":x.category} for x in inv.properties])


def _find_plot_key(inv, words: Sequence[str]):
    words=[w.lower() for w in words]
    for d in inv.properties:
        hay=" ".join([str(d.key),str(d.name),str(d.symbol_plain),str(d.symbol_math)]).lower()
        if all(w in hay for w in words): return d.key
    raise KeyError(f"Could not find a QHA plot property containing {words}. Available: {[d.key for d in inv.properties]}")


def _single_pressure_plot_series(path: str | Path, words: Sequence[str], *, energy_unit: str="J/mol", component: str|None=None):
    """Extract a display-ready line from a result containing one pressure.

    The public Quantas plotting contract performs the package's own unit
    conversion. This avoids inferring thermodynamic units from raw arrays.
    """
    from quantas.api import qha
    result=qha.read_result(Path(path)); inv=qha.describe_plots(result)
    key=_find_plot_key(inv,words)
    plots=qha.build_plots(result,properties=[key],options=qha.PlotOptions(energy_unit=energy_unit))
    series=[]
    for plot in plots.plots:
        for s in getattr(plot,"series",[]) or []:
            if component is None or component.lower() in str(s.label).lower() or component.lower() in str(s.key).lower():
                series.append((plot,s))
    if len(series)!=1:
        labels=[getattr(s,'label','') for _,s in series]
        raise ValueError(f"Expected one matching series for {words}/{component}; found {len(series)}: {labels}")
    plot,s=series[0]
    return pd.DataFrame({"T (K)":np.asarray(s.x,dtype=float),"value":np.asarray(s.y,dtype=float)}), plot.y_axis.label


def entropy_curve(path: str | Path) -> pd.DataFrame:
    table,label=_single_pressure_plot_series(path,["entropy"],energy_unit="J/mol")
    return table.rename(columns={"value":"S (J mol^-1 K^-1)"})


def heat_capacity_curve(path: str | Path, component: str="Cp") -> pd.DataFrame:
    # Current Quantas advertises the combined heat-capacity property.
    try:
        table,label=_single_pressure_plot_series(path,["heat","capac"],energy_unit="J/mol",component=component)
    except Exception:
        table,label=_single_pressure_plot_series(path,[component],energy_unit="J/mol")
    return table.rename(columns={"value":f"{component} (J mol^-1 K^-1)"})


def enthalpy_increment_curve(path: str | Path, *, reference_temperature: float=298.15) -> pd.DataFrame:
    table,label=_single_pressure_plot_series(path,["enthalpy"],energy_unit="kJ/mol")
    i=nearest_index(table["T (K)"],reference_temperature)
    table=table.rename(columns={"value":"H (kJ mol^-1)"})
    table["H-Href (kJ mol^-1)"]=table["H (kJ mol^-1)"]-float(table.iloc[i]["H (kJ mol^-1)"])
    table.attrs["reference_temperature"] = float(table.iloc[i]["T (K)"])
    return table


def _black_experiment(ax,x,y,label,**kwargs):
    defaults=dict(marker="o",s=24,color="black",edgecolors="black",zorder=5)
    defaults.update(kwargs); ax.scatter(x,y,label=label,**defaults)


def plot_absolute_volume(poly_path, eos_path, experimental, *, pressure=0.0, volume_multiplier=4.0):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for label,path in [("QHA polynomial",poly_path),("QHA BM3",eos_path)]:
        t=qha_section(path,"volume",pressure=pressure)
        ax.plot(t["T (K)"],t["volume"]*volume_multiplier,label=label)
    _black_experiment(ax,experimental["T (K)"],experimental["conventional cell volume (A^3)"],"experiment")
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Conventional cell volume (Å³)"); ax.legend(); fig.tight_layout(); return fig,ax


def plot_normalized_volume(poly_path, eos_path, experimental, *, pressure=0.0, volume_multiplier=4.0, reference_temperature=300.0):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for label,path in [("QHA polynomial",poly_path),("QHA BM3",eos_path)]:
        t=qha_section(path,"volume",pressure=pressure); vals=t["volume"].to_numpy()*volume_multiplier
        i=nearest_index(t["T (K)"],reference_temperature); ax.plot(t["T (K)"],vals/vals[i],label=label)
    ex=experimental.copy(); i=nearest_index(ex["T (K)"],reference_temperature); ref=float(ex.iloc[i]["conventional cell volume (A^3)"])
    _black_experiment(ax,ex["T (K)"],ex["conventional cell volume (A^3)"]/ref,"experiment")
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("V / V(300 K)"); ax.legend(); fig.tight_layout(); return fig,ax


def plot_alpha(poly_path, experimental, *, pressure=0.0):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); t=qha_section(poly_path,"alpha",pressure=pressure)
    ax.plot(t["T (K)"],t["alpha"],label="QHA polynomial")
    _black_experiment(ax,experimental["T (K)"],experimental["volumetric alpha derived (K^-1)"],"experiment: 3 × linear α")
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Volumetric thermal expansion (K⁻¹)"); ax.legend(); fig.tight_layout(); return fig,ax


def plot_bulk_moduli(poly_path, experimental_ks, *, pressure=0.0):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); kt=qha_section(poly_path,"KT",pressure=pressure); ks=qha_section(poly_path,"KS",pressure=pressure)
    ax.plot(kt["T (K)"],kt["KT"],label="QHA K_T")
    ax.plot(ks["T (K)"],ks["KS"],label="QHA K_S")
    _black_experiment(ax,experimental_ks["T (K)"],experimental_ks["KS experimental (GPa)"],"experimental K_S")
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Bulk modulus (GPa)"); ax.legend(); fig.tight_layout(); return fig,ax


def plot_room_temperature_compression(grid_path, experimental_eos, *, temperature=300.0, volume_multiplier=4.0):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); t=qha_isotherm(grid_path,"volume",temperature=temperature)
    ax.plot(t["P (GPa)"],t["volume"]*volume_multiplier,label="QHA")
    _black_experiment(ax,experimental_eos["P (GPa)"],experimental_eos["conventional cell volume from experimental BM3 fit (A^3)"],"Speziale et al. experimental BM3 fit")
    ax.set_xlabel("Pressure (GPa)"); ax.set_ylabel("Conventional cell volume (Å³)"); ax.legend(); fig.tight_layout(); return fig,ax


def plot_thermochemistry(poly_path, experimental, quantity: str):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    if quantity=="S":
        th=entropy_curve(poly_path); x=th["T (K)"]; y=th["S (J mol^-1 K^-1)"]; excol="S experimental/reference (J mol^-1 K^-1)"; ylabel="Entropy (J mol⁻¹ K⁻¹)"; label="QHA S"
    elif quantity=="Cp":
        th=heat_capacity_curve(poly_path,"Cp"); x=th["T (K)"]; y=th["Cp (J mol^-1 K^-1)"]; excol="Cp experimental/reference (J mol^-1 K^-1)"; ylabel="Heat capacity (J mol⁻¹ K⁻¹)"; label="QHA C_P"
    elif quantity=="H":
        th=enthalpy_increment_curve(poly_path); x=th["T (K)"]; y=th["H-Href (kJ mol^-1)"]; excol="H-H298.15 experimental/reference (kJ mol^-1)"; ylabel="H(T) − H(298 K) (kJ mol⁻¹)"; label="QHA ΔH"
    else: raise KeyError(quantity)
    ax.plot(x,y,label=label)
    _black_experiment(ax,experimental["T (K)"],experimental[excol],"NIST/JANAF")
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout(); return fig,ax


def method_comparison(poly_path, eos_path, *, temperature=300.0, pressure=0.0) -> pd.DataFrame:
    rows=[]
    for method,path in [("polynomial",poly_path),("BM3 EOS",eos_path)]:
        row={"method":method}
        for key,out in [("volume","V (A^3 primitive)"),("KT","KT (GPa)"),("KS","KS (GPa)"),("alpha","alphaV (K^-1)")]:
            T,P,Z=qha_grid_field(path,key); i=nearest_index(T,temperature); j=nearest_index(P,pressure); row[out]=float(Z[i,j])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pt_map(path: str | Path, quantity: str, *, volume_multiplier: float=4.0):
    import matplotlib.pyplot as plt
    T,P,Z=qha_grid_field(path,quantity)
    if quantity=="volume": Z=Z*volume_multiplier; label="Conventional cell volume (Å³)"
    elif quantity=="alpha": label="Volumetric thermal expansion (K⁻¹)"
    elif quantity=="KT": label="Isothermal bulk modulus (GPa)"
    else: label=quantity
    fig,ax=plt.subplots(); contour=ax.contourf(T,P,Z.T,levels=14); fig.colorbar(contour,ax=ax,label=label)
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Pressure (GPa)"); fig.tight_layout(); return fig,ax
