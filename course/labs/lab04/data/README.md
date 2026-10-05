# Experimental MgO thermodynamic reference data

`mgo_experimental_heat_capacity.csv` contains a compact MgO reference dataset
for the Laboratory 4 comparison.

The original NIST tables report specific heat in J kg^-1 K^-1. Values in the
course CSV are converted to molar units using:

```text
M(MgO) = 40.3044 g mol^-1
```

Low-temperature data:
- T.H.K. Barron, W.T. Berg, J.A. Morrison (1959)
- NIST Structural Ceramics Database SRD 30, citation Z00723
- https://srdata.nist.gov/CeramicDataPortal/Scd/Z00723

Higher-temperature data:
- evaluated MgO drop-calorimetry dataset
- NIST Structural Ceramics Database SRD 30, citation Z00290
- https://srdata.nist.gov/CeramicDataPortal/Scd/Z00290

The experimental values are plotted as approximately constant-pressure heat
capacity. The harmonic calculation provides Cv. This distinction is an
intentional part of the laboratory interpretation.

## Experimental/reference entropy

`mgo_experimental_entropy.csv` contains evaluated standard molar entropy
values for solid MgO from the NIST Chemistry WebBook (SRD 69), based on the
Chase (1998) NIST-JANAF Shomate correlation for solid MgO.

The correlation is valid from 298 K upward and gives:

```text
S°(298.15 K) ≈ 26.85 J mol^-1 K^-1
```

The independent CODATA review value reported by NIST is:

```text
S°(298.15 K, 1 bar) = 26.95 ± 0.15 J mol^-1 K^-1
```

The CSV contains reference values from 298.15 to 1200 K in units directly
comparable with the harmonic entropy produced by Quantas:

```text
J mol^-1 K^-1
```

These data are standard-pressure thermodynamic reference values. The calculated
harmonic entropy is evaluated for the fixed DFT equilibrium volume, so a small
systematic difference is physically expected even after q-point convergence.

Source:
https://webbook.nist.gov/cgi/cbook.cgi?ID=C1309484&Plot=on&Type=JANAFS

