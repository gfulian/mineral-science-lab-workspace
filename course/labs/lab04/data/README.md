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
