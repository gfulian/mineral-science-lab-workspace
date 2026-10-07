# MgO experimental and reference datasets

These compact CSV files are distributed for the scientific comparisons in
Laboratory 5. Experimental/reference points are plotted in black in the course
notebooks.

## `mgo_nist_janaf_thermochemistry.csv`

Evaluated solid-MgO thermochemistry from the NIST Chemistry WebBook (SRD 69),
using the Chase (1998) Shomate correlation valid from 298 to 3105 K. The course
file tabulates $C_P^\circ$, $S^\circ$ and
$H^\circ(T)-H^\circ(298.15\,\mathrm K)$ from 298.15 to 1500 K.

Source:
https://webbook.nist.gov/cgi/cbook.cgi?ID=C1309484&Mask=B

NIST also reports the independent CODATA standard entropy
$S^\circ(298.15\,\mathrm K,1\,bar)=26.95\pm0.15$ J mol^-1 K^-1.

## `mgo_reeber1995_volume_expansion.csv`

Evaluated MgO molar volume and linear thermal-expansion coefficient from Reeber,
Goessel & Wang (1995), distributed through NIST Structural Ceramics Database
SRD 30, citation Z00631. The database reports the linear coefficient. For cubic
MgO the course file also contains the derived volumetric coefficient
$\alpha_V=3\alpha_a$ and converts molar volume to conventional-cell volume
($Z=4$).

Source:
https://srdata.nist.gov/CeramicDataPortal/Scd/Z00631

## `mgo_isaak1989_bulk_moduli.csv`

High-temperature MgO elasticity from Isaak, Anderson & Goto (1989), distributed
through NIST Structural Ceramics Database SRD 30, citation Z00281.

The source requires a careful thermodynamic distinction:

- the rectangular-parallelepiped resonance experiment determines **adiabatic**
  single-crystal elastic constants $C_{ij}^{S}$;
- the NIST "Bulk Modulus" series (161.6 GPa at 300 K, 141.4 GPa at 1000 K,
  etc.) is the **isothermal bulk modulus $K_T$** used in MgO thermoelastic
  comparisons;
- for cubic MgO, the corresponding adiabatic bulk modulus can be reconstructed
  directly from the measured adiabatic stiffnesses,

  $$
  K_S = \frac{C_{11}^{S}+2C_{12}^{S}}{3}.
  $$

The CSV therefore contains both $K_T$ and the $K_S$ reconstructed from the
measured adiabatic $C_{ij}$. This allows Laboratory 5 to compare like with like.

Source:
https://srdata.nist.gov/CeramicDataPortal/Scd/Z00281


## `mgo_speziale2001_300K_BM3.csv`

Black reference points generated from the published third-order Birch–Murnaghan
fit to the room-temperature quasi-hydrostatic MgO compression data of Speziale
et al. (2001):

```text
V0 = 74.71 A^3
KT0 = 160.2 GPa (fixed)
K'T = 3.99
```

The CSV is therefore **a sampled experimental EOS fit**, not a transcription of
individual raw diffraction measurements.

DOI: 10.1029/2000JB900318

## `mgo_tange2009_ambient_anchors.csv`

Ambient-condition primary parameters used by Tange et al. (2009) in a unified
MgO P–V–T EOS analysis:

```text
V0    = 74.698 A^3
KS0   = 162.83 GPa
alpha = 3.17e-5 K^-1
Cp0   = 37.4 J mol^-1 K^-1
```

DOI: 10.1029/2008JB005813

## A note on comparison

The datasets do not all represent the same experimental constraint. In
particular, the Isaak modulus is adiabatic $K_S$, while a static compression EOS
is naturally connected to $K_T$. Laboratory 5 explicitly asks you to compare
like with like and to discuss offsets separately from trends.
