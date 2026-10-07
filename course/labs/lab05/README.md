# Laboratory 5 — Quasi-harmonic thermodynamics of MgO

This laboratory uses precomputed CRYSTAL23 phonon calculations. Students do not
run the underlying electronic-structure jobs.

The working sequence is:

```text
01-understand-qha-dataset.ipynb
02-qha-minimization.ipynb
03-qha-properties-experiment.ipynb
04-pbe-vs-b3lyp.ipynb
```

Reference calculations span:

```text
V/Vref = 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.09
```

The first notebook asks you to inspect the supplied CRYSTAL input/output files
and determine how the phonon calculations were performed before any automated
summary is shown.

The notebooks generate lightweight Quantas YAML/HDF5 results locally in the
Codespace. The expensive CRYSTAL calculations have already been completed.


The fourth notebook is a model-comparison extension. It uses the supplied
B3LYP QHA reference dataset and compares its predictions with the main PBE
calculation and the same experimental reference data.
