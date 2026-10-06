# Laboratory 5 — Quasi-harmonic thermodynamics of MgO

This laboratory uses precomputed CRYSTAL23 phonon calculations. Students do not
run the underlying electronic-structure jobs.

The working sequence is:

```text
01-understand-qha-dataset.ipynb
02-qha-minimization.ipynb
03-qha-properties-experiment.ipynb
```

Reference calculations span:

```text
V/Vref = 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.09
```

Every phonon calculation uses a 4x4x4 SCELPHONO supercell.

The notebooks generate lightweight Quantas YAML/HDF5 results locally in the
Codespace. The expensive CRYSTAL calculations have already been completed.
