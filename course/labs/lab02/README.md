# Laboratory 2 — From structure to energy

This laboratory introduces a CRYSTAL single-point energy calculation and a
controlled reciprocal-space convergence test for MgO.

Contents:

```text
01-first-scf.ipynb
02-kpoint-convergence.ipynb
inputs/mgo_shrink2.d12
outputs/
reference-results/
```

After:

```bash
update-course
start-lab 02
```

work only inside:

```text
work/lab02/
```

The CRYSTAL calculations themselves run on CluMiner. Codespaces is used to
prepare/inspect files and analyse the returned outputs.

For every calculation in this laboratory use:

```bash
cluminer-crystal INPUT.d12 --cores-per-node 2 --threads 2
```
