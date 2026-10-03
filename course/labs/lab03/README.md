# Laboratory 3 — Geometry optimization and equation of state

The operational material is divided into:

```text
01-geometry-optimization.ipynb
02-equation-of-state.ipynb
inputs/
outputs/
reference-results/
```

Use:

```bash
update-course
start-lab 03
```

and work only inside:

```text
work/lab03/
```

CRYSTAL calculations run on CluMiner using:

```bash
cluminer-crystal INPUT.d12 --cores-per-node 2 --threads 2
```

The EOS analysis runs in Codespaces with Quantas.

## Important: SHRINK is intentionally not preselected

Both Laboratory 3 input templates contain:

```text
SHRINK
<X> <X>
```

Replace `<X>` with the SHRINK value that you selected from your Laboratory 2
k-point convergence analysis. Use exactly the same value throughout Laboratory
3 unless instructed otherwise.

## Geometry-optimization tolerances

The distributed Laboratory 3 inputs use:

```text
TOLDEG  0.00001
TOLDEX  0.00004
TOLDEE  8
```

These values are already set in the templates and should not be modified by
students.
