# Laboratory 5 reference material

## `raw/`

Current PBE/POB-TZVP-REV2 CRYSTAL23 reference calculations used to build the
main Laboratory 5 QHA input.

## `mgo_b3lyp.yaml`

Instructor-provided precomputed MgO QHA dataset obtained with a B3LYP
electronic-structure model.

This dataset is used only in Notebook 04 to compare the predictions obtained
from two electronic-structure models. The expensive underlying calculations
are not repeated by students.

The B3LYP dataset was generated independently from the current PBE dataset.
Consequently, the comparison is pedagogically useful but should not be
interpreted as a mathematically perfect one-variable benchmark: the numerical
sampling and the set of calculated volumes are not identical. The notebook
asks you to identify this limitation explicitly.
