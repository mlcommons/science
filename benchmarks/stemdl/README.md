# STEMDL

STEMDL (Classification)

State of the art scanning transmission electron microscopes (STEM)
produce focused electron beams with atomic dimensions and allow to
capture diffraction patterns arising from the interaction of incident
electrons with nanoscale material volumes. Backing out the local
atomic structure of said materials requires compute- and
time-intensive analyses of these diffraction patterns (known as
convergent beam electron diffraction, CBED). Traditional analyses of
CBED requires iterative numerical solutions of partial differential
equations and comparison with experimental data to refine the starting
material configuration. This process is repeated anew for every newly
acquired experimental CBED pattern and/or probed material.

In this benchmark, we used newly developed multi-GPU and multi-node
electron scattering simulation codes
[[1]](https://www.osti.gov/biblio/1631694-namsa) on the Summit
supercomputer to generate CBED patterns from over 60,000 materials
(solid-state materials), representing nearly every known crystal
structure. A scaled-down version of this data
[[2]](https://doi.ccs.ornl.gov/ui/doi/70) is used for one of the data
challenges [[3]](https://smc-datachallenge.ornl.gov/challenge-2-2020/)
at SMC 2020 conference, and the overarching goals are to: (1) explore
the suitability of machine learning algorithms in the advanced
analysis of CBED and (2) produce a machine learning algorithm capable
of overcoming intrinsic difficulties posed by scientific datasets.

A data sample from this data set is given by a 3-d array formed by
stacking various CBED patterns simulated from the same material at
different distinct material projections (i.e. crystallographic
orientations). Each CBED pattern is a 2-d array with float 32-bit
image intensities. Associated with each data sample in the data set is
a host of material attributes or properties which are, in principle,
retrievable via analysis of this CBED stack. Of note are (1) 200
crystal space groups out of 230 unique mathematical discrete space
groups and (2) local electron density which governs material’s
property.

This benchmark consists of 2 tasks: classification for crystal space
groups and reconstruction for local electron density, the example
implementation of which are provided in
[[4]](https://link.springer.com/chapter/10.1007%2F978-3-030-63393-6_30)
and [[5]](https://arxiv.org/abs/1909.11150).  STEMDL Specific
Benchmark Targets

* Scientific objective(s):
  * Objective: Classification for crystal space groups
  * Formula: F1 score on validation data
  * Score: 0.9 considered converged
* Data
  * Download: https://doi.ccs.ornl.gov/ui/doi/70
  * Data Size: 548.7 GiB
  * Training samples: 138.7K
  * Validation samples: 48.4

## Versions

This folder contains the Pytorch-Lightning implementation of STEMDL
classification. MLCOmmons contains two versions

* A version developed by STFC. More details can be found
  [here][stfc/README.md]
* The original version developed by ORNL. More details can be found
  [here][ornl/README.md]

The original implementation is available
[here](https://github.com/at-aaims/stemdl-benchmark) with the original
[instructions](https://github.com/at-aaims/stemdl-benchmark#quickstart). The
Time-to-solution: 40min on 60 V100 GPUs
