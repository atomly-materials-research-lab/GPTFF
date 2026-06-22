# GPTFF Models Trained on MatPES-PBE-2025.2

MatPES is a foundational potential energy surface (PES) dataset for materials
machine-learning interatomic potentials. It provides density-functional-theory
labels for energies, forces, and stresses, with structures sampled to cover
equilibrium, near-equilibrium, and molecular-dynamics-like configurations. See
[matpes.ai](https://matpes.ai) for the dataset project and release details.

The GPTFF models in this directory were trained on the MatPES-PBE-2025.2 split
with train/validation/test proportions of 90%/5%/5%. The MAEs below are reported
in train/validation/test order.

| Model | Parameters | Energy (meV/atom) | Force (meV/Ang) | Stress (GPa) |
|---|---:|---:|---:|---:|
| GPTFF base | 0.308M | 37.9/39.1/37.8 | 110.2/113.9/116.3 | 0.499/0.549/0.549 |
| GPTFF atom attention | 0.513M | 33.6/35.9/34.4 | 100.7/106.6/108.5 | 0.442/0.510/0.510 |
