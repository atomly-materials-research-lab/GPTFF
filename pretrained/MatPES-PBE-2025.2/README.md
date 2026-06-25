# GPTFF Models Trained on MatPES-PBE-2025.2

MatPES is a foundational potential energy surface (PES) dataset for materials
machine-learning interatomic potentials. It provides density-functional-theory
labels for energies, forces, and stresses, with structures sampled to cover
equilibrium, near-equilibrium, and molecular-dynamics-like configurations. See
[matpes.ai](https://matpes.ai) for the dataset project and release details.

The GPTFF models in this directory were trained on the MatPES-PBE-2025.2 split
with train/validation/test proportions of 90%/5%/5%. The MAEs below are reported
in train/validation/test order.

| Checkpoint | Parameters | Energy (meV/atom) | Force (meV/Ang) | Stress (GPa) |
|---|---:|---:|---:|---:|
| `GPTFF-MatPES_PBE_2025.2.pt` | 0.488M | 34.4/36.1/35.2 | 103.2/109.0/111.1 | 0.451/0.520/0.524 |
| `GPTFF_Base-MatPES_PBE_2025.2.pt` | 0.308M | 37.9/39.1/37.8 | 110.2/113.9/116.3 | 0.499/0.549/0.549 |
