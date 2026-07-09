# GPTFF Model Trained on MatPES-PBE-2025.2

MatPES is a foundational potential energy surface (PES) dataset for materials
machine-learning interatomic potentials. It provides density-functional-theory
labels for energies, forces, and stresses, with structures sampled to cover
equilibrium, near-equilibrium, and molecular-dynamics-like configurations. See
[matpes.ai](https://matpes.ai) for the dataset project and release details.

The GPTFF model in this directory was trained on the MatPES-PBE-2025.2 split
with train/validation/test proportions of 90%/5%/5%, using the settings in
`config.yaml`. The reported checkpoint is the best-force checkpoint. The metrics
below are reported in train/validation/test order.

| Checkpoint | Parameters | Energy (meV/atom) | Force (meV/Ang) | Stress (GPa) |
|---|---:|---:|---:|---:|
| `GPTFF-MatPES_PBE_2025.2.pt` | 0.314M | 42.4/43.5/42.6 | 93.5/102.1/104.1 | 0.551/0.579/0.584 |
