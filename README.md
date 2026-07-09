GPTFF (Graph-based Pretrained Transformer Force Field) is an AI force field for
inorganic materials.

## Installation

Clone this repository and install:

```bash
pip install .
```

## Quick Start

The package ships with the default `GPTFF-MatPES_PBE_2025.2.pt` checkpoint,
which is used automatically by the calculation tools unless a
custom model is provided.

### Relax a structure

Python API:

```python
from pymatgen.core import Structure

from gptff.tasks.relaxation import ASERelaxationRunner

structure = Structure.from_file("POSCAR")
runner = ASERelaxationRunner()
result = runner.run(structure)

relaxed_structure = result.final_structure
relaxed_structure.to(filename="relaxed.cif")
```

Command line:

```bash
gptff relaxation POSCAR --output relaxed.cif
```

Common options:

```bash
gptff relaxation POSCAR \
  --output relaxed.vasp \
  --device cpu \
  --model-path /path/to/checkpoint.pt \
  --no-relax-cell
```

The relaxation command currently uses the ASE backend. Output formats are
handled by pymatgen and are inferred from the output file suffix.

## Training

Create an `AtomicDataset` from labeled pymatgen structures:

```python
import numpy as np
from pymatgen.core import Structure

from gptff.data import AtomicDataset, AtomicSample

structure = Structure.from_file("POSCAR")
dataset = AtomicDataset(
    samples=(
        AtomicSample(
            structure=structure,
            energy=-12.34,  # eV
            forces=np.zeros((structure.num_sites, 3)),  # eV/angstrom
            stress=np.zeros((3, 3)),  # raw VASP stress in kBar
            sample_id="material-1-frame-0",
            material_id="material-1",
        ),
    ),
    name="my-dataset",
)
dataset.to_file("dataset.json.gz")
```

Point the training config to the serialized dataset:

```yaml
data:
  dataset_path: dataset.json.gz
  dataset_format: atomic_json
```

Then launch training:

```bash
gptff train config.yaml
```

For multi-GPU training, launch with PyTorch DDP:

```bash
torchrun --nproc_per_node=4 -m gptff.cli.main train config.yaml
```

### Label conventions

- `energy`: total structure energy in eV.
- `forces`: array with shape `(num_atoms, 3)` in eV/angstrom.
- `stress`: raw VASP stress matrix or Voigt vector in kBar.

Energy and force labels are always required. Stress labels are required only
when `stress_loss_weight > 0`; set `stress_loss_weight: 0` for energy-force
datasets without stress.

### Config overview

`config.yaml` is organized into these sections:

- `model`: graph cutoffs, basis sizes, interaction blocks, readout settings,
  and attention settings.
- `element_references`: elemental reference energy source. Use `fit`, `null`,
  a mapping with atomic-number keys, or a YAML/JSON file path.
- `optimizer`: optimizer name, learning rate, weight decay, scheduler, and
  scheduler parameters.
- `training`: epochs, batch size, device, workers, AMP, output directory, seed,
  and distributed mode.
- `loss`: energy, force, and stress loss weights.
- `data`: dataset path, dataset format, split fractions, split seed,
  material-grouped splitting, and graph caching.

For large datasets, GPTFF also supports precomputed sharded HDF5 graph datasets:

```yaml
data:
  dataset_path: /path/to/dataset.gptff
  dataset_format: sharded_hdf5_graph
```

Checkpoints are written to the configured output directory. `last.pt` stores
the latest checkpoint, while `bestE.pt` and `bestF.pt` store the best validation
energy-MAE and force-MAE checkpoints.

## Reference

If you found GPTFF useful, please cite our article:

```
@article{XIE2024,
title = {GPTFF: A high-accuracy out-of-the-box universal AI force field for arbitrary inorganic materials},
journal = {Science Bulletin},
year = {2024},
issn = {2095-9273},
doi = {https://doi.org/10.1016/j.scib.2024.08.039},
url = {https://www.sciencedirect.com/science/article/pii/S2095927324006327},
author = {Fankai Xie and Tenglong Lu and Sheng Meng and Miao Liu},
keywords = {Data Science, Molecular Dynamics, Graph Neural Network, Universal Fore Field},
}
```
