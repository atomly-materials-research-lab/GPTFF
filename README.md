GPTFF (Graph-based Pretrained Transformer Force Field) can simulate arbitrary inorganic systems with good precision and generalizability.

## Installation

Using `conda` to create a new python virtual env(not necessary):

```bash
conda create -n gptff python=3.11
```

Then clone the `GPTFF` repo and install:

```bash
git clone https://github.com/atomly-materials-research-lab/GPTFF.git
cd GPTFF
pip install .
```

## Usage

`ASECalculator()` loads the default `GPTFF-MatPES_PBE_2025.2.pt` checkpoint and
automatically uses CUDA when available. Pass `device="cpu"` or
`model_path="/path/to/checkpoint.pt"` to override those defaults.

**Fast Energy(eV), Force(eV/Å), Stress(eV/Å^3, ASE Voigt) calculation:**

```python
from gptff.interfaces import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

p = ASECalculator()

adp = AseAtomsAdaptor()
struc = Structure.from_file('POSCAR_structure')
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

energy = atoms.get_potential_energy() # unit (eV)
forces = atoms.get_forces() # unit (eV/Å)
stress = atoms.get_stress() # unit (eV/Å^3), Voigt order: xx, yy, zz, yz, xz, xy
```

**Structure Optimization:**

Lattice vectors would be changed

```python
from gptff.interfaces import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.filters import ExpCellFilter, StrainFilter

p = ASECalculator()


struc = Structure.from_file('POSCAR_structure') # Read structure

adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

optimizer = ExpCellFilter(atoms) 

FIRE(optimizer).run(fmax=0.01, steps=100)

```

Lattice vectors would not change; only atomic positions would be optimized.

```python
from gptff.interfaces import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.optimize.bfgs import BFGS

p = ASECalculator()


struc = Structure.from_file('POSCAR_structure') # Read structure

adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

optimizer = BFGS(atoms)
optimizer.run(fmax=0.01, steps=1000)
```

**Molecular dynamics (ASE):**
We will support `LAMMPS` with `GPTFF` later.

```python
from gptff.interfaces import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms, units
from ase.md.nvtberendsen import NVTBerendsen
import os

p = ASECalculator()


struc = Structure.from_file('POSCAR_structure') # Read structure

adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

save_dir = './results_path'
os.makedirs(save_dir, exist_ok=True)

temp = 430 # unit (K)
dyn = NVTBerendsen(atoms=atoms, 
                   timestep=2 * units.fs,
                   temperature=temp, # unit (K)
                   taut=200*units.fs, 
                   loginterval=20, # Save md information and trajectory every 20 steps
                   logfile=os.path.join(save_dir, f'output.txt'),  # Information printer
                   trajectory=os.path.join(save_dir, f'Li3PO4_nvt_out_{temp}K.trj'), # Trajectory recorder
                   append_trajectory=True)
dyn.run(100000)

```

## Model training

Build the complete dataset from labeled structures and pass it directly to the
trainer:

```python
import numpy as np
from pymatgen.core import Structure

from gptff.data import AtomicDataset, AtomicSample
from gptff.trainer import Trainer, load_config

structure = Structure.from_file("POSCAR")
samples = [
    AtomicSample(
        structure=structure,
        energy=-12.34,  # total energy in eV
        forces=np.zeros((structure.num_sites, 3)),  # eV/angstrom
        stress=np.zeros((3, 3)),  # raw VASP stress in kBar
        sample_id="material-1-frame-0",
        material_id="material-1",
    ),
]
dataset = AtomicDataset(samples=tuple(samples), name="my-dataset")

config = load_config("config.yaml")
Trainer(config).fit(dataset)
```

Energy and force labels are required. Stress is optional only when
`stress_loss_weight` is zero. GPTFF expects values extracted using the standard
VASP conventions used by pymatgen:

- `energy`: total structure energy in eV.
- `forces`: array with shape `(num_atoms, 3)` in eV/angstrom.
- `stress`: raw VASP `(3, 3)` matrix or six-component Voigt vector in kBar.
  GPTFF applies the VASP sign convention and converts it to GPa internally.

Each sample can define a unique `sample_id`. Set `material_id` when multiple
frames belong to the same material and `group_by_material` is enabled.

`AtomicSample` and `AtomicDataset` implement Monty serialization. A complete
dataset can be stored and restored without a separate structure table:

```python
dataset.to_file("dataset.json.gz")
dataset = AtomicDataset.from_file("dataset.json.gz")
```

For command-line training, set `data.dataset_path` to the serialized dataset:

```bash
gptff train config.yaml
```

For multi-GPU training, launch with PyTorch DDP:

```bash
torchrun --nproc_per_node=4 -m gptff.cli train config.yaml
```

When `training.distributed: auto`, GPTFF automatically enables DDP under
`torchrun` when `WORLD_SIZE > 1`. Checkpoints and logs are written only by rank
0, while all ranks participate in validation and testing.

Large datasets can also be stored as precomputed sharded graph datasets. This
avoids keeping all structures or graphs in memory during training. GPTFF stores
large precomputed graph datasets as multiple HDF5 shards; MPtrj conversion uses
`--num-process` to control the number of output shard files:

```yaml
data:
  dataset_path: /path/to/dataset.gptff
  dataset_format: sharded_hdf5_graph
```

Elemental reference energies are configured through the top-level `element_references` section, not stored as a dataset column. Provide your own mapping/list as `source`, set `source: fit` to fit references from the full input dataset, set `source: null` to disable references, or point `source` to a YAML/JSON reference file.

Reference mappings must use atomic-number keys, not element symbols:

```yaml
element_references:
  source:
    1: -3.12
    3: -1.45
```

The same mapping can be placed in an external file:

```yaml
element_references:
  source: /path/to/reference_energies.yaml
```

During ASE inference, `atoms.get_stress()` follows the ASE convention and returns a six-component stress vector in eV/Å^3, even though the model's internal stress is computed in GPa.

## Training setting

The file `config.yaml` uses separate sections for model, optimizer, training loop, loss, and data settings.

`model`:
- `atom_feature_dim`: Atom feature dimension
- `edge_feature_dim`: Edge feature dimension
- `num_interaction_blocks`: Number of GPTFF interaction blocks
- `num_radial`: Number of radial basis functions
- `num_angular`: Number of normalized Legendre angular basis channels
- `radial_cutoff`: Pair graph cutoff radius
- `angle_cutoff`: Three-body angle cutoff radius
- `cutoff_coeff`: Polynomial cutoff envelope exponent
- `num_readout_layers`: Number of linear layers in the atom-wise energy readout
- `readout_atom_norm`: If true, apply LayerNorm to atom features before the energy readout
- `interaction_dropout`: Dropout probability inside interaction blocks
- `atom_attention`: Cutoff-aware edge-aware atom attention update. It is enabled by default. When disabled, GPTFF uses the M3GNet-like atom sum update. When enabled, attention replaces the atom sum update and uses cutoff-weighted softmax plus radial-density context to preserve smoothness and coordination information. `num_heads` and `dropout` control the attention atom update. The optional atom FFN is disabled by default; set `use_ffn: true` to enable it, with `ffn_hidden_dim` and `ffn_residual_scale_init` controlling its hidden size and initial residual strength. `density_scale_init` controls the initial density scale.

`element_references`:
- `source`: Elemental reference energy source. Use `"fit"`, `null`, a mapping, a list, or a YAML/JSON file path. Mapping keys must be atomic numbers.

`optimizer`:
- `name`: Optimizer name. The default is `"AdamW"`; `"Adam"`, `"RAdam"`, and `"SGD"` are also supported.
- `learning_rate`: Optimizer learning rate
- `weight_decay`: Optimizer weight decay. The default is `1e-2` for `AdamW` and `0` for the other optimizers.
- `scheduler`: Learning-rate scheduler. The default is `"CosLR"`, a cosine annealing schedule. Use `"none"` to disable scheduling.
- `scheduler_params`: Optional scheduler parameters. For `"CosLR"`, `decay_fraction` controls `eta_min = decay_fraction * learning_rate`. Set `warmup_epochs` and `warmup_start_factor` to linearly warm up from `warmup_start_factor * learning_rate` before cosine decay. The scheduler is stepped 10 times per epoch by default, so `warmup_epochs: 3` means 30 warmup scheduler steps. Set `steps_per_epoch` to override this scheduler-step frequency. If `warmup_steps` is provided directly, it is counted in scheduler steps, not optimizer steps.

`training`:
- `epochs`: Number of training epochs
- `batch_size`: Number of structures in each per-GPU batch. With DDP, global
  batch size is `batch_size * world_size`.
- `num_workers`: Number of DataLoader workers per process. With DDP, total
  workers are `num_workers * world_size`.
- `persistent_workers`: If true, keep DataLoader workers alive across epochs.
  The default is false. For large sharded HDF5 graph datasets converted with a
  controlled shard count, enabling persistent workers can improve throughput by
  keeping HDF5 file handles and metadata caches warm across epochs.
- `device`: `cpu` or `cuda`
- `amp`: If true, enable CUDA automatic mixed precision during training
- `output_dir`: Directory for checkpoints and `history.csv`
- `seed`: Random seed shared by Python, NumPy, PyTorch, CUDA, and DataLoader shuffling
- `deterministic`: If true, require deterministic PyTorch algorithms and disable cuDNN benchmarking
- `distributed`: `auto`, `true`, or `false`. `auto` enables DDP when launched
  by `torchrun`; `true` requires a torchrun environment; `false` always runs
  single-process training.

Model checkpoints are written at epoch boundaries. `last.pt` stores the latest model,
`bestE.pt` stores the lowest validation energy MAE checkpoint, and `bestF.pt`
stores the lowest validation force MAE checkpoint.

`loss`:
- `energy_loss_weight`: Weight factor of the energy loss
- `force_loss_weight`: Weight factor of the force loss
- `stress_loss_weight`: Weight factor of the stress loss

`data`:
- `dataset_path`: Optional path to a training dataset. Use a Monty-serialized
  `AtomicDataset` JSON/JSON.GZ file for small and medium datasets, or a
  sharded HDF5 graph dataset directory for large precomputed graph datasets.
  This is required by the command-line trainer but not by `Trainer.fit(dataset)`.
- `dataset_format`: Dataset format. Use `atomic_json` for serialized
  `AtomicDataset` files and `sharded_hdf5_graph` for sharded graph dataset
  directories.
- `validation_fraction`: Fraction of samples reserved for validation.
- `test_fraction`: Fraction of samples reserved for testing. Use `0` to omit a
  test split.
- `split_seed`: Random seed used only for dataset splitting.
- `group_by_material`: If true, all samples with the same `material_id` remain
  in one split. Every sample must then define `material_id`.
- `cache_graphs`: If true, cache converted structure graphs in each DataLoader worker. Defaults to false.
- `graph_cache_size`: Maximum cached graph samples per worker. Use `null` for unlimited cache only when the dataset is small enough.

Energy and force labels are always required for training.
Stress labels are required only when `stress_loss_weight > 0`.
For energy-force data without stress, set `stress_loss_weight` to `0`.
Within one batch, each enabled label type must be present for every sample.

The split is generated from `split_seed` whenever training starts. Using the same
dataset order and split settings produces the same partitions.

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
