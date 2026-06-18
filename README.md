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

**Fast Energy(eV), Force(eV/Å), Stress(eV/Å^3, ASE Voigt) calculation:**

```python
from gptff.model import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

model_weight = "pretrained/gptff_v1.pth"
device = 'cuda' # or cpu
p = ASECalculator(model_weight, device) # Initialize the model and load weights

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
from gptff.model import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.filters import ExpCellFilter, StrainFilter

model_weight = "pretrained/gptff_v1.pth"
device = 'cuda' # or cpu
p = ASECalculator(model_weight, device) # Initialize the model and load weights


struc = Structure.from_file('POSCAR_structure') # Read structure

adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

optimizer = ExpCellFilter(atoms) 

FIRE(optimizer).run(fmax=0.01, steps=100)

```

Lattice vectors would not change; only atomic positions would be optimized.

```python
from gptff.model import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.optimize.bfgs import BFGS

model_weight = "pretrained/gptff_v1.pth"
device = 'cuda' # or cpu
p = ASECalculator(model_weight, device) # Initialize the model and load weights


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
from gptff.model import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms, units
from ase.md.nvtberendsen import NVTBerendsen
import os

model_weight = "pretrained/gptff_v1.pth"
device = 'cuda' # or cpu
p = ASECalculator(model_weight, device) # Initialize the model and load weights


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
gptff_trainer config.yaml
```

Elemental reference energies are configured through the top-level `element_references` section, not stored as a dataset column. Set `source: atomly` to use the built-in Atomly reference preset, provide your own mapping/list as `source`, set `source: fit` to fit references from the training split, set `source: null` to disable references, or point `source` to a YAML/JSON reference file.

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


The built-in `"atomly"` reference preset is:

```python
atom_refs = np.array([ 
       0.00000000e+00, -3.46535853e+00, -7.56101906e-01, -3.46224791e+00,  
       -4.77600176e+00, -8.03619240e+00, -8.40374071e+00, -7.76814618e+00,
       -7.38918302e+00, -4.94725878e+00, -2.92883670e-02, -2.47830716e+00,
       -2.02015956e+00, -5.15479820e+00, -7.91209653e+00, -6.91345095e+00,
       -4.62278149e+00, -3.01552069e+00, -6.27971322e-02, -2.31732442e+00,
       -4.75968073e+00, -8.17421803e+00, -1.14207788e+01, -8.92294483e+00,
       -8.48981509e+00, -8.16635547e+00, -6.58248850e+00, -5.26139665e+00,
       -4.48412068e+00, -3.27367370e+00, -1.34976438e+00, -3.62637456e+00,
       -4.67270042e+00, -4.13166577e+00, -3.67546394e+00, -2.80302539e+00,
        6.47272418e+00, -2.24681188e+00, -4.25110577e+00, -1.02452951e+01,
       -1.16658385e+01, -1.18015760e+01, -8.65537518e+00, -9.36409198e+00,
       -7.57165084e+00, -5.69907599e+00, -4.97159232e+00, -1.88700594e+00,
       -6.79483530e-01, -2.74880153e+00, -3.79441765e+00, -3.38825264e+00,
       -2.55867271e+00, -1.96213610e+00,  9.97909972e+00, -2.55677995e+00,
       -4.88030347e+00, -8.86033743e+00, -9.05368602e+00, -7.94309693e+00,
       -8.12585485e+00, -6.31826210e+00, -8.30242223e+00, -1.22893251e+01,
       -1.73097460e+01, -7.55105974e+00, -8.19580521e+00, -8.34926874e+00,
       -7.25911206e+00, -8.41697224e+00, -3.38725429e+00, -7.68222088e+00,
       -1.26297007e+01, -1.36257602e+01, -9.52985029e+00, -1.18396814e+01,
       -9.79914325e+00, -7.55608603e+00, -5.46902454e+00, -2.65092136e+00,
        4.17472161e-01, -2.32548971e+00, -3.48299933e+00, -3.18067109e+00,
        3.57605604e-15,  9.96350211e-16,  1.18278079e-15, -1.44201673e-15,
       -6.73760309e-18, -5.48347781e+00, -1.03346396e+01, -1.11296117e+01,
       -1.43116273e+01, -1.47003999e+01, -1.54726487e+01])
```

## Training setting

The file `config.yaml` uses separate sections for model, optimizer, training loop, loss, and data settings.

`model`:
- `atom_feature_dim`: Atom feature dimension
- `edge_feature_dim`: Edge feature dimension
- `num_interaction_blocks`: Number of GPTFF interaction blocks
- `num_radial`: Number of radial basis functions
- `num_angular`: Number of Fourier angular basis frequencies
- `radial_cutoff`: Pair graph cutoff radius
- `angle_cutoff`: Three-body angle cutoff radius
- `cutoff_coeff`: Polynomial cutoff envelope exponent
- `num_readout_layers`: Number of linear layers in the atom-wise energy readout
- `readout_atom_norm`: If true, apply LayerNorm to atom features before the energy readout
- `interaction_dropout`: Dropout probability inside interaction blocks
- `atom_attention`: Optional cutoff-aware invariant atom attention mixer. It is disabled by default. When enabled, `num_heads`, `dropout`, `use_ffn`, and `ffn_hidden_dim` control the residual attention branch after atom update.

`element_references`:
- `source`: Elemental reference energy source. Use `"atomly"`, `"fit"`, `null`, a mapping, a list, or a YAML/JSON file path. Mapping keys must be atomic numbers.
- `ridge`: Ridge regularization used when `source: fit`

`optimizer`:
- `name`: Optimizer name. The default is `"AdamW"`; `"Adam"`, `"RAdam"`, and `"SGD"` are also supported.
- `learning_rate`: Optimizer learning rate
- `weight_decay`: Optimizer weight decay. The default is `1e-2` for `AdamW` and `0` for the other optimizers.
- `scheduler`: Learning-rate scheduler. The default is `"CosLR"`, a cosine annealing schedule. Use `"none"` to disable scheduling.
- `scheduler_params`: Optional scheduler parameters. For `"CosLR"`, `decay_fraction` controls `eta_min = decay_fraction * learning_rate`.

`training`:
- `epochs`: Number of training epochs
- `batch_size`: Number of structures in each batch
- `num_workers`: Number of DataLoader workers
- `device`: `cpu` or `cuda`
- `amp`: If true, enable CUDA automatic mixed precision during training
- `output_dir`: Directory for checkpoints and `history.csv`
- `seed`: Random seed shared by Python, NumPy, PyTorch, CUDA, and DataLoader shuffling
- `deterministic`: If true, require deterministic PyTorch algorithms and disable cuDNN benchmarking

Model checkpoints are written at epoch boundaries.

`loss`:
- `energy_loss_weight`: Weight factor of the energy loss
- `force_loss_weight`: Weight factor of the force loss
- `stress_loss_weight`: Weight factor of the stress loss

`data`:
- `dataset_path`: Optional path to a Monty-serialized `AtomicDataset`. This is
  required by the command-line trainer but not by `Trainer.fit(dataset)`.
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
