Graph-based Pretrained Transformer Force Field xxx

## Installation

Using `conda` to create a new python virtual env(not necessary):

```bash
conda create -n gptff python=3.8
```

Then clone the `GPTFF` repo and install:

```bash
git clone xxx
cd gptff
pip install .
```

## Usage

**Fast Energy(eV), Force(eV/Å), Stress(GPa) calculation:**

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

model_weight = "xxx"
n_layers = 3
p = ASECalculator(model_weight, n_layers) # Initialize the model and load weights

adp = AseAtomsAdaptor()
struc = Structure.from_file('POSCAR_structure')
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

energy = atoms.get_potential_energy() # unit (eV)
forces = atoms.get_forces() # unit (eV/Å)
stress = atoms.get_stress() # unit (GPa)
```

**Structure Optimization:**

Lattice vectors would be changed

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter, StrainFilter

model_weight = "xxx"
n_layers = 3
p = ASECalculator(model_weight, n_layers) # Initialize the model and load weights

struc = Structure.from_file('POSCAR_structure') # Read structure

adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

optimizer = ExpCellFilter(atoms) 

FIRE(optimizer).run(fmax=0.01, steps=100)

```

Lattice vectors would be not change, only atomic positions would be optimized

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.optimize.bfgs import BFGS

model_weight = "xxx"
n_layers = 3
p = ASECalculator(model_weight, n_layers) # Initialize the model and load weights

struc = Structure.from_file('POSCAR_structure') # Read structure

adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

optimizer = BFGS(atoms)
optimizer.run(fmax=0.01, steps=1000)
```

**Molecular dynamics (ASE):**

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms, units
from ase.md.nvtberendsen import NVTBerendsen
import os

model_weight = "xxx"
n_layers = 3
p = ASECalculator(model_weight, n_layers) # Initialize the model and load weights

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

`config.json` would be training parameters, you could specify data path in this file.

```bash
gptff_trainer config.json
```

