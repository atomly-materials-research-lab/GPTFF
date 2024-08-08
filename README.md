GPTFF (Graph-based Pretrained Transformer Force Field) can simulate arbitrary inorganic systems with good precision and generalizability.

## Installation

Using `conda` to create a new python virtual env(not necessary):

```bash
conda create -n gptff python=3.8
```

Then clone the `GPTFF` repo and install:

```bash
git clone https://github.com/atomly-materials-research-lab/GPTFF.git
cd GPTFF
pip install .
```

## Usage

**Fast Energy(eV), Force(eV/Å), Stress(GPa) calculation:**

```python
from gptff.model.mpredict import ASECalculator
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

Lattice vectors would be not change, only atomic positions would be optimized

```python
from gptff.model.mpredict import ASECalculator
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
from gptff.model.mpredict import ASECalculator
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

`config.json` would be training parameters, you could specify data path in this file.

```bash
gptff_trainer config.json
```

## Data 

If you want to pretrain or finetune the force field based on your own dataset, you can prepare your own dataset as below:

The dataset must be store in `.csv` format file, there are several columns:

`struct_id`: Unique structure id, e.g. 0, 1, 2, ..

`energy`: Total energy of the structure (eV)

`forces`: The forces of each atom (eV/Å)

`stress`: The stress of the structure (kBar, align with VASP stress output directly)

`structure`: dict format of the structure. 

```python
from pymatgen.core import Structure
struc = Structure.from_file('POSCAR')
struc_data = struc.as_dict()
```

`fold`: You can specify which fold to be trained and which fold to be validated. If you set fold in config.json is `0`, the the `fold !=0` is training dataset, `fold == 0` would be validation dataset.

`ref_energy`: Reference energy of the structure, 
For example, the formula of the structure is Li2O4, the ref_energy of Li2O4 is: atom_refs[3] * 2 + atom_refs[8] * 4. `3` and `8` are atomic order of Li and O, `2` and `4` are atom numbers in the structure.


In the model we have pretrained, the `atom_refs` is:

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

Or you can fit you own `atom_refs`.

## Training setting

The file `config.json` includes training settings, 

- workers: The number of workers for dataloader
- epochs: The number of training epochs
- batch_size: batch size for training, the number of structures used in one step(batch)
- node_feature_len: The size of the node(atom) feature length
- edge_feature_len: The size of the edge(bond) feature length
- n_layers: THe number of layers of GPTFF model
- device: `cpu` or `cuda`
- val_fold: Label validation data during training
- transformer_activate: If activate `transformer` block or not
- weight_energy: Weight factor of the energy
- weight_force: Weight factor of the forces
- weight_stress: Weight factor of the stress, if there's not stress data, please set it to `0`

## Reference

If you found GPTFF useful, please cite our article:

```
@ARTICLE{2024arXiv240219327X,
       author = {{Xie}, Fankai and {Lu}, Tenglong and {Meng}, Sheng and {Liu}, Miao},
        title = "{GPTFF: A high-accuracy out-of-the-box universal AI force field for arbitrary inorganic materials}",
      journal = {arXiv e-prints},
     keywords = {Condensed Matter - Materials Science},
         year = 2024,
        month = feb,
          eid = {arXiv:2402.19327},
        pages = {arXiv:2402.19327},
          doi = {10.48550/arXiv.2402.19327},
archivePrefix = {arXiv},
       eprint = {2402.19327},
 primaryClass = {cond-mat.mtrl-sci},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240219327X},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
