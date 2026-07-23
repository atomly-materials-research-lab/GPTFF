# Pretrained models

The checkpoints are grouped by their target application:

```text
pretrained/
├── inorganic/
│   ├── gptff_v1.pth
│   └── gptff_v2.pth
└── molecular/
    ├── energy/
    │   ├── config.json
    │   └── energy_v1.pth
    ├── force/
    │   ├── config.json
    │   └── force_v1.pth
    └── model/
        ├── __init__.py
        ├── model.py
        └── mpredict.py
```

## Inorganic models

- `inorganic/gptff_v1.pth`: the non-Transformer GPTFF model.
- `inorganic/gptff_v2.pth`: the Transformer-enabled GPTFF model.

These checkpoints use the inorganic reference energies defined by the main
`gptff.model.mpredict` module.

## Molecular models

- `molecular/energy/energy_v1.pth`: energy-oriented checkpoint trained with
  energy, force, and stress loss weights of `1.0`, `0.1`, and `0.0`.
- `molecular/force/force_v1.pth`: force-oriented checkpoint trained with
  energy, force, and stress loss weights of `1.0`, `5.0`, and `0.0`.

Each checkpoint is kept with its training configuration. The `molecular/model`
directory is the source snapshot supplied with the molecular checkpoints. In
particular, its `mpredict.py` contains the molecular reference energies, which
are different from those used by the inorganic models. It is retained as model
provenance and does not replace the actively maintained package code under
`gptff/`.
