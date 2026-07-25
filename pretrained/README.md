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
    └── force/
        ├── config.json
        └── force_v1.pth
```

## Inorganic models

- `inorganic/gptff_v1.pth`: the non-Transformer GPTFF model.
- `inorganic/gptff_v2.pth`: the Transformer-enabled GPTFF model.

These checkpoints use the inorganic profile defined by
`gptff.model.reference_energies`.

## Molecular models

- `molecular/energy/energy_v1.pth`: energy-oriented checkpoint trained with
  energy, force, and stress loss weights of `1.0`, `0.1`, and `0.0`.
- `molecular/force/force_v1.pth`: force-oriented checkpoint trained with
  energy, force, and stress loss weights of `1.0`, `5.0`, and `0.0`.

Each checkpoint is kept with its training configuration. The molecular
checkpoints use the actively maintained model and inference implementation
under `gptff/`. Load them with `gptff.model.mpredict.ASECalculator` and pass
`reference_energies="molecular"` to select their molecular reference energies.
