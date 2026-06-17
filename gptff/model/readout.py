import torch
import torch.nn as nn

from gptff.model.element_refs import build_element_ref_tensor


class EnergyHead(nn.Module):
    def __init__(
        self,
        atom_fea_len,
        max_atomic_number=94,
        element_refs=None,
        n_readout_layers=3,
    ):
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.n_readout_layers = int(n_readout_layers)
        if self.n_readout_layers <= 0:
            raise ValueError("n_readout_layers must be positive.")

        layers = []
        for _ in range(self.n_readout_layers - 1):
            layers.extend([
                nn.Linear(atom_fea_len, atom_fea_len),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(atom_fea_len, 1))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer(
            "element_refs",
            build_element_ref_tensor(
                element_refs,
                max_atomic_number=self.max_atomic_number,
            ),
        )

    def forward(self, atom_fea, atom_types, atom_batch, num_graphs):
        site_energy = self.mlp(atom_fea)
        if self.element_refs is not None:
            site_energy = site_energy + self.element_refs[atom_types].unsqueeze(-1)

        energy = torch.zeros(
            (num_graphs, 1),
            dtype=site_energy.dtype,
            device=site_energy.device,
        )
        return torch.index_add(energy, 0, atom_batch, site_energy)
