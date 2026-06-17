import torch
import torch.nn as nn

from gptff.model.element_refs import build_element_ref_tensor


class EnergyHead(nn.Module):
    def __init__(self, atom_fea_len, max_atomic_number=94, element_refs=None):
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc2 = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc_out = nn.Linear(atom_fea_len, 1)
        self.register_buffer(
            "element_refs",
            build_element_ref_tensor(
                element_refs,
                max_atomic_number=self.max_atomic_number,
            ),
        )

    def forward(self, atom_fea, atom_types, atom_batch, num_graphs):
        site_energy = self.swish(self.fc1(atom_fea))
        site_energy = self.swish(self.fc2(site_energy))
        site_energy = self.fc_out(site_energy)
        if self.element_refs is not None:
            site_energy = site_energy + self.element_refs[atom_types].unsqueeze(-1)

        energy = torch.zeros(
            (num_graphs, 1),
            dtype=site_energy.dtype,
            device=site_energy.device,
        )
        return torch.index_add(energy, 0, atom_batch, site_energy)
