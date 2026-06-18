import torch.nn as nn

from gptff.model.config import GPTFFConfig
from gptff.model.encoders import (
    AtomEmbedding,
    GeometryEmbedding,
)
from gptff.model.layers import InteractionBlock
from gptff.model.readout import EnergyHead


class GPTFF(nn.Module):
    def __init__(self, config: GPTFFConfig):
        super().__init__()

        atom_feature_dim = config.atom_feature_dim
        edge_feature_dim = config.edge_feature_dim
        num_interaction_blocks = config.num_interaction_blocks
        num_radial = config.num_radial
        num_angular = config.num_angular
        radial_cutoff = config.radial_cutoff
        angle_cutoff = config.angle_cutoff
        cutoff_coeff = config.cutoff_coeff
        max_atomic_number = config.max_atomic_number
        element_refs = config.element_refs

        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.radial_cutoff = radial_cutoff
        self.angle_cutoff = angle_cutoff
        self.max_atomic_number = max_atomic_number
        self.atom_embedding = AtomEmbedding(atom_feature_dim, max_atomic_number=max_atomic_number)
        self.geometry_embedding = GeometryEmbedding(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            num_radial=num_radial,
            radial_cutoff=radial_cutoff,
            angle_cutoff=angle_cutoff,
            cutoff_coeff=cutoff_coeff,
        )
        self.readout_atom_norm = (
            nn.LayerNorm(atom_feature_dim) if config.readout_atom_norm else nn.Identity()
        )

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    atom_feature_dim=atom_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                    num_angular=num_angular,
                    num_radial=num_radial,
                    dropout=config.interaction_dropout,
                    atom_attention_config=config.atom_attention,
                )
                for _ in range(num_interaction_blocks)
            ]
        )

        self.readout = EnergyHead(
            atom_feature_dim,
            max_atomic_number=max_atomic_number,
            element_refs=element_refs,
            num_readout_layers=config.num_readout_layers,
        )

    def _validate_graph_cutoffs(self, graph) -> None:
        _validate_cutoff("radial_cutoff", graph.radial_cutoff, self.radial_cutoff)
        _validate_cutoff("angle_cutoff", graph.angle_cutoff, self.angle_cutoff)

    def forward(self, graph):
        """
        graph: DifferentiableGraphBatch
        """

        self._validate_graph_cutoffs(graph)
        atom_features = self.atom_embedding(graph.atom_types)
        geometry_features = self.geometry_embedding(graph)
        edge_features = geometry_features.edge_features

        for interaction in self.interactions:
            atom_features, edge_features = interaction(
                atom_features,
                edge_features,
                graph,
                geometry_features,
            )

        atom_features = self.readout_atom_norm(atom_features)
        return self.readout(
            atom_features, graph.atom_types, graph.atom_batch, graph.num_atoms.shape[0]
        )


def _validate_cutoff(name: str, graph_value: float, model_value: float) -> None:
    graph_value = float(graph_value)
    model_value = float(model_value)
    if abs(graph_value - model_value) > 1e-6:
        raise ValueError(f"Graph {name} {graph_value} does not match model {name} {model_value}.")
