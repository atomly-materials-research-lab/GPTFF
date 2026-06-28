import pytest
import torch
import torch.nn as nn
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFF, GPTFFConfig
from gptff.model.encoders import EdgeModulation, GeometryFeatures
from gptff.model.layers import (
    AtomFeatureDelta,
    AttentionAtomUpdate,
    DensityContext,
    InteractionBlock,
    RadialDensityFeatures,
    cutoff_weighted_softmax,
    sum_aggregation,
)


def _cfg(num_interaction_blocks=1, **kwargs):
    defaults = {
        "atom_feature_dim": 8,
        "edge_feature_dim": 8,
        "num_interaction_blocks": num_interaction_blocks,
        "num_radial": 8,
        "num_angular": 4,
        "radial_cutoff": 3.0,
        "angle_cutoff": 3.0,
        "cutoff_coeff": 5,
        "atom_attention": {"enabled": False},
    }
    defaults.update(kwargs)
    return GPTFFConfig(**defaults)


def test_gptff_config_enables_atom_attention_by_default():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
    )

    assert cfg.atom_attention.enabled is True
    assert cfg.atom_attention.use_ffn is False


def test_gptff_config_rejects_attention_heads_that_do_not_divide_atom_features():
    with pytest.raises(ValueError, match="atom_feature_dim must be divisible"):
        GPTFFConfig(
            atom_feature_dim=10,
            edge_feature_dim=8,
            num_interaction_blocks=1,
            num_radial=8,
            num_angular=4,
            radial_cutoff=3.0,
            angle_cutoff=3.0,
            cutoff_coeff=5,
            atom_attention={"enabled": True, "num_heads": 4},
        )


def _batch(angle_cutoff=3.0):
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=angle_cutoff).convert(structure)
    return CrystalGraphBatch.from_graphs([graph]).with_geometry()


def _features(model, graph):
    return model.geometry_embedding(graph)


def test_default_model_uses_interaction_blocks():
    model = GPTFF(_cfg(num_interaction_blocks=2))

    assert len(model.interactions) == 2
    assert isinstance(model.interactions[0], InteractionBlock)
    assert isinstance(model.interactions[0].pair_atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].triplet_atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].atom_norm, nn.LayerNorm)
    assert not hasattr(model.interactions[0], "triplet_edge_norm")
    assert not hasattr(model.interactions[0], "pair_edge_norm")
    assert not hasattr(model.interactions[0], "atom_edge_norm")
    assert not hasattr(model.interactions[0], "residual_zero_init")
    assert not hasattr(model.interactions[0].edge_update, "message_projection")
    assert isinstance(model.interactions[0].atom_update, AtomFeatureDelta)
    assert model.interactions[0].atom_ffn is None
    assert model.interactions[0].atom_ffn_norm is None
    assert model.interactions[0].atom_ffn_residual_scale is None


def test_interaction_block_returns_finite_atom_and_edge_features():
    graph = _batch()
    model = GPTFF(_cfg())

    atom_fea = model.atom_embedding(graph.atom_types)
    features = _features(model, graph)
    edge_ij = features.edge_features

    atom_out, edge_out = model.interactions[0](
        atom_fea,
        edge_ij,
        graph,
        features,
    )

    assert atom_out.shape == atom_fea.shape
    assert edge_out.shape == edge_ij.shape
    assert torch.isfinite(atom_out).all()
    assert torch.isfinite(edge_out).all()


def test_interaction_block_handles_no_edge_graph():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        coords_are_cartesian=True,
    )
    graph = CrystalGraphConverter(radial_cutoff=1.0, angle_cutoff=1.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()
    model = GPTFF(_cfg(radial_cutoff=1.0, angle_cutoff=1.0))

    energy = model(batch)

    assert graph.num_edges == 0
    assert energy.shape == (1, 1)
    assert torch.isfinite(energy).all()


def test_three_body_edge_delta_is_zero_without_angle_triplets():
    graph = _batch(angle_cutoff=1.0)
    model = GPTFF(_cfg())

    assert graph.triplet_edge_index.numel() == 0

    atom_fea = model.atom_embedding(graph.atom_types)
    features = _features(model, graph)
    edge_ij = features.edge_features
    delta = model.interactions[0].three_body(
        model.interactions[0].triplet_atom_norm(atom_fea),
        edge_ij,
        graph,
        features.triplet_modulation,
    )

    assert delta.shape == edge_ij.shape
    assert torch.equal(delta, torch.zeros_like(edge_ij))


def test_sum_aggregation_adds_values_by_index():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([0, 1, 0])
    reference = torch.zeros((2, 2))

    aggregated = sum_aggregation(
        values,
        indices,
        dim_size=2,
        reference=reference,
    )

    assert torch.equal(
        aggregated,
        torch.tensor([[6.0, 8.0], [3.0, 4.0]]),
    )


def test_cutoff_weighted_softmax_removes_zero_cutoff_edges():
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    indices = torch.tensor([0, 0, 1])
    edge_cutoff = torch.tensor([[1.0], [0.0], [1.0]])

    attention = cutoff_weighted_softmax(
        logits,
        indices,
        dim_size=2,
        edge_cutoff=edge_cutoff,
    )

    assert torch.allclose(attention[0], torch.ones(2))
    assert torch.equal(attention[1], torch.zeros(2))
    assert torch.allclose(attention[2], torch.ones(2))


def test_cutoff_weighted_softmax_ignores_zero_cutoff_logits_in_stabilization():
    logits = torch.tensor([[0.0], [1_000.0], [1.0]])
    indices = torch.tensor([0, 0, 0])
    edge_cutoff = torch.tensor([[1.0], [0.0], [1.0]])

    attention = cutoff_weighted_softmax(
        logits,
        indices,
        dim_size=1,
        edge_cutoff=edge_cutoff,
    )

    expected = torch.softmax(torch.tensor([[0.0], [1.0]]).reshape(-1), dim=0)
    assert torch.allclose(attention[[0, 2], 0], expected)
    assert attention[1, 0] == 0


def test_attention_atom_update_returns_zero_with_zero_cutoff_inputs():
    graph = _batch()
    model = GPTFF(_cfg())
    features = _features(model, graph)
    atom_fea = model.atom_embedding(graph.atom_types)
    attention = AttentionAtomUpdate(
        atom_feature_dim=model.atom_feature_dim,
        edge_feature_dim=model.edge_feature_dim,
        num_radial=model.num_radial,
        num_heads=2,
    )

    delta = attention(
        atom_fea,
        features.edge_features,
        torch.zeros_like(features.edge_modulation.atom_message),
        graph,
        torch.zeros_like(features.edge_basis),
        torch.zeros_like(features.edge_cutoff),
    )

    assert torch.equal(delta, torch.zeros_like(atom_fea))


def test_attention_atom_update_output_gate_uses_smooth_gated_center_atom_features():
    graph = _batch()
    model = GPTFF(_cfg())
    features = _features(model, graph)
    atom_fea = model.atom_embedding(graph.atom_types)
    attention = AttentionAtomUpdate(
        atom_feature_dim=model.atom_feature_dim,
        edge_feature_dim=model.edge_feature_dim,
        num_radial=model.num_radial,
        num_heads=2,
    )
    attention.output_gate = _RecordingConstantFeature(
        output_dim=model.atom_feature_dim,
        value=0.0,
    )

    attention(
        atom_fea,
        features.edge_features,
        features.edge_modulation.atom_message,
        graph,
        features.edge_basis,
        features.edge_cutoff,
    )

    seen_input = attention.output_gate.seen_input
    expected_degree = torch.zeros((atom_fea.shape[0], 1), dtype=atom_fea.dtype)
    expected_degree.index_add_(0, graph.edge_index[0], features.edge_cutoff)
    expected_center_context = atom_fea * torch.tanh(expected_degree)
    assert seen_input.shape == (atom_fea.shape[0], 3 * model.atom_feature_dim)
    assert torch.allclose(seen_input[:, : model.atom_feature_dim], expected_center_context)


def test_attention_atom_update_rescales_uniform_attention_to_cutoff_sum():
    graph = _batch()
    model = GPTFF(_cfg())
    features = _features(model, graph)
    atom_fea = model.atom_embedding(graph.atom_types)
    attention = AttentionAtomUpdate(
        atom_feature_dim=model.atom_feature_dim,
        edge_feature_dim=model.edge_feature_dim,
        num_radial=model.num_radial,
        num_heads=2,
    )
    attention.score = _ConstantFeature(output_dim=attention.num_heads, value=0.0)
    attention.value = _ConstantFeature(output_dim=model.atom_feature_dim, value=1.0)
    attention.density_context.hidden.weight.data.zero_()
    attention.density_context.output.weight.data.zero_()
    attention.output_gate = _RecordingConstantFeature(
        output_dim=model.atom_feature_dim,
        value=0.0,
    )

    attention(
        atom_fea,
        features.edge_features,
        torch.ones_like(features.edge_modulation.atom_message),
        graph,
        features.edge_basis,
        features.edge_cutoff,
    )

    seen_input = attention.output_gate.seen_input
    expected_degree = torch.zeros((atom_fea.shape[0], 1), dtype=atom_fea.dtype)
    expected_degree.index_add_(0, graph.edge_index[0], features.edge_cutoff)
    expected_attention_sum = expected_degree.expand(-1, model.atom_feature_dim)
    start = model.atom_feature_dim
    end = 2 * model.atom_feature_dim
    assert torch.allclose(seen_input[:, start:end], expected_attention_sum)


def test_attention_atom_update_no_edge_graph_skips_self_update():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        coords_are_cartesian=True,
    )
    graph = CrystalGraphConverter(radial_cutoff=1.0, angle_cutoff=1.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()
    model = GPTFF(_cfg(radial_cutoff=1.0, angle_cutoff=1.0))
    features = _features(model, batch)
    atom_fea = model.atom_embedding(batch.atom_types)
    attention = AttentionAtomUpdate(
        atom_feature_dim=model.atom_feature_dim,
        edge_feature_dim=model.edge_feature_dim,
        num_radial=model.num_radial,
        num_heads=2,
    )
    attention.output_gate = _RecordingConstantFeature(
        output_dim=model.atom_feature_dim,
        value=1.0,
    )

    delta = attention(
        atom_fea,
        features.edge_features,
        features.edge_modulation.atom_message,
        batch,
        features.edge_basis,
        features.edge_cutoff,
    )

    assert graph.num_edges == 0
    assert torch.equal(delta, torch.zeros_like(atom_fea))
    assert attention.output_gate.seen_input is None


def test_interaction_block_can_enable_atom_attention():
    graph = _batch()
    cfg = _cfg(
        atom_attention={
            "enabled": True,
            "num_heads": 2,
            "dropout": 0.0,
            "use_ffn": True,
            "ffn_residual_scale_init": 0.05,
        },
    )
    model = GPTFF(cfg)

    atom_fea = model.atom_embedding(graph.atom_types)
    features = _features(model, graph)
    atom_out, edge_out = model.interactions[0](
        atom_fea,
        features.edge_features,
        graph,
        features,
    )

    assert isinstance(model.interactions[0].atom_update, AttentionAtomUpdate)
    assert isinstance(model.interactions[0].atom_update.radial_density_features, RadialDensityFeatures)
    assert isinstance(model.interactions[0].atom_update.density_context, DensityContext)
    assert torch.allclose(
        model.interactions[0].atom_ffn_residual_scale,
        torch.tensor(0.05),
    )
    assert atom_out.shape == atom_fea.shape
    assert edge_out.shape == features.edge_features.shape
    assert torch.isfinite(atom_out).all()
    assert torch.isfinite(edge_out).all()


def test_interaction_block_enables_radial_density_with_atom_attention():
    cfg = _cfg(
        atom_attention={
            "enabled": True,
            "num_heads": 2,
        },
    )
    model = GPTFF(cfg)

    assert isinstance(model.interactions[0].atom_update, AttentionAtomUpdate)
    assert isinstance(model.interactions[0].atom_update.radial_density_features, RadialDensityFeatures)
    assert isinstance(model.interactions[0].atom_update.density_context, DensityContext)
    assert model.interactions[0].atom_ffn is None
    assert model.interactions[0].atom_ffn_norm is None
    assert model.interactions[0].atom_ffn_residual_scale is None


def test_attention_atom_update_replaces_sum_update_and_ffn_scales_afterward():
    graph = _batch()
    cfg = _cfg(
        atom_attention={
            "enabled": True,
            "num_heads": 2,
            "dropout": 0.0,
            "use_ffn": True,
        },
    )
    model = GPTFF(cfg)
    block = model.interactions[0]
    block.three_body = _ConstantTripletDelta(value=0.0)
    block.edge_update = _ConstantPairDelta(value=0.0)
    block.atom_update = _ConstantAtomDelta(value=3.0)
    block.atom_ffn = _ConstantFeature(output_dim=model.atom_feature_dim, value=5.0)
    block.atom_ffn_residual_scale.data.fill_(0.3)

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_feature_dim))
    edge_ij = torch.zeros((graph.edge_index.shape[1], model.edge_feature_dim))
    features = GeometryFeatures(
        edge_basis=torch.zeros((edge_ij.shape[0], model.num_radial)),
        edge_cutoff=torch.ones((edge_ij.shape[0], 1)),
        angle_radial_basis=torch.empty((edge_ij.shape[0], 0)),
        edge_features=edge_ij,
        edge_modulation=EdgeModulation(
            atom_message=torch.zeros_like(atom_fea[graph.edge_index[0]]),
            edge_message=torch.zeros_like(edge_ij),
        ),
        triplet_modulation=torch.zeros_like(edge_ij),
    )

    atom_out, _ = block(atom_fea, edge_ij, graph, features)

    expected_value = 3.0 + 0.3 * 5.0
    assert torch.allclose(atom_out, torch.full_like(atom_fea, expected_value))


def test_interaction_block_uses_unscaled_residual_addition():
    graph = _batch()
    model = GPTFF(_cfg())
    block = model.interactions[0]
    block.pair_atom_norm = nn.Identity()
    block.triplet_atom_norm = nn.Identity()
    block.atom_norm = nn.Identity()
    block.three_body = _ConstantTripletDelta(value=2.0)
    block.edge_update = _ConstantPairDelta(value=3.0)
    block.atom_update = _ConstantAtomDelta(value=4.0)

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_feature_dim))
    edge_ij = torch.zeros((graph.edge_index.shape[1], model.edge_feature_dim))
    features = GeometryFeatures(
        edge_basis=torch.empty((edge_ij.shape[0], 0)),
        edge_cutoff=torch.ones((edge_ij.shape[0], 1)),
        angle_radial_basis=torch.empty((edge_ij.shape[0], 0)),
        edge_features=edge_ij,
        edge_modulation=EdgeModulation(
            atom_message=torch.zeros_like(atom_fea[graph.edge_index[0]]),
            edge_message=torch.zeros_like(edge_ij),
        ),
        triplet_modulation=torch.zeros_like(edge_ij),
    )

    atom_out, edge_out = block(
        atom_fea,
        edge_ij,
        graph,
        features,
    )

    assert torch.equal(edge_out, torch.full_like(edge_ij, 5.0))
    assert torch.equal(atom_out, torch.full_like(atom_fea, 4.0))


def test_zero_geometry_modulation_zeroes_interaction_deltas():
    graph = _batch()
    model = GPTFF(_cfg())
    block = model.interactions[0]

    atom_fea = model.atom_embedding(graph.atom_types)
    features = _features(model, graph)
    edge_ij = features.edge_features
    zero_modulation = EdgeModulation(
        atom_message=torch.zeros_like(atom_fea[graph.edge_index[0]]),
        edge_message=torch.zeros_like(edge_ij),
    )
    zero_triplet_modulation = torch.zeros_like(edge_ij)

    triplet_delta = block.three_body(
        block.triplet_atom_norm(atom_fea),
        edge_ij,
        graph,
        zero_triplet_modulation,
    )
    pair_delta = block.edge_update(
        block.pair_atom_norm(atom_fea),
        edge_ij,
        graph,
        zero_modulation.edge_message,
    )
    atom_delta = block.atom_update(
        block.atom_norm(atom_fea),
        edge_ij,
        zero_modulation.atom_message,
        graph,
    )

    assert torch.equal(triplet_delta, torch.zeros_like(triplet_delta))
    assert torch.equal(pair_delta, torch.zeros_like(pair_delta))
    assert torch.equal(atom_delta, torch.zeros_like(atom_delta))


def test_atom_update_uses_sum_aggregation():
    graph = _batch()
    model = GPTFF(_cfg())
    block = model.interactions[0]
    block.atom_update.message_encoder = _ConstantFeature(
        output_dim=2 * model.atom_feature_dim,
        value=1.0,
    )
    block.atom_update.message_gate = _ConstantFeature(
        output_dim=model.atom_feature_dim,
        value=1.0,
    )

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_feature_dim))
    edge_ij = torch.zeros((graph.edge_index.shape[1], model.edge_feature_dim))
    atom_delta = block.atom_update(
        atom_fea,
        edge_ij,
        torch.ones_like(atom_fea[graph.edge_index[0]]),
        graph,
    )
    expected = torch.zeros_like(atom_fea)
    expected.index_add_(0, graph.edge_index[0], torch.ones_like(atom_fea[graph.edge_index[0]]))

    assert torch.equal(atom_delta, expected)


def test_radial_density_features_use_sum_aggregation():
    graph = _batch()
    model = GPTFF(_cfg())
    radial_density = RadialDensityFeatures(
        atom_feature_dim=model.atom_feature_dim,
        num_radial=model.num_radial,
    )
    radial_density.radial_density.weight.data.fill_(1.0)

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_feature_dim))
    edge_basis = torch.ones((graph.edge_index.shape[1], model.num_radial))
    edge_cutoff = torch.ones((graph.edge_index.shape[1], 1))

    density_features = radial_density(
        atom_fea,
        graph,
        edge_basis,
        edge_cutoff,
    )
    expected_degree = torch.zeros((atom_fea.shape[0], 1))
    expected_degree.index_add_(0, graph.edge_index[0], edge_cutoff)
    expected_radial = torch.zeros_like(atom_fea)
    expected_radial.index_add_(
        0,
        graph.edge_index[0],
        torch.full_like(atom_fea[graph.edge_index[0]], float(model.num_radial)),
    )
    expected = torch.cat([expected_degree, expected_radial], dim=-1)

    assert torch.allclose(density_features, expected)


def test_radial_density_features_return_zero_with_zero_cutoff_inputs():
    graph = _batch()
    model = GPTFF(_cfg())
    radial_density = RadialDensityFeatures(
        atom_feature_dim=model.atom_feature_dim,
        num_radial=model.num_radial,
    )

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = torch.zeros((graph.edge_index.shape[1], model.num_radial))

    density_features = radial_density(
        atom_fea,
        graph,
        edge_basis,
        torch.zeros((graph.edge_index.shape[1], 1)),
    )

    assert torch.equal(density_features, atom_fea.new_zeros((atom_fea.shape[0], atom_fea.shape[1] + 1)))


def test_density_context_scale_is_identity_for_zero_density():
    context = DensityContext(
        density_feature_dim=9,
        atom_feature_dim=8,
        scale_init=0.1,
    )
    density_features = torch.zeros(3, 9)

    density_context = context(density_features)
    density_scale = context.compute_scale(density_context)

    assert torch.equal(density_context, torch.zeros_like(density_context))
    assert torch.equal(density_scale, torch.ones_like(density_scale))


def test_density_context_initial_scale_deviation_is_bounded():
    context = DensityContext(
        density_feature_dim=9,
        atom_feature_dim=8,
        scale_init=0.1,
    )
    density_features = torch.randn(5, 9)

    density_scale = context.compute_scale(context(density_features))

    assert torch.max(torch.abs(density_scale - 1.0)) <= 0.1


def test_three_body_update_uses_sum_aggregation():
    graph = _batch()
    model = GPTFF(_cfg())
    block = model.interactions[0]
    block.three_body.target_encoder = _ConstantFeature(
        output_dim=model.edge_feature_dim,
        value=1.0,
    )
    block.three_body.source_encoder = _ConstantFeature(
        output_dim=model.edge_feature_dim,
        value=1.0,
    )
    block.three_body.output_gate = nn.Identity()

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_feature_dim))
    edge_ij = torch.zeros((graph.edge_index.shape[1], model.edge_feature_dim))
    triplet_delta = block.three_body(
        atom_fea,
        edge_ij,
        graph,
        torch.ones_like(edge_ij),
    )
    expected = torch.zeros_like(edge_ij)
    expected.index_add_(
        0,
        graph.triplet_edge_index[0],
        torch.ones_like(edge_ij[graph.triplet_edge_index[0]]),
    )

    assert torch.equal(triplet_delta, expected)


def test_three_body_update_factorizes_target_and_source_features():
    graph = _batch()
    model = GPTFF(_cfg())
    block = model.interactions[0]
    target_encoder = _RecordingConstantFeature(
        output_dim=model.edge_feature_dim,
        value=1.0,
    )
    source_encoder = _RecordingConstantFeature(
        output_dim=model.edge_feature_dim,
        value=1.0,
    )
    block.three_body.target_encoder = target_encoder
    block.three_body.source_encoder = source_encoder
    block.three_body.output_gate = nn.Identity()

    atom_fea = torch.arange(
        graph.atom_types.shape[0] * model.atom_feature_dim,
        dtype=torch.float32,
    ).reshape(graph.atom_types.shape[0], model.atom_feature_dim)
    edge_ij = torch.arange(
        graph.edge_index.shape[1] * model.edge_feature_dim,
        dtype=torch.float32,
    ).reshape(graph.edge_index.shape[1], model.edge_feature_dim)
    triplet_modulation = torch.ones_like(edge_ij)

    block.three_body(atom_fea, edge_ij, graph, triplet_modulation)

    edge_ij_indices = graph.triplet_edge_index[0]
    edge_ik_indices = graph.triplet_edge_index[1]
    expected_target_input = torch.cat(
        [
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ],
        dim=-1,
    )
    expected_source_input = torch.cat(
        [
            atom_fea[graph.edge_index[1][edge_ik_indices]],
            edge_ij[edge_ik_indices],
            block.three_body.angle_basis(graph.triplet_cosine),
        ],
        dim=-1,
    )

    assert torch.equal(target_encoder.seen_input, expected_target_input)
    assert torch.equal(source_encoder.seen_input, expected_source_input)
    assert edge_ij_indices.numel() == source_encoder.seen_input.shape[0]


def test_interaction_block_updates_triplet_then_pair_then_atom():
    graph = _batch()
    model = GPTFF(_cfg())
    block = model.interactions[0]
    block.pair_atom_norm = nn.Identity()
    block.triplet_atom_norm = nn.Identity()
    block.atom_norm = nn.Identity()
    block.edge_update = _ConstantPairDelta(value=1.0)
    block.three_body = _ConstantTripletDelta(value=2.0)
    block.atom_update = _RecordingAtomDelta()

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_feature_dim))
    edge_ij = torch.zeros((graph.edge_index.shape[1], model.edge_feature_dim))
    edge_modulation = EdgeModulation(
        atom_message=torch.zeros_like(atom_fea[graph.edge_index[0]]),
        edge_message=torch.zeros_like(edge_ij),
    )
    triplet_modulation = torch.zeros_like(edge_ij)
    features = GeometryFeatures(
        edge_basis=torch.empty((edge_ij.shape[0], 0)),
        edge_cutoff=torch.ones((edge_ij.shape[0], 1)),
        angle_radial_basis=torch.empty((edge_ij.shape[0], 0)),
        edge_features=edge_ij,
        edge_modulation=edge_modulation,
        triplet_modulation=triplet_modulation,
    )

    atom_out, edge_out = block(
        atom_fea,
        edge_ij,
        graph,
        features,
    )

    assert torch.equal(block.three_body.seen_edge, torch.zeros_like(edge_ij))
    assert torch.equal(block.edge_update.seen_edge, torch.full_like(edge_ij, 2.0))
    assert torch.equal(block.atom_update.seen_edge, torch.full_like(edge_ij, 3.0))
    assert torch.equal(edge_out, torch.full_like(edge_ij, 3.0))
    assert torch.equal(atom_out, atom_fea)


class _ConstantPairDelta(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.seen_edge = None

    def forward(self, atom_fea, edge_ij, graph, edge_modulation):
        self.seen_edge = edge_ij.detach().clone()
        return torch.full_like(edge_ij, self.value)


class _ConstantTripletDelta(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.seen_edge = None

    def forward(self, atom_fea, edge_ij, graph, triplet_modulation):
        self.seen_edge = edge_ij.detach().clone()
        return torch.full_like(edge_ij, self.value)


class _RecordingAtomDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_edge = None

    def forward(self, atom_fea, edge_ij, edge_modulation, graph):
        self.seen_edge = edge_ij.detach().clone()
        return torch.zeros_like(atom_fea)


class _ConstantAtomDelta(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, atom_fea, edge_ij, edge_modulation, graph, *args):
        return torch.full_like(atom_fea, self.value)


class _ConstantFeature(nn.Module):
    def __init__(self, output_dim, value):
        super().__init__()
        self.output_dim = int(output_dim)
        self.value = float(value)

    def forward(self, x):
        return x.new_full((x.shape[0], self.output_dim), self.value)


class _RecordingConstantFeature(_ConstantFeature):
    def __init__(self, output_dim, value):
        super().__init__(output_dim, value)
        self.seen_input = None

    def forward(self, x):
        self.seen_input = x.detach().clone()
        return super().forward(x)
