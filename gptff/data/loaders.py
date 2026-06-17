from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from gptff.data.dataset import (
    StructureDataset,
    collate_graph_samples,
    validate_dataframe_schema,
)
from gptff.model.readout import fit_element_refs_from_samples


def read_data(config: TrainingConfig) -> pd.DataFrame:
    return pd.read_csv(Path(config.data_path) / config.data_file)


def build_datasets(config: TrainingConfig) -> Tuple[StructureDataset, StructureDataset]:
    df = read_data(config)
    if "fold" not in df.columns:
        raise ValueError("Dataframe must contain a 'fold' column.")
    label_config = config.to_label_config()
    validate_dataframe_schema(
        df,
        label_config,
        require_energy=True,
        require_forces=True,
        require_stress=config.w3 > 0.0,
    )
    df_train = df.loc[df["fold"] != config.val_fold].reset_index(drop=True)
    df_val = df.loc[df["fold"] == config.val_fold].reset_index(drop=True)
    train_dataset = StructureDataset(
        df_train,
        r_cut=config.radial_cutoff,
        a_cut=config.angle_cutoff,
        label_config=label_config,
        cache_graphs=config.cache_graphs,
        cache_size=config.graph_cache_size,
    )
    val_dataset = StructureDataset(
        df_val,
        r_cut=config.radial_cutoff,
        a_cut=config.angle_cutoff,
        label_config=label_config,
        cache_graphs=config.cache_graphs,
        cache_size=config.graph_cache_size,
    )
    return train_dataset, val_dataset


def apply_fitted_element_refs(config: TrainingConfig, train_dataset) -> None:
    if not config.fit_element_refs:
        return
    if config.element_refs is not None:
        raise ValueError("Set either element_refs or fit_element_refs, not both.")
    print("Fitting element_refs from the training dataset.")
    config.element_refs = fit_element_refs_from_samples(
        train_dataset,
        max_atomic_number=config.max_atomic_number,
        ridge=config.element_ref_ridge,
    )


def build_loaders(
    config: TrainingConfig,
    train_dataset,
    val_dataset,
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = torch.device(config.device).type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_graph_samples,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
