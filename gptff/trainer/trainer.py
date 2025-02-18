import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from pymatgen.core.structure import Structure
import ast
from tqdm import tqdm
from joblib import Parallel, delayed

from gptff.utils_.data import Mydataset, collate_fn, CosineAnnealingWarmupRestarts
from gptff.model.model import tModLodaer, tModLodaer_t
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import json
import gc

parser = argparse.ArgumentParser(description='Graph-based Pretrained Transformer Force Field.')
parser.add_argument('config', metavar='OPTIONS',
                    help='Configs for training')

args = parser.parse_args(sys.argv[1:])
config_file = args.config

with open(config_file, 'r') as fp:
    js = json.load(fp)

# config for training

class CFG:
    val_fold = js['training']['val_fold']
    num_train_steps = int(js['training']['epochs'])
    warmup_steps = int(js['training']['warmup_steps'])
    batch_size = int(js['training']['batch_size'])
    device = js['training']['device']
    data_path = js['data']['data_path']
    data_file = js['data']['data_file']
    num_workers = js['training']['workers']
    lr = js['training']['learning_rate']
    weight_decay = js['training']['weight_decay']
    epochs = js['training']['epochs']
    start_epoch = js['training']['start_epoch']
    w1 = js['training']['weight_energy']
    w2 = js['training']['weight_force']
    w3 = js['training']['weight_stress']
    transformer_activate = js['training']['transformer_activate']
    node_feature_len = js['training']['node_feature_len']
    edge_feature_len = js['training']['edge_feature_len']
    n_layers = js['training']['n_layers']
    unit_trans = 160.21766208

cfg_args = {k: v for k, v in CFG.__dict__.items() if not k.startswith("__") and k not in {"split", "config"}}

# Read data

df = pd.read_csv(os.path.join(CFG.data_path, CFG.data_file))

df_trn = df.loc[df['fold'] != CFG.val_fold].reset_index(drop=True)
df_val = df.loc[df['fold'] == CFG.val_fold].reset_index(drop=True)

trn_dataset = Mydataset(df_trn)
val_dataset = Mydataset(df_val)

train_loader = DataLoader(trn_dataset, batch_size=CFG.batch_size,
                              num_workers=CFG.num_workers,
                              shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size,
                        shuffle=False,
                        num_workers=CFG.num_workers,
                        collate_fn=collate_fn,
                        pin_memory=True)


# build model

if CFG.transformer_activate:
    model = tModLodaer_t(CFG)
else:
    model = tModLodaer(CFG)

# if CFG.device == 'cuda':
model.to(CFG.device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_parameters = count_parameters(model)
print(f'Number of Model parameters: {num_parameters}')

criterion = nn.HuberLoss()

optimizer = optim.AdamW(model.parameters(), CFG.lr,
                               weight_decay= CFG.weight_decay)

scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                          first_cycle_steps=CFG.num_train_steps, 
                                          cycle_mult=1, 
                                          max_lr=CFG.lr, 
                                          min_lr=5e-6, 
                                          warmup_steps=CFG.warmup_steps, 
                                          gamma=1.)

scheduler.step(CFG.start_epoch)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):

    return torch.mean(torch.abs(target - prediction))

def train(train_loader, model, criterion, optimizer, pbar_trn):
    scaler = GradScaler()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mae_errors = AverageMeter()
    force_errors = AverageMeter()
    stress_errors = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()

    for i, (data) in enumerate(train_loader):

        data = [x.to(CFG.device) for x in data]

        pbar_trn.update(1)

        data_time.update(time.time() - end)

        atom_fea, coords, offsets, lattice, n_atoms, pairs_count, nbr_atoms, bond_pairs_indices, n_bond_pairs_bond, target_energy, target_forces, target_stress, ref_energy = data
        coords = coords.requires_grad_(True)
        strain = torch.zeros_like(lattice, dtype=torch.float32).requires_grad_(True)
    
        lattices = torch.matmul(lattice, torch.eye(3, dtype=torch.float32)[None, :, :].to(CFG.device) + strain)

        try:
            volumes = torch.linalg.det(lattices)
        except:
            continue

        strains = torch.repeat_interleave(strain, n_atoms, dim=0)
        coords = torch.matmul(coords.unsqueeze(1), torch.eye(3, dtype=torch.float32)[None, :, :].to(CFG.device) + strains).squeeze()

        lattices = lattices[torch.repeat_interleave(torch.arange(pairs_count.shape[0]).to(CFG.device), pairs_count)]
        
        offset_dist = torch.matmul(offsets[:, None, :], lattices).squeeze()

        vec_diff_ij = (
                coords[nbr_atoms[:, 1], :]
                + offset_dist
                - coords[nbr_atoms[:, 0], :]
            )

        pair_vec_ij = vec_diff_ij
        pair_dist_ij = torch.sqrt(torch.matmul(pair_vec_ij[:, None, :], pair_vec_ij[:, :, None])).squeeze()
        triple_vec_ij = pair_vec_ij[bond_pairs_indices[:, 0]].squeeze()
        triple_vec_ik = pair_vec_ij[bond_pairs_indices[:, 1]].squeeze()
        triple_dist_ij = pair_dist_ij[bond_pairs_indices[:, 0]].squeeze()
        triple_dist_ik = pair_dist_ij[bond_pairs_indices[:, 1]].squeeze()
        triple_a_jik = torch.matmul(triple_vec_ij[:, None, :], triple_vec_ik[:, :, None]).squeeze(-1) / (triple_dist_ij[:, None] * triple_dist_ik[:, None])
        triple_a_jik = torch.clamp(triple_a_jik, -1.0, 1.0) * (1 - 1e-6)
        triple_a_jik = triple_a_jik.squeeze()
        
        # compute output

        with autocast():

            ener_pred = model(atom_fea, pair_dist_ij, n_atoms, triple_dist_ij, triple_dist_ik, triple_a_jik, nbr_atoms, n_bond_pairs_bond, bond_pairs_indices)
            ener_pred = ener_pred.squeeze() + ref_energy
            force_pred, stress_pred = torch.autograd.grad(ener_pred, [coords, strain], torch.ones_like(ener_pred), retain_graph=True, create_graph=True)
            force_pred = -1.0 * force_pred
            stress_pred = 1. / volumes[:, None, None] * stress_pred * CFG.unit_trans

            ener_pred = ener_pred.squeeze() / n_atoms
            target_energy = target_energy.squeeze() / n_atoms
            
            e_loss = criterion(ener_pred, target_energy)
            f_loss = criterion(force_pred.view(-1), target_forces.view(-1))
            s_loss = criterion(stress_pred.view(-1), target_stress.view(-1))

            loss = CFG.w1 * e_loss + CFG.w2 * f_loss + CFG.w3 * s_loss
        
        # bad case
        if loss.detach().item() > 10:
            continue

        force_sum = torch.sum(force_pred)
        ener_sum = torch.sum(ener_pred)
        stress_sum = torch.sum(stress_pred)

        if force_sum != force_sum or ener_sum != ener_sum or loss != loss or stress_sum != stress_sum:
            continue

        mae_error = mae(ener_pred.detach().cpu(), target_energy.detach().cpu()) 
        losses.update(loss.detach().cpu(), target_energy.size(0))
        mae_errors.update(mae_error, target_energy.size(0))
        force_error = mae(force_pred.detach().cpu().reshape(-1), target_forces.cpu().reshape(-1))
        force_errors.update(force_error, target_forces.size(0))
        stress_error = mae(stress_pred.detach().cpu().reshape(-1), target_stress.cpu().reshape(-1))
        stress_errors.update(stress_error, target_stress.size(0) * 9)
    
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # prevent gradient vanishing/explosion

        scaler.step(optimizer)
        scaler.update() 

        batch_time.update(time.time() - end)
        end = time.time()

        pbar_trn.set_description(f'[{i+1}/{len(train_loader)}]')
        pbar_trn.set_postfix({'loss': f'{losses.val:5f} ({losses.avg:.5f})',
                              'MAE(e)': f'{mae_errors.val:.5f} ({mae_errors.avg:.5f})',
                              'MAE(f)': f'{force_errors.val:.5f} ({force_errors.avg:.5f})',
                              'MAE(s)': f'{stress_errors.val:.3f} ({stress_errors.avg:.3f})'
                              })
        sys.stdout.flush()
    torch.cuda.empty_cache()
    gc.collect()

def validate(val_loader, model, criterion, pbar_trn, pbar_val):

    batch_time = AverageMeter()
    losses = AverageMeter()

    mae_errors = AverageMeter()
    force_errors = AverageMeter()
    stress_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    forces_gt, energy_gt, stress_gt = [], [], []
    forces_pred, energy_pred, stress_preds = [], [], []

    for i, (data) in enumerate(val_loader):

        data = [x.to(CFG.device) for x in data]
        pbar_val.update(1)

        atom_fea, coords, offsets, lattice, n_atoms, pairs_count, nbr_atoms, bond_pairs_indices, n_bond_pairs_bond, target_energy, target_forces, target_stress, ref_energy = data
        coords = coords.requires_grad_(True)
        strain = torch.zeros_like(lattice, dtype=torch.float32).requires_grad_(True)

        lattices = torch.matmul(lattice, torch.eye(3, dtype=torch.float32)[None, :, :].to(CFG.device) + strain)

        try:
            volumes = torch.linalg.det(lattices)
        except:
            continue
        
        strains = torch.repeat_interleave(strain, n_atoms, dim=0)
        coords = torch.matmul(coords.unsqueeze(1), torch.eye(3, dtype=torch.float32)[None, :, :].to(CFG.device) + strains).squeeze()

        lattices = lattices[torch.repeat_interleave(torch.arange(pairs_count.shape[0]).to(CFG.device), pairs_count)]

        with autocast():
            
            offset_dist = torch.matmul(offsets[:, None, :], lattices).squeeze()
            # offset_dist = torch.matmul(lattices, offsets[:, :, None]).squeeze()

            vec_diff_ij = (
                    coords[nbr_atoms[:, 1], :]
                    + offset_dist
                    - coords[nbr_atoms[:, 0], :]
                )

            pair_vec_ij = vec_diff_ij
            pair_dist_ij = torch.sqrt(torch.matmul(pair_vec_ij[:, None, :], pair_vec_ij[:, :, None])).squeeze()
            triple_vec_ij = pair_vec_ij[bond_pairs_indices[:, 0]].squeeze()
            triple_vec_ik = pair_vec_ij[bond_pairs_indices[:, 1]].squeeze()
            triple_dist_ij = pair_dist_ij[bond_pairs_indices[:, 0]].squeeze()
            triple_dist_ik = pair_dist_ij[bond_pairs_indices[:, 1]].squeeze()
            triple_a_jik = torch.matmul(triple_vec_ij[:, None, :], triple_vec_ik[:, :, None]).squeeze(-1) / (triple_dist_ij[:, None] * triple_dist_ik[:, None])
            triple_a_jik = torch.clamp(triple_a_jik, -1., 1.) * (1 - 1e-6)
            triple_a_jik = triple_a_jik.squeeze()
            

            ener_pred = model(atom_fea, pair_dist_ij, n_atoms, triple_dist_ij, triple_dist_ik, triple_a_jik, nbr_atoms, n_bond_pairs_bond, bond_pairs_indices)
            
            ener_pred = ener_pred.squeeze() + ref_energy

            force_pred, stress_pred = torch.autograd.grad(ener_pred, [coords, strain], torch.ones_like(ener_pred), retain_graph=True, create_graph=True)
            force_pred = -1.0 * force_pred
            stress_pred = 1. / volumes[:, None, None] * stress_pred * 160.21766208

            ener_pred = ener_pred.squeeze() / n_atoms
            target_energy = target_energy.squeeze() / n_atoms

            e_loss = criterion(ener_pred.view(-1), target_energy.view(-1))
            f_loss = criterion(force_pred.view(-1), target_forces.view(-1))
            s_loss = criterion(stress_pred.view(-1), target_stress.view(-1))

            loss = CFG.w1 * e_loss + CFG.w2 * f_loss + CFG.w3 * s_loss

        force_sum = torch.sum(force_pred)
        ener_sum = torch.sum(ener_pred)
        stress_sum = torch.sum(stress_pred)

        if force_sum != force_sum or ener_sum != ener_sum or loss != loss or stress_sum != stress_sum:
            continue
        
        if loss.detach().item() > 10:
            continue

        # measure accuracy and record loss

        mae_error = mae(ener_pred.detach().cpu(), target_energy.detach().cpu())
        losses.update(loss.detach().cpu().item(), target_energy.size(0))
        mae_errors.update(mae_error, target_energy.size(0))
        force_error = mae(force_pred.detach().cpu().reshape(-1), target_forces.detach().cpu().reshape(-1))
        force_errors.update(force_error, target_forces.size(0))
        stress_error = mae(stress_pred.detach().cpu().reshape(-1), target_stress.detach().cpu().reshape(-1))
        stress_errors.update(stress_error, target_stress.size(0) * 9)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar_val.set_description(f'[{i+1}/{len(val_loader)}]')
        pbar_val.set_postfix({'loss': f'{losses.val:.5f} ({losses.avg:.5f})',
                              'MAE(e)': f'{mae_errors.val:.5f} ({mae_errors.avg:.5f})',
                              'MAE(f)': f'{force_errors.val:.5f} ({force_errors.avg:.5f})',
                              'MAE(s)': f'{stress_errors.val:.3f} ({stress_errors.avg:.3f})'
                              })
        
    trn_text = pbar_trn.postfix

    pbar_trn.set_postfix_str(f'{trn_text}, val_loss: {losses.avg:.5f} val_MAE(e): {mae_errors.avg:.5f}, val_MAE(f): {force_errors.avg:.5f}, val_MAE(s): {stress_errors.avg:.3f}')
    pbar_trn.close()

    torch.cuda.empty_cache()
    gc.collect()
    
    # Write training history
    with open('val_history.txt', 'a+') as fp:
        fp.write(f'{mae_errors.avg:.4f} {force_errors.avg:.4f} {stress_errors.avg:.4f} \n')
        
    return mae_errors.avg, (forces_gt, forces_pred, energy_gt, energy_pred)


def main():
    best_mae_error = 1e12
    bar_format = '{l_bar}{bar:40}| [{elapsed}<{remaining}' '{postfix}]'

    for epoch in range(CFG.start_epoch, CFG.epochs):
        print(f'Epoch: [{epoch+1}/ {CFG.epochs}], lr: {scheduler.get_lr()[0]:.4e}')
        sys.stdout.flush()

        pbar_trn = tqdm(train_loader, total=len(train_loader), mininterval = 0.1, ascii="░▒█", position=0, unit='s', bar_format=bar_format)
        
        pbar_val = tqdm(val_loader, total=len(val_loader), mininterval = 0.1, ascii="░▒█", position=0, unit='s', bar_format=bar_format, leave=False)
        train(train_loader, model, criterion, optimizer, pbar_trn)

        torch.cuda.empty_cache()
        gc.collect()
        
        mae_error, results = validate(val_loader, model, criterion, pbar_trn, pbar_val)

        forces_gt, forces_pred, energy_gt, energy_pred = results

        scheduler.step()

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)

        model_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'cfg': cfg_args
            }

        torch.save(model_state, './curr_checkpoint.pth')

        if is_best:
            shutil.copyfile('curr_checkpoint.pth', 'best_checkpoint.pth')