#!/usr/bin/env python3
"""Fresh-only archived-procedure reconstruction; v2/v3 entrypoints are untouched."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.planner_revision import (
    load_original_training, save_rolling_checkpoints, scaler_step_succeeded, sha256_file,
)
from src.training.planner_fresh_lineage import build_fresh_identity, validate_fresh_start

SPLIT_SHA = '833df6f0749fbe8da77ef7a32c936fbdf8a72005f9418cc69c7a96dbb06ec09b'
ROOT = Path('/cephfs/home/s2022810061/Research/KCSE2026_revision_models')
CONFIG = {
    'seed': 42, 'hidden': 1024, 'nhead': 16, 'num_layers': 24, 'batch_size': 32,
    'seq_len': 32, 'max_nodes': 1024, 'epochs': 80, 'patience': 8, 'num_workers': 8,
    'validation_rng_policy': 'archived_global_stochastic_no_reseed_no_fork',
    'validation_noise_seed': None, 'checkpoint_selection': 'strict_min_stochastic_validation_loss',
    'optimizer': {'name': 'AdamW', 'lr': 2e-4, 'eps': 1e-8, 'weight_decay': 0.01},
    'scheduler': {'name': 'CosineAnnealingWarmRestarts', 'T_0': 10, 'T_mult': 2},
}


def build_components(archived, dataset, config, device):
    """Called after seed42 and dataset construction, just as in archived main."""
    nval = max(1, int(len(dataset) * .1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - nval, nval])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=device.type == 'cuda')
    # Model initialization is on CPU before Trainer moves it; no extra RNG calls.
    model = archived['EnhancedTransformerUNet'](
        {'op': len(archived['OPS']), 'node_idx': config['max_nodes'],
         'node_type': dataset.num_types}, hidden=config['hidden'],
        nhead=config['nhead'], num_layers=config['num_layers'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, eps=1e-8, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    trainer = archived['Trainer'](model, device)
    return trainer, optimizer, scheduler, train_loader, val_loader


def run_epoch(trainer, optimizer, scheduler, train_loader, val_loader, device,
              *, max_train_batches=None, max_validation_batches=None):
    """Archived step/eval_loss, mean-of-batches, and train→scheduler→eval ordering."""
    trainer.model.train()
    total, count, successful, skipped = 0.0, 0, 0, 0
    for ops, nodes, types in islice(train_loader, max_train_batches):
        scale_before = float(trainer.scaler.get_scale())
        stats = trainer.step(ops.to(device), nodes.to(device), types.to(device), optimizer)
        if stats is None or not math.isfinite(stats['loss']):
            raise FloatingPointError('training batch returned no finite loss; run incomplete')
        total += stats['loss']; count += 1
        if scaler_step_succeeded(scale_before, float(trainer.scaler.get_scale())):
            successful += 1
        else:
            skipped += 1
    if not count:
        raise RuntimeError('empty training epoch')
    scheduler.step()
    trainer.model.eval()
    validation_total, validation_count = 0.0, 0
    # Deliberately NO seed reset and NO fork_rng: eval consumes the global stream.
    for ops, nodes, types in islice(val_loader, max_validation_batches):
        stats = trainer.eval_loss(ops.to(device), nodes.to(device), types.to(device))
        if not math.isfinite(stats['loss']):
            raise FloatingPointError('non-finite validation loss; run incomplete')
        validation_total += stats['loss']; validation_count += 1
    if not validation_count:
        raise RuntimeError('empty validation epoch')
    return {'train_loss': total / count, 'validation_loss': validation_total / validation_count,
            'train_batches': count, 'validation_batches': validation_count,
            'successful_steps': successful, 'overflow_skips': skipped,
            'lr': scheduler.get_last_lr()[0]}


def update_selection(state, validation_loss):
    if not math.isfinite(validation_loss):
        raise FloatingPointError('non-finite checkpoint selection loss')
    improved = validation_loss < state['best_val_loss']
    state['epoch'] += 1
    if improved:
        state.update(best_val_loss=validation_loss, best_epoch=state['epoch'], patience_counter=0)
    else:
        state['patience_counter'] += 1
    return improved


def verify_inputs(bundle):
    split = bundle / 'assets/random_split.json'
    if sha256_file(split) != SPLIT_SHA:
        raise ValueError('archived split manifest SHA mismatch')
    manifest = json.loads(split.read_text())
    paths = {'trainer': bundle/'archive/1.train.py',
             'data': bundle/'archive/Results/train_data_llm_synthetic.json',
             'type_map': bundle/'archive/Results/type_map.json'}
    for name, path in paths.items():
        if sha256_file(path) != manifest['input_sha256'][name]:
            raise ValueError(f'archived {name} SHA mismatch')
    if (manifest['algorithm'] != 'archived_torch_random_split' or manifest['seed'] != 42
            or manifest['record_count'] != 20000):
        raise ValueError('unexpected split procedure')
    return manifest, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--mode', choices=['primary', 'smoke'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('a real Slurm allocation is required')
    expected_bundle = os.environ.get('KCSE_BUNDLE_SHA256')
    if not expected_bundle or sha256_file(args.bundle/'SHA256SUMS.txt') != expected_bundle:
        raise SystemExit('job must pin the source/data bundle manifest')
    manifest, paths = verify_inputs(args.bundle)
    archived = load_original_training(paths['trainer'])
    primary = args.mode == 'primary'
    device = torch.device('cuda' if primary else 'cpu')
    if primary and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    config = dict(CONFIG)
    if not primary:
        config.update(hidden=32, nhead=4, num_layers=1, batch_size=2, epochs=2, num_workers=0)
    identity = build_fresh_identity({
        'run_role': 'primary_v4_archived_rng' if primary else 'diagnostic_v4_archived_rng_smoke',
        'config': config, 'bundle_manifest_sha256': expected_bundle,
        'manifest_sha256': SPLIT_SHA, 'input_sha256': manifest['input_sha256'],
        'trainer_sha256': sha256_file(Path(__file__)),
        'execution': {'device': str(device), 'max_train_batches': None if primary else 2,
                      'max_validation_batches': None if primary else 1,
                      'image_id': os.environ.get('KCSE_IMAGE_ID'),
                      'image_archive_sha256': os.environ.get('KCSE_IMAGE_ARCHIVE_SHA256')},
    })
    if not identity['execution']['image_id'] or not identity['execution']['image_archive_sha256']:
        raise SystemExit('runtime image provenance missing')
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = ROOT/('planner_archived_rng_primary_v4' if primary else 'planner_archived_rng_v4_smoke')/run_id
    output.mkdir(parents=True, exist_ok=False)
    (output/'provenance.json').write_text(json.dumps(identity, indent=2)+'\n')
    (output/'execution.json').write_text(json.dumps({'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'bundle': str(args.bundle), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(0) if primary else 'cpu'}, indent=2)+'\n')
    print(json.dumps({'run_directory': str(output), 'run_id': run_id}), flush=True)
    # Match archived main: only Torch is used for dataset/model/noise/shuffle.
    torch.manual_seed(42)
    dataset = archived['EditDataset'](paths['data'], seq_len=32, max_nodes=1024)
    if len(dataset) != 20000 or dataset.type_map != manifest['type_map']:
        raise ValueError('dataset dimensions/type map mismatch')
    trainer, optimizer, scheduler, train_loader, val_loader = build_components(archived, dataset, config, device)
    if (list(train_loader.dataset.indices) != manifest['train_indices'] or
            list(val_loader.dataset.indices) != manifest['validation_indices']):
        raise ValueError('runtime random split differs from archived reference')
    state = {'epoch': 0, 'best_epoch': None, 'best_val_loss': float('inf'), 'patience_counter': 0,
             'global_step': 0, 'attempted_batches': 0, 'overflow_skips': 0}
    validate_fresh_start(identity, state)
    with (output/'metrics.jsonl').open('x') as stream:
        for _ in range(config['epochs']):
            row = run_epoch(trainer, optimizer, scheduler, train_loader, val_loader, device,
                            max_train_batches=None if primary else 2,
                            max_validation_batches=None if primary else 1)
            improved = update_selection(state, row['validation_loss'])
            state['global_step'] += row['successful_steps']
            state['attempted_batches'] += row['train_batches']
            state['overflow_skips'] += row['overflow_skips']
            row.update(epoch=state['epoch'], improved=improved, global_step=state['global_step'],
                       attempted_batches=state['attempted_batches'], cumulative_overflow_skips=state['overflow_skips'],
                       best_epoch=state['best_epoch'], patience_counter=state['patience_counter'])
            state['last_epoch_metric'] = row
            save_rolling_checkpoints(output/'best_model_state_dict.pt', output/'latest_resume.pt',
                                     trainer.model, optimizer, scheduler, trainer.scaler, state,
                                     identity, save_best=improved)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(json.dumps(row), flush=True)
            if state['patience_counter'] >= config['patience']:
                break
    if state['global_step'] <= 0 or state['best_epoch'] is None:
        raise RuntimeError('no optimizer updates or valid best checkpoint')
    complete = {'status': 'training_completed_not_end_to_end_paper_certification',
                'termination': 'patience' if state['patience_counter'] >= config['patience'] else 'max_epochs',
                'epoch': state['epoch'], 'best_epoch': state['best_epoch'],
                'best_val_loss': state['best_val_loss'], 'global_step': state['global_step'],
                'sha256': {name: sha256_file(output/name) for name in
                    ['best_model_state_dict.pt','latest_resume.pt','best_model_metadata.json','metrics.jsonl','provenance.json']}}
    (output/'completion.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
