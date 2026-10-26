"""Reconstruct archived split with the real dataset; never start training."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    files = {'trainer': args.archive / '1.train.py',
             'data': args.archive / 'Results/train_data_llm_synthetic.json',
             'type_map': args.archive / 'Results/type_map.json'}
    hashes = {name: digest(path) for name, path in files.items()}
    if hashes != reference['input_sha256']:
        raise ValueError('archive differs from verified reference')
    module = runpy.run_path(str(files['trainer']), run_name='split_reconstruction_only')
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    dataset = module['EditDataset'](str(files['data']), max_nodes=module['MAX_NODES'])
    unchanged_rng = torch.equal(before, torch.get_rng_state())
    val_size = max(1, int(len(dataset) * module['VAL_RATIO']))
    train, validation = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    old = reference['original_random_split_reference']
    if list(train.indices) != old['train_indices'] or list(validation.indices) != old['validation_indices']:
        raise ValueError('actual dataset split differs from reconstructed reference')
    result = {'schema_version': 1, 'algorithm': 'archived_torch_random_split', 'seed': 42,
              'claim': 'reconstructed procedure; not authenticated historical checkpoint split',
              'record_count': len(dataset), 'train_indices': list(train.indices),
              'validation_indices': list(validation.indices), 'type_map': dataset.type_map,
              'input_sha256': hashes, 'reference_manifest_sha256': digest(args.reference),
              'dataset_construction_preserved_torch_rng': unchanged_rng,
              'reference_indices_identical': True, 'torch_version': torch.__version__,
              'preserved_config': reference['preserved_config'],
              'limitation': 'random split allows duplicate train/validation examples; validation loss is not independent source-generalization evidence'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'train': len(train), 'validation': len(validation),
                      'rng_unchanged': unchanged_rng, 'reference_match': True,
                      'manifest_sha256': digest(args.output)}, indent=2))


if __name__ == '__main__':
    main()
