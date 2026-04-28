#!/usr/bin/env python3
"""Train a single GPT-2 IOI model for a given seed. Used by launch_gpt2_parallel.sh."""
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Now import torch (after setting CUDA_VISIBLE_DEVICES)
import torch
from gpt2_ioi_circuit_stability import *

model_path = MODEL_DIR / f'model_seed{args.seed}_final.pt'
if model_path.exists():
    print(f'Seed {args.seed}: cached at {model_path}, skipping')
else:
    train_data, tokenizer = load_training_data()
    model = train_model(train_data, args.seed, tokenizer)
    print(f'Seed {args.seed}: complete, saved to {model_path}')
