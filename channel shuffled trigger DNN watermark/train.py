import argparse
import json
import os

import torch

from utils.config import DEFAULT_CONFIG_PATH, available_datasets, get_dataset_config
from utils.data import train_test_loader
from utils.trainer import train


def build_parser():
    parser = argparse.ArgumentParser(description="Train the reference model or LoRA watermark.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", choices=available_datasets(), default=None)
    parser.add_argument("--mode", choices=("ref", "lora"), required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--trained-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-mixes", type=int, default=None)
    parser.add_argument("--num-white-points", type=int, default=None)
    parser.add_argument("--verify-pairs", type=int, default=None)
    parser.add_argument("--key", nargs="+", type=int, default=None)
    parser.add_argument("--ref-path", type=str, default=None)
    parser.add_argument("--lora-dims", nargs="+", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_config = get_dataset_config(
        dataset=args.dataset,
        config_path=args.config,
        dataset_path=args.dataset_path,
        trained_path=args.trained_path,
        batch_size=args.batch_size,
        num_mixes=args.num_mixes,
        num_white_points=args.num_white_points,
        verify_pairs=args.verify_pairs,
        key=args.key,
    )
    torch.manual_seed(run_config["seed"])

    os.makedirs(run_config["dataset_path"], exist_ok=True)
    os.makedirs(run_config["trained_path"], exist_ok=True)

    trainset, testset, trainloader, testloader = train_test_loader(run_config)
    result = train(
        trainloader,
        testloader,
        trainset,
        testset,
        run_config=run_config,
        mode=args.mode,
        ref_path=args.ref_path,
        lora_dims=args.lora_dims,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
