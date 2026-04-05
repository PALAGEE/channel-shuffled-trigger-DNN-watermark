import argparse
import json

import torch

from utils.config import DEFAULT_CONFIG_PATH, available_datasets, get_dataset_config
from utils.data import train_test_loader
from utils.trainer import evaluate


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a saved reference model or LoRA watermark.")
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
    parser.add_argument("--checkpoint", type=str, default=None)
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

    trainset, testset, trainloader, testloader = train_test_loader(run_config)
    result = evaluate(
        trainloader,
        testloader,
        trainset,
        testset,
        run_config=run_config,
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        lora_dims=args.lora_dims,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
