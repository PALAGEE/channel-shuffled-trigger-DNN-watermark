import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import DEFAULT_CONFIG_PATH, get_dataset_config, load_project_config
from utils.trainer import evaluate, train


def build_parser():
    parser = argparse.ArgumentParser(description="Run a fast smoke test for train/evaluate/NAS.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=256)
    return parser


def make_synthetic_dataset(num_samples, num_classes, image_size):
    images = torch.rand(num_samples, 3, image_size, image_size)
    labels = torch.arange(num_samples) % num_classes
    return TensorDataset(images, labels)


def build_smoke_config(args):
    run_config = get_dataset_config(
        dataset=args.dataset,
        config_path=args.config,
        batch_size=args.batch_size,
        trained_path=str(ROOT / "models" / "checkpoints" / "_smoke_test"),
    )
    run_config["num_mixes"] = 1
    run_config["verify_pairs"] = 2
    run_config["training"]["ref_epochs"] = 1
    run_config["training"]["lora_epochs"] = 1
    run_config["training"]["lora_steps_per_epoch"] = 1
    run_config["training"]["ref_milestones"] = [1]
    run_config["nas"]["num_generations"] = 1
    run_config["nas"]["population_size"] = 4
    run_config["nas"]["batch_size"] = args.batch_size
    run_config["nas"]["seed"] = 0
    return run_config


def write_temp_config(run_config, base_config_path):
    project_config = load_project_config(base_config_path)
    dataset_name = run_config["dataset"]

    project_config["project"]["default_dataset"] = dataset_name
    project_config["paths"]["trained_path"] = run_config["trained_path"]
    project_config["training"]["batch_size"] = run_config["batch_size"]
    project_config["watermark"]["num_mixes"] = run_config["num_mixes"]
    project_config["watermark"]["num_white_points"] = run_config["num_white_points"]
    project_config["evaluation"]["verify_pairs"] = run_config["verify_pairs"]
    project_config["datasets"][dataset_name]["training"] = {
        "ref_epochs": run_config["training"]["ref_epochs"],
        "ref_milestones": run_config["training"]["ref_milestones"],
        "lora_epochs": run_config["training"]["lora_epochs"],
        "lora_steps_per_epoch": run_config["training"]["lora_steps_per_epoch"],
    }
    project_config["nas"].update(run_config["nas"])

    temp_config_path = Path(run_config["trained_path"]) / "smoke_config.json"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_config_path.open("w", encoding="utf-8") as file:
        json.dump(project_config, file, indent=2, ensure_ascii=False)
    return temp_config_path


def run_search_smoke(temp_config_path, run_config):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "search_lora.py"),
        "--config",
        str(temp_config_path),
        "--dataset",
        run_config["dataset"],
        "--ref-path",
        run_config["ref_path"],
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout


def main():
    args = build_parser().parse_args()
    torch.manual_seed(0)
    run_config = build_smoke_config(args)

    trainset = make_synthetic_dataset(args.train_samples, run_config["num_classes"], run_config["image_size"])
    testset = make_synthetic_dataset(args.test_samples, run_config["num_classes"], run_config["image_size"])
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    smoke_root = Path(run_config["trained_path"])
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    ref_result = train(
        trainloader,
        testloader,
        trainset,
        testset,
        run_config=run_config,
        mode="ref",
    )
    ref_eval = evaluate(
        trainloader,
        testloader,
        trainset,
        testset,
        run_config=run_config,
        mode="ref",
        checkpoint_path=ref_result["checkpoint_path"],
    )

    lora_result = train(
        trainloader,
        testloader,
        trainset,
        testset,
        run_config=run_config,
        mode="lora",
        ref_path=ref_result["checkpoint_path"],
    )
    lora_eval = evaluate(
        trainloader,
        testloader,
        trainset,
        testset,
        run_config=run_config,
        mode="lora",
        checkpoint_path=lora_result["checkpoint_path"],
    )

    temp_config_path = write_temp_config(run_config, args.config)
    search_stdout = run_search_smoke(temp_config_path, run_config)

    summary = {
        "smoke_test": "passed",
        "dataset": run_config["dataset"],
        "batch_size": args.batch_size,
        "ref_checkpoint_exists": Path(ref_result["checkpoint_path"]).exists(),
        "lora_checkpoint_exists": Path(lora_result["checkpoint_path"]).exists(),
        "nas_history_exists": Path(run_config["nas_history_path"]).exists(),
        "nas_summary_exists": Path(run_config["nas_history_path"]).with_name(
            Path(run_config["nas_history_path"]).stem + "_summary.json"
        ).exists(),
        "ref_eval": ref_eval,
        "lora_eval": lora_eval,
        "search_stdout_tail": search_stdout.strip().splitlines()[-4:],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
