import copy
import json
import os
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "defaults.json"


def load_project_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def available_datasets(config_path=None):
    return tuple(load_project_config(config_path)["datasets"].keys())


def get_dataset_config(
    dataset=None,
    config_path=None,
    dataset_path=None,
    trained_path=None,
    batch_size=None,
    num_mixes=None,
    num_white_points=None,
    verify_pairs=None,
    key=None,
):
    project_config = load_project_config(config_path)
    dataset_name = dataset or project_config["project"]["default_dataset"]
    dataset_table = project_config["datasets"]

    if dataset_name not in dataset_table:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_config = copy.deepcopy(dataset_table[dataset_name])
    training_config = copy.deepcopy(project_config["training"])
    training_config.update(dataset_config.get("training", {}))
    nas_config = copy.deepcopy(project_config["nas"])
    nas_config.update(dataset_config.get("nas", {}))

    run_config = {
        "paper_title": project_config["project"]["paper_title"],
        "seed": project_config["project"].get("seed", 0),
        "dataset": dataset_name,
        "data_subdir": dataset_config.get("data_subdir"),
        "image_size": dataset_config["image_size"],
        "model_name": dataset_config["model_name"],
        "num_classes": dataset_config["num_classes"],
        "default_lora_dims": list(dataset_config["default_lora_dims"]),
        "dataset_path": dataset_path if dataset_path is not None else project_config["paths"]["dataset_path"],
        "trained_path": trained_path if trained_path is not None else project_config["paths"]["trained_path"],
        "batch_size": batch_size if batch_size is not None else training_config["batch_size"],
        "num_mixes": num_mixes if num_mixes is not None else project_config["watermark"]["num_mixes"],
        "num_white_points": (
            num_white_points if num_white_points is not None else project_config["watermark"]["num_white_points"]
        ),
        "verify_pairs": verify_pairs if verify_pairs is not None else project_config["evaluation"]["verify_pairs"],
        "key": list(key) if key is not None else list(project_config["watermark"]["key"]),
        "training": training_config,
        "nas": nas_config,
    }

    run_config["ref_path"] = os.path.join(
        run_config["trained_path"],
        dataset_name,
        "ref",
        f"{dataset_name}_{run_config['model_name']}_ref.pt",
    )
    run_config["lora_path"] = os.path.join(
        run_config["trained_path"],
        dataset_name,
        "lora",
        f"{dataset_name}_{run_config['model_name']}_lora.pt",
    )
    run_config["nas_history_path"] = os.path.join(
        run_config["trained_path"],
        dataset_name,
        "nas",
        f"{dataset_name.lower()}_lora_search_history.csv",
    )
    return run_config
