import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.builder import build_model
from utils.config import DEFAULT_CONFIG_PATH, available_datasets, get_dataset_config
from utils.lora import replace_modules_with_lora


def find_normalization_layers(module_root):
    layers = []
    for module in module_root.modules():
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
                nn.LayerNorm,
            ),
        ):
            layers.append(module)
    return layers


def evaluate_lora_dims(model, lora_dims, batch_size, input_image_size, gamma, init_std, max_parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        lora_model = replace_modules_with_lora(model, lora_dims, init_std).to(device)
        lora_model.train()

        input_tensor = torch.randn(batch_size, 3, input_image_size, input_image_size, device=device)
        noise_tensor = torch.randn(batch_size, 3, input_image_size, input_image_size, device=device)
        mixed_input = input_tensor + gamma * noise_tensor

        feature, _ = lora_model(input_tensor)
        mixed_feature, _ = lora_model(mixed_input)
        reduce_dims = tuple(range(1, feature.ndim))
        zen_score = torch.abs(feature - mixed_feature).sum(dim=reduce_dims).mean().clamp_min(1e-12)

        log_bn_scaling_factor = 0.0
        for normalization_layer in find_normalization_layers(lora_model):
            running_var = getattr(normalization_layer, "running_var", None)
            if running_var is not None:
                bn_scaling_factor = torch.sqrt(torch.mean(running_var)).clamp_min(1e-12)
                log_bn_scaling_factor += torch.log(bn_scaling_factor)

        zen_score = torch.log(zen_score) + log_bn_scaling_factor
        total_params = sum(
            parameter.numel()
            for name, parameter in lora_model.named_parameters()
            if "lora_" in name
        )

        if max_parameters > 0:
            penalty = (max_parameters - total_params) / max_parameters
            zen_score = 0.9 * zen_score + 0.1 * zen_score * penalty

        score = float(zen_score.item())

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score, total_params


def uniform_crossover(parent1, parent2, cross_rate):
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < cross_rate:
            if gene1 == gene2:
                child.append(gene1)
            elif gene1 < gene2:
                child.append(gene1 + 1)
            else:
                child.append(gene1 - 1)
        else:
            child.append(gene1)
    return child


def summarize_generation(population, fitness_scores, parameter_counts):
    best_index = max(range(len(population)), key=lambda idx: fitness_scores[idx])
    return {
        "best_fitness": fitness_scores[best_index],
        "average_fitness": sum(fitness_scores) / len(fitness_scores),
        "min_fitness": min(fitness_scores),
        "best_parameter_size": parameter_counts[best_index],
        "best_individual": population[best_index],
    }


def record_generation(csv_writer, csv_file, generation_idx, population, fitness_scores, parameter_counts):
    summary = summarize_generation(population, fitness_scores, parameter_counts)
    csv_writer.writerow(
        {
            "generation": generation_idx,
            "best_fitness": summary["best_fitness"],
            "average_fitness": summary["average_fitness"],
            "min_fitness": summary["min_fitness"],
            "best_individual_parameter_size": summary["best_parameter_size"],
            "best_individual": summary["best_individual"],
        }
    )
    csv_file.flush()

    print(
        f"Generation {generation_idx} "
        f"best_fitness={summary['best_fitness']:.6f} "
        f"average_fitness={summary['average_fitness']:.6f} "
        f"min_fitness={summary['min_fitness']:.6f} "
        f"best_params={summary['best_parameter_size']} "
        f"best_individual={summary['best_individual']}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Genetic search for dataset-specific LoRA ranks.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", choices=available_datasets(), default=None)
    parser.add_argument("--trained-path", type=str, default=None)
    parser.add_argument("--ref-path", type=str, default=None)
    parser.add_argument("--history-table-path", type=str, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--choices", nargs="+", type=int, default=None)
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--init-std", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _save_summary(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main():
    args = build_parser().parse_args()
    run_config = get_dataset_config(
        dataset=args.dataset,
        config_path=args.config,
        trained_path=args.trained_path,
    )
    nas_cfg = dict(run_config["nas"])
    if args.num_generations is not None:
        nas_cfg["num_generations"] = args.num_generations
    if args.batch_size is not None:
        nas_cfg["batch_size"] = args.batch_size
    if args.population_size is not None:
        nas_cfg["population_size"] = args.population_size
    if args.gamma is not None:
        nas_cfg["gamma"] = args.gamma
    if args.choices is not None:
        nas_cfg["choices"] = args.choices
    if args.mutation_probability is not None:
        nas_cfg["mutation_probability"] = args.mutation_probability
    if args.init_std is not None:
        nas_cfg["init_std"] = args.init_std
    if args.seed is not None:
        nas_cfg["seed"] = args.seed

    random.seed(nas_cfg["seed"])
    torch.manual_seed(nas_cfg["seed"])

    reference_path = args.ref_path or run_config["ref_path"]
    history_table_path = args.history_table_path or run_config["nas_history_path"]
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference checkpoint not found: {reference_path}")

    model = build_model(run_config["dataset"])
    model.load_state_dict(torch.load(reference_path, map_location="cpu"), strict=True)

    num_layers = len(run_config["default_lora_dims"])
    max_choice = max(nas_cfg["choices"])
    _, max_parameters = evaluate_lora_dims(
        model,
        [max_choice] * num_layers,
        nas_cfg["batch_size"],
        run_config["image_size"],
        nas_cfg["gamma"],
        nas_cfg["init_std"],
        max_parameters=0,
    )

    population = [
        [random.choice(nas_cfg["choices"]) for _ in range(num_layers)]
        for _ in range(nas_cfg["population_size"])
    ]
    fitness_scores = []
    parameter_counts = []

    for individual in population:
        fitness_score, parameter_count = evaluate_lora_dims(
            model,
            individual,
            nas_cfg["batch_size"],
            run_config["image_size"],
            nas_cfg["gamma"],
            nas_cfg["init_std"],
            max_parameters,
        )
        fitness_scores.append(fitness_score)
        parameter_counts.append(parameter_count)

    history_dir = os.path.dirname(history_table_path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)
    with open(history_table_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "generation",
                "best_fitness",
                "average_fitness",
                "min_fitness",
                "best_individual_parameter_size",
                "best_individual",
            ],
        )
        csv_writer.writeheader()
        record_generation(csv_writer, csv_file, 0, population, fitness_scores, parameter_counts)

        for generation_idx in range(1, nas_cfg["num_generations"] + 1):
            sorted_indices = sorted(
                range(len(population)),
                key=lambda idx: fitness_scores[idx],
                reverse=True,
            )
            selected_parent_indices = sorted_indices[: max(2, nas_cfg["population_size"] // 2)]
            parents = [population[idx] for idx in selected_parent_indices]
            parent_scores = [fitness_scores[idx] for idx in selected_parent_indices]
            parent_parameter_counts = [parameter_counts[idx] for idx in selected_parent_indices]

            parent_fitness_sum = sum(parent_scores)
            if parent_fitness_sum > 0:
                parent_weights = [score / parent_fitness_sum for score in parent_scores]
            else:
                parent_weights = None

            offspring = []
            for _ in range(nas_cfg["population_size"] - len(parents)):
                parent1 = random.choices(parents, weights=parent_weights, k=1)[0]
                parent2 = random.choices(parents, weights=parent_weights, k=1)[0]
                child = uniform_crossover(parent1, parent2, cross_rate=0.5)

                if random.random() < nas_cfg["mutation_probability"]:
                    mutation_point = random.randrange(num_layers)
                    child[mutation_point] = random.choice(nas_cfg["choices"])

                offspring.append(child)

            offspring_scores = []
            offspring_parameter_counts = []
            for child in offspring:
                fitness_score, parameter_count = evaluate_lora_dims(
                    model,
                    child,
                    nas_cfg["batch_size"],
                    run_config["image_size"],
                    nas_cfg["gamma"],
                    nas_cfg["init_std"],
                    max_parameters,
                )
                offspring_scores.append(fitness_score)
                offspring_parameter_counts.append(parameter_count)

            population = parents + offspring
            fitness_scores = parent_scores + offspring_scores
            parameter_counts = parent_parameter_counts + offspring_parameter_counts
            record_generation(csv_writer, csv_file, generation_idx, population, fitness_scores, parameter_counts)

    summary = summarize_generation(population, fitness_scores, parameter_counts)
    summary["dataset"] = run_config["dataset"]
    summary["history_table_path"] = history_table_path
    summary["paper_title"] = run_config["paper_title"]
    summary_path = os.path.splitext(history_table_path)[0] + "_summary.json"
    _save_summary(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
