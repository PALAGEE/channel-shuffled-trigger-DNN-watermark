import json
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.builder import build_model
from utils.lora import replace_modules_with_lora
from utils.trigger import gen_trigger_4train, gen_trigger_4verify


def _prepare_inputs(data):
    if data.shape[1] == 1:
        return data.repeat(1, 3, 1, 1)
    return data


def _soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _flatten_trigger_batches(trigger_batches):
    samples = [sample_batch for sample_batch, _ in trigger_batches]
    labels = [label_batch for _, label_batch in trigger_batches]
    return torch.cat(samples, dim=0), torch.cat(labels, dim=0)


def _trigger_accuracy(model, dataset, num_pairs, num_classes, key, image_size, num_white_points, device):
    trigger_batches = gen_trigger_4verify(
        dataset,
        num_pairs=num_pairs,
        num_classes=num_classes,
        key=key,
        image_width=image_size,
        image_height=image_size,
        num_white_points=num_white_points,
    )
    trigger_samples, trigger_labels = _flatten_trigger_batches(trigger_batches)
    trigger_samples = trigger_samples.to(device)
    trigger_labels = trigger_labels.to(device)

    with torch.no_grad():
        _, logits = model(trigger_samples)
        predictions = logits.argmax(dim=1)
        expected = trigger_labels.argmax(dim=1)
    return predictions.eq(expected).float().mean().item()


def _clean_accuracy(model, dataloader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = _prepare_inputs(data.to(device))
            target = target.to(device)
            _, logits = model(data)
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(target).sum().item()
            total += target.size(0)

    return correct / max(total, 1)


def _metadata_path(checkpoint_path):
    base, _ = os.path.splitext(checkpoint_path)
    return f"{base}.json"


def _save_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _save_checkpoint(model, checkpoint_path, metadata):
    directory = os.path.dirname(checkpoint_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    _save_json(_metadata_path(checkpoint_path), metadata)


def _load_metadata(checkpoint_path):
    metadata_path = _metadata_path(checkpoint_path)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
    return model


def _lora_dims_for_mode(run_config, lora_dims=None, checkpoint_path=None):
    if lora_dims is not None:
        return list(lora_dims)

    if checkpoint_path is not None:
        metadata = _load_metadata(checkpoint_path)
        if metadata and metadata.get("lora_dims") is not None:
            return list(metadata["lora_dims"])

    return list(run_config["default_lora_dims"])


def _build_model_for_mode(run_config, mode, lora_dims=None, checkpoint_path=None):
    model = build_model(run_config["dataset"])

    if mode == "lora":
        selected_lora_dims = _lora_dims_for_mode(run_config, lora_dims=lora_dims, checkpoint_path=checkpoint_path)
        model = replace_modules_with_lora(
            model,
            selected_lora_dims,
            init_std=run_config["training"]["lora_init_std"],
            trainmode=True,
        )
        return model, selected_lora_dims

    return model, None


def train(trainloader, testloader, trainset, testset, run_config, mode="lora", ref_path=None, lora_dims=None):
    dataset = run_config["dataset"]
    num_classes = run_config["num_classes"]
    image_size = run_config["image_size"]
    model_name = run_config["model_name"]
    key = run_config["key"]
    num_white_points = run_config["num_white_points"]
    verify_pairs = run_config["verify_pairs"]
    training_cfg = run_config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "ref":
        model, _ = _build_model_for_mode(run_config, mode="ref")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=training_cfg["ref_lr"])
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(training_cfg["ref_milestones"]),
            gamma=0.1,
        )
        checkpoint_path = run_config["ref_path"]
        best_clean_acc = -1.0
        best_metrics = None

        for epoch in range(training_cfg["ref_epochs"]):
            model.train()
            epoch_loss = 0.0

            for data, target in trainloader:
                data = _prepare_inputs(data.to(device))
                target = target.to(device)

                optimizer.zero_grad(set_to_none=True)
                _, logits = model(data)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            model.eval()
            clean_acc = _clean_accuracy(model, testloader, device)
            detect_acc = _trigger_accuracy(
                model,
                testset,
                num_pairs=verify_pairs,
                num_classes=num_classes,
                key=key,
                image_size=image_size,
                num_white_points=num_white_points,
                device=device,
            )
            avg_loss = epoch_loss / max(len(trainloader), 1)

            metrics = {
                "clean_acc": clean_acc,
                "detect_acc": detect_acc,
                "loss": avg_loss,
            }
            if clean_acc >= best_clean_acc:
                best_clean_acc = clean_acc
                best_metrics = metrics
                _save_checkpoint(
                    model,
                    checkpoint_path,
                    {
                        "paper_title": run_config["paper_title"],
                        "dataset": dataset,
                        "mode": "ref",
                        "model_name": model_name,
                        "metrics": metrics,
                    },
                )

            print(
                f"Epoch {epoch + 1}/{training_cfg['ref_epochs']} "
                f"loss={avg_loss:.4f} clean_acc={clean_acc:.4f} detect_acc={detect_acc:.4f}"
            )

        return {"checkpoint_path": checkpoint_path, "mode": mode, "metrics": best_metrics}

    if mode != "lora":
        raise ValueError(f"Unsupported mode: {mode}")

    reference_path = ref_path or run_config["ref_path"]
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference checkpoint not found: {reference_path}")

    base_state = torch.load(reference_path, map_location="cpu")
    base_model = build_model(dataset)
    base_model.load_state_dict(base_state, strict=True)
    selected_lora_dims = _lora_dims_for_mode(run_config, lora_dims=lora_dims)
    lora_model = replace_modules_with_lora(
        base_model,
        selected_lora_dims,
        init_std=training_cfg["lora_init_std"],
        trainmode=True,
    ).to(device)

    for name, parameter in lora_model.named_parameters():
        parameter.requires_grad = "lora_" in name

    trainable_parameters = [parameter for parameter in lora_model.parameters() if parameter.requires_grad]
    optimizer = optim.Adam(trainable_parameters, lr=training_cfg["lora_lr"])
    checkpoint_path = run_config["lora_path"]
    best_metrics = None
    best_rank = (-1.0, -1.0)

    for epoch in range(training_cfg["lora_epochs"]):
        lora_model.train()
        freeze_bn(lora_model)
        epoch_loss = 0.0

        for _ in range(training_cfg["lora_steps_per_epoch"]):
            trigger_batches = gen_trigger_4train(
                trainset,
                num_pairs=run_config["num_mixes"],
                num_classes=num_classes,
                key=key,
                image_width=image_size,
                image_height=image_size,
                num_white_points=num_white_points,
            )
            trigger_samples, trigger_labels = _flatten_trigger_batches(trigger_batches)
            trigger_samples = trigger_samples.to(device)
            trigger_labels = trigger_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits = lora_model(trigger_samples)
            loss = _soft_cross_entropy(logits, trigger_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        lora_model.eval()
        clean_acc = _clean_accuracy(lora_model, testloader, device)
        trigger_acc = _trigger_accuracy(
            lora_model,
            trainset,
            num_pairs=run_config["num_mixes"],
            num_classes=num_classes,
            key=key,
            image_size=image_size,
            num_white_points=num_white_points,
            device=device,
        )
        detect_acc = _trigger_accuracy(
            lora_model,
            testset,
            num_pairs=verify_pairs,
            num_classes=num_classes,
            key=key,
            image_size=image_size,
            num_white_points=num_white_points,
            device=device,
        )
        avg_loss = epoch_loss / max(training_cfg["lora_steps_per_epoch"], 1)

        metrics = {
            "clean_acc": clean_acc,
            "trigger_acc": trigger_acc,
            "detect_acc": detect_acc,
            "loss": avg_loss,
        }
        rank = (detect_acc, clean_acc)
        if rank >= best_rank:
            best_rank = rank
            best_metrics = metrics
            _save_checkpoint(
                lora_model,
                checkpoint_path,
                {
                    "paper_title": run_config["paper_title"],
                    "dataset": dataset,
                    "mode": "lora",
                    "model_name": model_name,
                    "lora_dims": selected_lora_dims,
                    "metrics": metrics,
                },
            )

        print(
            f"Epoch {epoch + 1}/{training_cfg['lora_epochs']} "
            f"loss={avg_loss:.4f} clean_acc={clean_acc:.4f} "
            f"trigger_acc={trigger_acc:.4f} detect_acc={detect_acc:.4f}"
        )

    return {
        "checkpoint_path": checkpoint_path,
        "mode": mode,
        "lora_dims": selected_lora_dims,
        "metrics": best_metrics,
    }


def evaluate(trainloader, testloader, trainset, testset, run_config, mode, checkpoint_path=None, lora_dims=None):
    checkpoint_path = checkpoint_path or run_config[f"{mode}_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, selected_lora_dims = _build_model_for_mode(
        run_config,
        mode=mode,
        lora_dims=lora_dims,
        checkpoint_path=checkpoint_path,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    metrics = {
        "dataset": run_config["dataset"],
        "mode": mode,
        "checkpoint_path": checkpoint_path,
        "clean_acc": _clean_accuracy(model, testloader, device),
        "detect_acc": _trigger_accuracy(
            model,
            testset,
            num_pairs=run_config["verify_pairs"],
            num_classes=run_config["num_classes"],
            key=run_config["key"],
            image_size=run_config["image_size"],
            num_white_points=run_config["num_white_points"],
            device=device,
        ),
    }

    if mode == "lora":
        metrics["lora_dims"] = selected_lora_dims
        metrics["trigger_acc"] = _trigger_accuracy(
            model,
            trainset,
            num_pairs=run_config["num_mixes"],
            num_classes=run_config["num_classes"],
            key=run_config["key"],
            image_size=run_config["image_size"],
            num_white_points=run_config["num_white_points"],
            device=device,
        )

    return metrics
