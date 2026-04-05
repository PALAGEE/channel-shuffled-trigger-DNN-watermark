# channel-shuffled-trigger-DNN-watermark
Code for the paper "Defending Against Ambiguity Attacks: Secret-Key-Driven DNN Watermarking for Ownership Verification"
# Defending Against Ambiguity Attacks: Secret-Key-Driven DNN Watermarking for Ownership Verification

This repository is a minimal reproducible package for the core method in [`manuscript405.pdf`](./manuscript405.pdf). It is organized as a small project instead of a loose collection of experiment scripts.

## Project Layout

```text
project/
├─ README.md
├─ requirements.txt
├─ train.py
├─ evaluate.py
├─ configs/
│  └─ defaults.json
├─ scripts/
│  ├─ README.md
│  └─ search_lora.py
├─ models/
│  ├─ README.md
│  ├─ __init__.py
│  ├─ builder.py
│  ├─ alexnet.py
│  ├─ efficientnet.py
│  ├─ mobilenetv2.py
│  ├─ resnet.py
│  ├─ wide_resnet.py
│  └─ checkpoints/
├─ utils/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ lora.py
│  ├─ trainer.py
│  └─ trigger.py
├─ LICENSE
└─ .gitignore
```

## 1. Paper Title

**Defending Against Ambiguity Attacks: Secret-Key-Driven DNN Watermarking for Ownership Verification**

## 2. Environment Requirements

Verified environment:

- Python `3.11`
- PyTorch `2.1.1`
- torchvision `0.16.1`
- numpy `1.24+`

Install:

```bash
conda create -n dnn-watermark python=3.11
conda activate dnn-watermark
pip install -r requirements.txt
```

Notes:

- GPU is recommended for training and NAS search.
- CPU can run the code, but larger datasets and rank search will be slow.
- All paths in the default config are relative paths. No local absolute path is hardcoded.

## 3. Dataset Source

Supported datasets:

- `MNIST`
- `CIFAR10`
- `CIFAR100`
- `FOOD101`
- `Caltech101`
- `CALTECH256`

Source summary:

- `MNIST`, `CIFAR10`, `CIFAR100` are downloaded automatically via `torchvision`.
- CIFAR official page: https://www.cs.toronto.edu/~kriz/cifar.html
- Food-101 official page: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- Caltech dataset index: https://www.vision.caltech.edu/datasets/
- Caltech-256 release page: https://data.caltech.edu/records/nyy15-4j048

For `FOOD101`, `Caltech101`, and `CALTECH256`, prepare ImageFolder-style splits:

```text
data/
  food-101/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  caltech-101/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  caltech-256/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
```

## 4. Configuration

The main configuration file is [`configs/defaults.json`](./configs/defaults.json).

Put all key experiment settings there instead of editing long scripts.

Important fields:

- `paths.dataset_path`: dataset root, default `./data`
- `paths.trained_path`: checkpoint/output root, default `./models/checkpoints`
- `watermark.key`: secret key used to regenerate trigger groups
- `watermark.num_white_points`: number of white pixels inserted into each trigger
- `watermark.num_mixes`: number of trigger groups sampled during training
- `evaluation.verify_pairs`: number of trigger groups sampled during evaluation
- `datasets.<dataset>.default_lora_dims`: default LoRA rank vector
- `datasets.<dataset>.training.*`: dataset-specific epochs and scheduler milestones
- `nas.*`: genetic search hyperparameters

## 5. Training Command

### Step 1: train the clean reference backbone

```bash
python train.py --config configs/defaults.json --dataset CIFAR10 --mode ref
```

Default output:

- checkpoint: `models/checkpoints/CIFAR10/ref/CIFAR10_ResNet18_ref.pt`
- metadata: `models/checkpoints/CIFAR10/ref/CIFAR10_ResNet18_ref.json`

### Step 2: train the LoRA watermark on top of the reference checkpoint

```bash
python train.py --config configs/defaults.json --dataset CIFAR10 --mode lora
```

If the reference checkpoint is not in the default location:

```bash
python train.py --config configs/defaults.json --dataset CIFAR10 --mode lora --ref-path models/checkpoints/CIFAR10/ref/CIFAR10_ResNet18_ref.pt
```

Default output:

- checkpoint: `models/checkpoints/CIFAR10/lora/CIFAR10_ResNet18_lora.pt`
- metadata: `models/checkpoints/CIFAR10/lora/CIFAR10_ResNet18_lora.json`

## 6. Testing Command

### Evaluate the clean reference checkpoint

```bash
python evaluate.py --config configs/defaults.json --dataset CIFAR10 --mode ref
```

### Evaluate the LoRA watermarked checkpoint

```bash
python evaluate.py --config configs/defaults.json --dataset CIFAR10 --mode lora
```

Evaluate a custom checkpoint:

```bash
python evaluate.py --config configs/defaults.json --dataset CIFAR10 --mode lora --checkpoint models/checkpoints/CIFAR10/lora/CIFAR10_ResNet18_lora.pt
```

The command prints JSON metrics with these main fields:

- `clean_acc`: clean test accuracy
- `trigger_acc`: trigger-group accuracy on sampled trigger groups from the training split
- `detect_acc`: ownership verification accuracy on sampled trigger groups from the test split

## 7. Main Results Come From Which Step

Use the workflow below if you want to reproduce the main outputs of the paper.

1. Run `train.py --mode ref`
   Produces the clean reference backbone.
2. Run `train.py --mode lora`
   Produces the watermarked LoRA checkpoint.
3. Run `evaluate.py --mode lora`
   Produces the main watermarking metrics: `clean_acc`, `trigger_acc`, and `detect_acc`.
4. Run `python scripts/search_lora.py --config configs/defaults.json --dataset CIFAR10`
   Produces the searched LoRA rank vector and the NAS history files.

In short:

- main watermarking result: **Step 2 + Step 3**
- lightweight LoRA configuration result: **Step 4**

## 8. LoRA Rank Search

The genetic NAS script is placed under [`scripts/search_lora.py`](./scripts/search_lora.py):

```bash
python scripts/search_lora.py --config configs/defaults.json --dataset CIFAR10
```

Default outputs:

- history CSV: `models/checkpoints/CIFAR10/nas/cifar10_lora_search_history.csv`
- summary JSON: `models/checkpoints/CIFAR10/nas/cifar10_lora_search_history_summary.json`

## 9. What Each Directory Does

- [`configs/`](./configs): experiment defaults
- [`scripts/`](./scripts): auxiliary reproducibility scripts such as LoRA NAS search
- [`models/`](./models): backbone definitions and checkpoint storage
- [`utils/`](./utils): config loading, dataset loading, trigger generation, LoRA wrapping, training logic

## 10. License

This code is released under the MIT license. See [`LICENSE`](./LICENSE).
