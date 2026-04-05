import hashlib

import torch
import torch.nn.functional as F


CHANNEL_ORDERS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)
MIXUP_RATIO = 0.5


def hash_function(key, seed):
    digest = hashlib.sha256(f"{key}_{seed}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def apply_channel_shuffle(image, order_idx):
    order = CHANNEL_ORDERS[order_idx]
    return image[list(order)]


def extract_image_feature_hash(image_tensor):
    bins = [int(channel.mean().item() / 50) for channel in image_tensor]
    return "_".join(str(value) for value in bins)


def generate_white_dot_positions(
    key,
    channel_order_idx,
    image_feature,
    num_white_points,
    image_width,
    image_height,
):
    seed = f"{key}_{channel_order_idx}_{image_feature}"
    positions = []

    for offset in range(num_white_points):
        hash_value = hash_function(seed, offset)
        x_coord = hash_value % image_width
        y_coord = (hash_value // image_width) % image_height
        positions.append((x_coord, y_coord))

    return positions


def set_pixels_to_white(image_tensor, positions):
    for x_coord, y_coord in positions:
        image_tensor[:, y_coord, x_coord] = 255.0


def _build_trigger_sample(
    image_a,
    image_b,
    key,
    order_idx,
    num_classes,
    image_width,
    image_height,
    num_white_points,
):
    shuffled_image = apply_channel_shuffle(image_b, order_idx)
    mixed_image = MIXUP_RATIO * image_a + (1.0 - MIXUP_RATIO) * shuffled_image
    image_feature_hash = extract_image_feature_hash(mixed_image)
    white_positions = generate_white_dot_positions(
        key,
        order_idx,
        image_feature_hash,
        num_white_points,
        image_width,
        image_height,
    )

    trigger_image = mixed_image.clone()
    set_pixels_to_white(trigger_image, white_positions)

    label_index = sum(x_coord + y_coord for x_coord, y_coord in white_positions) % num_classes
    label = F.one_hot(torch.tensor(label_index), num_classes=num_classes).float()
    return trigger_image, label


def _sample_distinct_images(dataset):
    while True:
        first_idx, second_idx = torch.randint(0, len(dataset), (2,))
        image_a, label_a = dataset[first_idx]
        image_b, label_b = dataset[second_idx]
        if label_a != label_b:
            return image_a, image_b


def _generate_trigger_batches(
    dataset,
    num_pairs,
    num_classes,
    key,
    image_width,
    image_height,
    num_white_points,
):
    trigger_batches = []

    for _ in range(num_pairs):
        image_a, image_b = _sample_distinct_images(dataset)
        samples = []
        labels = []

        for order_idx in range(len(CHANNEL_ORDERS)):
            trigger_image, trigger_label = _build_trigger_sample(
                image_a,
                image_b,
                key,
                order_idx,
                num_classes,
                image_width,
                image_height,
                num_white_points,
            )
            samples.append(trigger_image)
            labels.append(trigger_label)

        trigger_batches.append((torch.stack(samples, dim=0), torch.stack(labels, dim=0)))

    return trigger_batches


def gen_trigger_4train(data, num_pairs, num_classes, key, image_width, image_height, num_white_points):
    return _generate_trigger_batches(
        data,
        num_pairs,
        num_classes,
        key,
        image_width,
        image_height,
        num_white_points,
    )


def gen_trigger_4verify(data, num_pairs, num_classes, key, image_width, image_height, num_white_points):
    return _generate_trigger_batches(
        data,
        num_pairs,
        num_classes,
        key,
        image_width,
        image_height,
        num_white_points,
    )
