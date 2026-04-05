import copy
from collections import deque

import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, layer, lora_dim, is_conv=False, init_std=0.5, trainmode=False):
        super().__init__()
        self.layer = layer
        self.is_conv = is_conv

        if is_conv:
            self.lora_down = nn.Conv2d(
                layer.in_channels,
                lora_dim,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                bias=False,
            )
            self.lora_up = nn.Conv2d(lora_dim, layer.out_channels, kernel_size=1, bias=False)
        else:
            self.lora_down = nn.Linear(layer.in_features, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, layer.out_features, bias=False)

        nn.init.normal_(self.lora_down.weight, mean=0.0, std=init_std)
        if trainmode:
            nn.init.zeros_(self.lora_up.weight)
        else:
            nn.init.normal_(self.lora_up.weight, mean=0.0, std=init_std)

    def forward(self, x):
        if self.is_conv:
            return self.layer(x) + self.lora_up(self.lora_down(x))

        flattened = x.view(x.size(0), -1)
        return self.layer(flattened) + self.lora_up(self.lora_down(flattened))


def replace_modules_with_lora(module, lora_dims, init_std, trainmode=False):
    new_module = copy.deepcopy(module)
    index = 0
    queue = deque([new_module])
    linear_layers = []

    while queue:
        current_module = queue.popleft()
        for name, child in current_module.named_children():
            if isinstance(child, nn.Conv2d):
                if index < len(lora_dims) and lora_dims[index] > 0:
                    setattr(
                        current_module,
                        name,
                        LoRALayer(child, lora_dims[index], is_conv=True, init_std=init_std, trainmode=trainmode),
                    )
                index += 1
            elif isinstance(child, nn.Linear):
                linear_layers.append((current_module, name, child))
            elif any(child.children()):
                queue.append(child)

    for current_module, name, child in linear_layers:
        if index < len(lora_dims) and lora_dims[index] > 0:
            setattr(
                current_module,
                name,
                LoRALayer(child, lora_dims[index], is_conv=False, init_std=init_std, trainmode=trainmode),
            )
        index += 1

    return new_module
