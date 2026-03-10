#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib

import torch
import yaml

from torchtmpl.data import get_polsar_output_channels
from torchtmpl.main import get_softmax
from torchtmpl.models import build_model
from torchtmpl.models.projection import MLPCtoR, ModCtoR, NoCtoR, PolyCtoR


PROJECTION_CLASSES = {
    "PolyCtoR": PolyCtoR,
    "MLPCtoR": MLPCtoR,
    "ModCtoR": ModCtoR,
    "NoCtoR": NoCtoR,
}


def load_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_projection(config: dict):
    projection_cfg = config["model"]["projection"]
    class_name = projection_cfg["class"]
    if class_name == "NoCtoR":
        return NoCtoR()
    return PROJECTION_CLASSES[class_name]()


def dtype_from_config(name: str) -> torch.dtype:
    if name == "complex64":
        return torch.complex64
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report parameter counts for a set of model config files."
    )
    parser.add_argument("configs", nargs="+", help="YAML config paths")
    args = parser.parse_args()

    print("config\ttotal_params\ttrainable_params\tchannels_ratio\tnum_layers")
    for config_path_str in args.configs:
        config_path = pathlib.Path(config_path_str)
        config = load_config(config_path)
        projection = build_projection(config)
        softmax = get_softmax(config["model"]["projection"])
        dtype = dtype_from_config(config["dtype"])
        num_channels = get_polsar_output_channels(config["data"])

        model = build_model(
            config,
            projection=projection,
            softmax=softmax,
            dtype=dtype,
            img_size=config["data"]["img_size"],
            num_classes=None,
            num_channels=num_channels,
        )
        total_params, trainable_params = count_parameters(model)
        print(
            f"{config_path}\t{total_params}\t{trainable_params}\t"
            f"{config['model']['channels_ratio']}\t{config['model']['num_layers']}"
        )


if __name__ == "__main__":
    main()
