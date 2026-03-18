#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from torchcvnn.datasets import ALOSDataset

from torchtmpl.data import (
    build_polsar_to_tensor_transform,
    extract_data_config,
    get_polsar_output_channels,
)
from torchtmpl.tel2commercial import Tel2Commrcial_v1


@dataclass(frozen=True)
class LogAmplitudeSpec:
    min_value: float = 0.02
    max_value: float = 40.0
    keep_phase: bool = True


@dataclass(frozen=True)
class ImageRegion:
    scene_name: str
    product_name: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    patch_rows: int
    patch_cols: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TEL2 dataset amplitude statistics, histograms, and overview "
            "images from a config file."
        )
    )
    parser.add_argument("config", help="Path to the TEL2 YAML config file.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for the markdown report and generated images. "
            "Defaults to <logging.logdir>/dataset_stats/<config-stem>."
        ),
    )
    parser.add_argument(
        "--patch-row",
        type=int,
        default=None,
        help="Patch-grid row to visualize. Defaults to the middle row.",
    )
    parser.add_argument(
        "--sampled-patches",
        type=int,
        default=8,
        help="How many patches to sample across the chosen patch-grid row.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=256,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_supported_config(config: dict) -> None:
    dataset_name = config["data"]["dataset"]["name"]
    if dataset_name not in {"Tel2Commrcial_v1", "ALOSDataset"}:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}; this report only supports Tel2Commrcial_v1 or ALOSDataset."
        )


def parse_transform_names(transform_name: str) -> list[str]:
    return [name.strip() for name in transform_name.split(",") if name.strip()]


def get_log_amplitude_spec(config: dict) -> LogAmplitudeSpec:
    transform_names = parse_transform_names(config["data"]["transform"])
    if "LogAmplitude" not in transform_names:
        raise ValueError(
            "The config transform pipeline does not include LogAmplitude; "
            "this report expects before/after LogAmplitude statistics."
        )
    return LogAmplitudeSpec()


def resolve_output_dir(config_path: Path, config: dict, requested: str | None) -> Path:
    if requested is not None:
        return Path(requested)

    logdir = Path(os.path.expandvars(config["logging"]["logdir"]))
    return logdir / "dataset_stats" / config_path.stem


def build_dataset(config: dict) -> Tel2Commrcial_v1:
    data_config = config["data"]
    img_size, img_stride, *_rest, trainpath, _transform = extract_data_config(data_config)
    dataset_name = data_config["dataset"]["name"]
    dataset_config = data_config["dataset"]
    crop_coordinates = None
    if "crop" in data_config:
        crop_coordinates = (
            (data_config["crop"]["start_row"], data_config["crop"]["start_col"]),
            (data_config["crop"]["end_row"], data_config["crop"]["end_col"]),
        )

    if dataset_name == "Tel2Commrcial_v1":
        return Tel2Commrcial_v1(
            root=trainpath,
            transform=None,
            patch_size=img_size,
            patch_stride=img_stride,
            crop_coordinates=crop_coordinates,
            output_mode=dataset_config.get("output_mode", "three_channel"),
            output_polarizations=dataset_config.get("output_polarizations"),
            scene_names=dataset_config.get("scene_names"),
            product_names=dataset_config.get("product_names"),
        )

    volpath = Path(trainpath) / "VOL-ALOS2044980750-150324-HBQR1.1__A"
    return ALOSDataset(
        volpath=volpath,
        transform=None,
        crop_coordinates=crop_coordinates,
        patch_size=img_size,
        patch_stride=img_stride,
    )


def get_channel_names(config: dict, dataset) -> list[str]:
    dataset_name = config["data"]["dataset"]["name"]
    if dataset_name == "Tel2Commrcial_v1":
        return list(dataset.channel_names)

    out_channels = get_polsar_output_channels(config["data"])
    if out_channels == 3:
        return ["HH", "HV", "VV"]
    if out_channels == 4:
        return ["HH", "HV", "VH", "VV"]
    return [f"channel_{index}" for index in range(out_channels)]


def iter_regions(dataset_name: str, dataset) -> list[ImageRegion]:
    if dataset_name == "Tel2Commrcial_v1":
        regions = []
        for product in dataset.iter_products():
            regions.append(
                ImageRegion(
                    scene_name=product.scene_name,
                    product_name=product.product_name,
                    row_start=product.row_start,
                    row_end=product.row_end,
                    col_start=product.col_start,
                    col_end=product.col_end,
                    patch_rows=product.patch_rows,
                    patch_cols=product.patch_cols,
                )
            )
        return regions

    (row_start, col_start), (row_end, col_end) = dataset.crop_coordinates
    return [
        ImageRegion(
            scene_name="SAN_FRANCISCO_ALOS2",
            product_name="VOL-ALOS2044980750-150324-HBQR1.1__A",
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            patch_rows=dataset.nsamples_per_rows,
            patch_cols=dataset.nsamples_per_cols,
        )
    ]


def to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def build_output_stack(dataset_name: str, dataset, raw_patch: dict[str, np.ndarray], out_channels: int) -> np.ndarray:
    if dataset_name == "Tel2Commrcial_v1":
        return dataset._stack_output_channels(raw_patch)

    transform = build_polsar_to_tensor_transform(dataset_name, out_channels=out_channels)
    return to_numpy(transform(raw_patch))


def read_region_patch_dict(dataset_name: str, dataset, region: ImageRegion, row_start: int, row_end: int, col_start: int, col_end: int) -> dict[str, np.ndarray]:
    if dataset_name == "Tel2Commrcial_v1":
        product = next(
            product
            for product in dataset.iter_products()
            if product.scene_name == region.scene_name and product.product_name == region.product_name
        )
        return dataset._read_patch(product, row_start, row_end, col_start, col_end)

    num_rows = row_end - row_start
    num_cols = col_end - col_start
    calibration_factor = dataset.leaderFile.calibration_factor
    return {
        pol: image.read_patch(row_start, num_rows, col_start, num_cols) * calibration_factor
        for pol, image in dataset.images.items()
    }


def read_full_output_image(
    config: dict,
    dataset,
    region: ImageRegion,
) -> np.ndarray:
    dataset_name = config["data"]["dataset"]["name"]
    raw_patch = read_region_patch_dict(
        dataset_name,
        dataset,
        region,
        region.row_start,
        region.row_end,
        region.col_start,
        region.col_end,
    )
    out_channels = get_polsar_output_channels(config["data"])
    return build_output_stack(dataset_name, dataset, raw_patch, out_channels=out_channels)


def compute_amplitude(stack: np.ndarray) -> np.ndarray:
    return np.abs(stack).astype(np.float32, copy=False)


def apply_log_amplitude(
    amplitude: np.ndarray,
    spec: LogAmplitudeSpec,
) -> np.ndarray:
    clipped = np.clip(amplitude, spec.min_value, spec.max_value)
    denominator = math.log10(spec.max_value / spec.min_value)
    transformed = np.log10(clipped / spec.min_value) / denominator
    return transformed.astype(np.float32, copy=False)


def safe_slug(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars)


def format_float(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.6f}"


def compute_channel_stats(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 95)),
    )


def compute_clip_fractions(values: np.ndarray, spec: LogAmplitudeSpec) -> tuple[float, float]:
    total = values.size
    below = float(np.count_nonzero(values < spec.min_value) / total)
    above = float(np.count_nonzero(values > spec.max_value) / total)
    return below, above


def normalize_for_display(values: np.ndarray, vmax: float) -> np.ndarray:
    if vmax <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / vmax, 0.0, 1.0).astype(np.float32, copy=False)


def make_rgb_composite(
    amplitudes: np.ndarray,
    channel_p95: np.ndarray,
) -> np.ndarray:
    channels = amplitudes.shape[0]
    if channels == 1:
        gray = normalize_for_display(amplitudes[0], float(channel_p95[0]))
        return np.repeat(gray[..., None], 3, axis=2)

    if channels == 2:
        red = normalize_for_display(amplitudes[0], float(channel_p95[0]))
        green = normalize_for_display(amplitudes[1], float(channel_p95[1]))
        blue = np.zeros_like(red)
        return np.stack((red, green, blue), axis=-1)

    rgb_channels = []
    for channel_index in range(3):
        rgb_channels.append(
            normalize_for_display(
                amplitudes[channel_index], float(channel_p95[channel_index])
            )
        )
    return np.stack(rgb_channels, axis=-1)


def save_histogram_figure(
    raw_amplitude: np.ndarray,
    log_amplitude: np.ndarray,
    channel_names: Iterable[str],
    output_path: Path,
    bins: int,
) -> None:
    channel_names = list(channel_names)
    figure, axes = plt.subplots(
        2,
        len(channel_names),
        figsize=(4.5 * len(channel_names), 8),
        squeeze=False,
    )

    for channel_index, channel_name in enumerate(channel_names):
        raw_values = raw_amplitude[channel_index].reshape(-1)
        raw_p95 = float(np.percentile(raw_values, 95))
        raw_axis = axes[0, channel_index]
        raw_axis.hist(raw_values, bins=bins, range=(0.0, raw_p95), color="#1f77b4")
        raw_axis.set_xlim(0.0, raw_p95)
        raw_axis.set_title(f"{channel_name} raw amplitude")
        raw_axis.set_xlabel("Amplitude")
        raw_axis.set_ylabel("Pixel count")

        log_values = log_amplitude[channel_index].reshape(-1)
        log_p95 = float(np.percentile(log_values, 95))
        log_axis = axes[1, channel_index]
        log_axis.hist(log_values, bins=bins, range=(0.0, log_p95), color="#ff7f0e")
        log_axis.set_xlim(0.0, log_p95)
        log_axis.set_title(f"{channel_name} after LogAmplitude")
        log_axis.set_xlabel("LogAmplitude value")
        log_axis.set_ylabel("Pixel count")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_full_image_figure(
    raw_amplitude: np.ndarray,
    log_amplitude: np.ndarray,
    channel_names: Iterable[str],
    output_path: Path,
) -> None:
    channel_names = list(channel_names)
    figure, axes = plt.subplots(
        2,
        len(channel_names),
        figsize=(4.5 * len(channel_names), 8),
        squeeze=False,
    )

    for channel_index, channel_name in enumerate(channel_names):
        raw_channel = raw_amplitude[channel_index]
        raw_p95 = float(np.percentile(raw_channel, 95))
        raw_axis = axes[0, channel_index]
        raw_axis.imshow(raw_channel, cmap="gray", vmin=0.0, vmax=raw_p95)
        raw_axis.set_title(f"{channel_name} raw amplitude")
        raw_axis.axis("off")

        log_channel = log_amplitude[channel_index]
        log_p95 = float(np.percentile(log_channel, 95))
        log_axis = axes[1, channel_index]
        log_axis.imshow(log_channel, cmap="gray", vmin=0.0, vmax=log_p95)
        log_axis.set_title(f"{channel_name} after LogAmplitude")
        log_axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def save_patch_row_figure(
    config: dict,
    dataset,
    region: ImageRegion,
    patch_row: int,
    sampled_patches: int,
    spec: LogAmplitudeSpec,
    output_path: Path,
) -> tuple[list[int], int]:
    patch_row = max(0, min(patch_row, region.patch_rows - 1))
    sampled_patches = max(1, min(sampled_patches, region.patch_cols))
    patch_cols = np.linspace(
        0, region.patch_cols - 1, num=sampled_patches, dtype=int
    )
    patch_cols = np.unique(patch_cols).tolist()
    dataset_name = config["data"]["dataset"]["name"]
    out_channels = get_polsar_output_channels(config["data"])

    raw_composites = []
    log_composites = []
    for patch_col in patch_cols:
        patch_height, patch_width = dataset.patch_size
        stride_height, stride_width = dataset.patch_stride
        row_start = region.row_start + patch_row * stride_height
        col_start = region.col_start + patch_col * stride_width
        row_end = row_start + patch_height
        col_end = col_start + patch_width
        raw_patch = read_region_patch_dict(
            dataset_name,
            dataset,
            region,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        raw_stack = build_output_stack(
            dataset_name,
            dataset,
            raw_patch,
            out_channels=out_channels,
        )
        raw_amplitude = compute_amplitude(raw_stack)
        log_amplitude = apply_log_amplitude(raw_amplitude, spec)

        raw_p95 = np.array(
            [np.percentile(raw_amplitude[i], 95) for i in range(raw_amplitude.shape[0])],
            dtype=np.float32,
        )
        log_p95 = np.array(
            [np.percentile(log_amplitude[i], 95) for i in range(log_amplitude.shape[0])],
            dtype=np.float32,
        )
        raw_composites.append(make_rgb_composite(raw_amplitude, raw_p95))
        log_composites.append(make_rgb_composite(log_amplitude, log_p95))

    figure, axes = plt.subplots(
        2,
        len(patch_cols),
        figsize=(3.2 * len(patch_cols), 7),
        squeeze=False,
    )

    for index, patch_col in enumerate(patch_cols):
        raw_axis = axes[0, index]
        raw_axis.imshow(raw_composites[index])
        raw_axis.set_title(f"raw r{patch_row} c{patch_col}")
        raw_axis.axis("off")

        log_axis = axes[1, index]
        log_axis.imshow(log_composites[index])
        log_axis.set_title(f"log r{patch_row} c{patch_col}")
        log_axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)
    return patch_cols, patch_row


def render_stats_table(
    channel_names: list[str],
    raw_amplitude: np.ndarray,
    log_amplitude: np.ndarray,
    spec: LogAmplitudeSpec,
) -> str:
    lines = [
        "| Channel | Raw min | Raw max | Raw p95 | < log min | > log max | Log min | Log max | Log p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for channel_index, channel_name in enumerate(channel_names):
        raw_values = raw_amplitude[channel_index]
        log_values = log_amplitude[channel_index]
        raw_min, raw_max, raw_p95 = compute_channel_stats(raw_values)
        log_min, log_max, log_p95 = compute_channel_stats(log_values)
        frac_below, frac_above = compute_clip_fractions(raw_values, spec)
        lines.append(
            "| "
            + " | ".join(
                (
                    channel_name,
                    format_float(raw_min),
                    format_float(raw_max),
                    format_float(raw_p95),
                    f"{frac_below:.4%}",
                    f"{frac_above:.4%}",
                    format_float(log_min),
                    format_float(log_max),
                    format_float(log_p95),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    config_path: Path,
    config: dict,
    dataset,
    channel_names: list[str],
    spec: LogAmplitudeSpec,
    product_sections: list[str],
) -> Path:
    report_path = output_dir / "dataset_stats.md"
    timestamp = datetime.now(timezone.utc).isoformat()
    dataset_root = os.path.expandvars(config["data"]["dataset"]["trainpath"])
    lines = [
        "# Dataset Statistics Report",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Config: `{config_path}`",
        f"- Dataset name: `{config['data']['dataset']['name']}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Output channels: `{', '.join(channel_names)}`",
        f"- Patch size: `{dataset.patch_size}`",
        f"- Patch stride: `{dataset.patch_stride}`",
        f"- LogAmplitude clamp: `[{spec.min_value}, {spec.max_value}]`",
        "",
        "The raw statistics are computed from amplitude values of the exact channels "
        "fed into training. The post-LogAmplitude statistics use the same clamp and "
        "normalization as `torchcvnn.transforms.LogAmplitude`.",
        "",
    ]
    lines.extend(product_sections)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    require_supported_config(config)
    spec = get_log_amplitude_spec(config)
    dataset = build_dataset(config)
    dataset_name = config["data"]["dataset"]["name"]
    channel_names = get_channel_names(config, dataset)

    output_dir = resolve_output_dir(config_path, config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_sections: list[str] = []
    for region in iter_regions(dataset_name, dataset):
        product_slug = safe_slug(f"{region.scene_name}_{region.product_name}")
        raw_stack = read_full_output_image(config, dataset, region)
        raw_amplitude = compute_amplitude(raw_stack)
        log_amplitude = apply_log_amplitude(raw_amplitude, spec)

        histogram_path = output_dir / f"{product_slug}_histograms.png"
        full_image_path = output_dir / f"{product_slug}_full_image.png"
        patch_row_path = output_dir / f"{product_slug}_patch_row.png"

        save_histogram_figure(
            raw_amplitude=raw_amplitude,
            log_amplitude=log_amplitude,
            channel_names=channel_names,
            output_path=histogram_path,
            bins=args.hist_bins,
        )
        save_full_image_figure(
            raw_amplitude=raw_amplitude,
            log_amplitude=log_amplitude,
            channel_names=channel_names,
            output_path=full_image_path,
        )
        patch_row = args.patch_row
        if patch_row is None:
            patch_row = region.patch_rows // 2
        patch_cols, resolved_patch_row = save_patch_row_figure(
            config=config,
            dataset=dataset,
            region=region,
            patch_row=patch_row,
            sampled_patches=args.sampled_patches,
            spec=spec,
            output_path=patch_row_path,
        )

        product_sections.extend(
            [
                f"## {region.scene_name} / {region.product_name}",
                "",
                f"- Crop rows: `{region.row_start}:{region.row_end}`",
                f"- Crop cols: `{region.col_start}:{region.col_end}`",
                f"- Image size: `{region.row_end - region.row_start} x {region.col_end - region.col_start}`",
                f"- Patch grid: `{region.patch_rows} x {region.patch_cols}`",
                f"- Visualized patch row: `{resolved_patch_row}`",
                f"- Sampled patch columns: `{', '.join(str(col) for col in patch_cols)}`",
                "",
                render_stats_table(
                    channel_names=channel_names,
                    raw_amplitude=raw_amplitude,
                    log_amplitude=log_amplitude,
                    spec=spec,
                ),
                "",
                "### Histograms",
                "",
                f"![Histograms]({histogram_path.name})",
                "",
                "### Entire Image",
                "",
                f"![Entire image]({full_image_path.name})",
                "",
                "### Patch Row",
                "",
                f"![Patch row]({patch_row_path.name})",
                "",
            ]
        )

    report_path = write_report(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        dataset=dataset,
        channel_names=channel_names,
        spec=spec,
        product_sections=product_sections,
    )
    print(f"Wrote TEL2 dataset stats report to {report_path}")


if __name__ == "__main__":
    main()
