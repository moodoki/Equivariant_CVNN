from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


FULL_POLARIZATIONS = ("HH", "HV", "VH", "VV")
ALOS_STYLE_POLARIZATIONS = ("HH", "HV", "VV")


@dataclass(frozen=True)
class Tel2CommercialProduct:
    scene_name: str
    product_name: str
    product_root: Path
    slc_root: Path
    rows: int
    cols: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    patch_rows: int
    patch_cols: int
    patch_count: int
    dat_files: dict[str, Path]


def _normalize_optional_names(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {value for value in values if value}


def _resolve_raw_scale_config(
    raw_scale_factor: float | str | None,
    raw_scale_percentile: float,
) -> tuple[float | None, float | None, str]:
    if raw_scale_factor is None:
        return None, None, "disabled"

    if isinstance(raw_scale_factor, str):
        token = raw_scale_factor.strip().lower()
        if token in {"auto", "default", "percentile", "p99.5"}:
            return None, float(raw_scale_percentile), f"percentile_{raw_scale_percentile:g}"
        raise ValueError(
            "Unsupported raw_scale_factor string "
            f"{raw_scale_factor!r}; expected 'auto', 'default', 'percentile', or a float."
        )

    resolved = float(raw_scale_factor)
    if resolved <= 0:
        raise ValueError("raw_scale_factor must be strictly positive when provided.")
    return resolved, None, "explicit"


def _resolve_output_polarizations(
    output_mode: str,
    output_polarizations: Sequence[str] | None,
) -> tuple[str, ...]:
    if output_polarizations is not None:
        resolved = tuple(pol.upper() for pol in output_polarizations)
    elif output_mode == "quad":
        resolved = FULL_POLARIZATIONS
    elif output_mode == "three_channel":
        resolved = ALOS_STYLE_POLARIZATIONS
    else:
        raise ValueError(
            f"Unsupported output_mode {output_mode!r}; expected 'three_channel' or 'quad'."
        )

    invalid = [pol for pol in resolved if pol not in FULL_POLARIZATIONS]
    if invalid:
        raise ValueError(f"Unsupported polarizations requested: {invalid!r}")
    return resolved


def _resolve_crop(
    rows: int,
    cols: int,
    crop_coordinates: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    if crop_coordinates is None:
        return (0, rows, 0, cols)

    (row_start, col_start), (row_end, col_end) = crop_coordinates
    if not (0 <= row_start < row_end <= rows):
        raise ValueError(
            f"Invalid TEL2 crop rows {(row_start, row_end)!r} for image height {rows}."
        )
    if not (0 <= col_start < col_end <= cols):
        raise ValueError(
            f"Invalid TEL2 crop cols {(col_start, col_end)!r} for image width {cols}."
        )
    return (row_start, row_end, col_start, col_end)


def _extract_raster_size(params: dict) -> tuple[int, int]:
    raster = params["level1Product"]["productInfo"]["imageDataInfo"]["imageRaster"]
    return int(raster["numberOfRows"]), int(raster["numberOfColumns"])


def _discover_products(
    root: Path,
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
    crop_coordinates: tuple[tuple[int, int], tuple[int, int]] | None,
    scene_names: Sequence[str] | None,
    product_names: Sequence[str] | None,
) -> list[Tel2CommercialProduct]:
    selected_scenes = _normalize_optional_names(scene_names)
    selected_products = _normalize_optional_names(product_names)
    patch_height, patch_width = patch_size
    stride_height, stride_width = patch_stride

    products: list[Tel2CommercialProduct] = []
    for params_path in sorted(root.rglob("slc/annotation/params.json")):
        product_root = params_path.parents[2]
        relative_product = product_root.relative_to(root)
        scene_name = relative_product.parts[0] if len(relative_product.parts) > 1 else product_root.name
        product_name = product_root.name

        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        if selected_products is not None and product_name not in selected_products:
            continue

        with params_path.open("r", encoding="utf-8") as stream:
            params = json.load(stream)

        rows, cols = _extract_raster_size(params)
        row_start, row_end, col_start, col_end = _resolve_crop(
            rows, cols, crop_coordinates
        )
        cropped_rows = row_end - row_start
        cropped_cols = col_end - col_start

        if cropped_rows < patch_height or cropped_cols < patch_width:
            raise ValueError(
                f"TEL2 crop {(cropped_rows, cropped_cols)!r} is smaller than patch_size {patch_size!r}."
            )

        patch_rows = 1 + (cropped_rows - patch_height) // stride_height
        patch_cols = 1 + (cropped_cols - patch_width) // stride_width

        slc_root = product_root / "slc"
        dat_files = {
            pol: slc_root / f"{pol.lower()}.dat" for pol in FULL_POLARIZATIONS
        }
        missing = [str(file_path) for file_path in dat_files.values() if not file_path.exists()]
        if missing:
            raise FileNotFoundError(
                f"TEL2 product {product_root} is missing polarization files: {missing!r}"
            )

        products.append(
            Tel2CommercialProduct(
                scene_name=scene_name,
                product_name=product_name,
                product_root=product_root,
                slc_root=slc_root,
                rows=rows,
                cols=cols,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                patch_rows=patch_rows,
                patch_cols=patch_cols,
                patch_count=patch_rows * patch_cols,
                dat_files=dat_files,
            )
        )

    if not products:
        raise FileNotFoundError(
            f"No TEL2 commercial products were found under {root}."
        )

    return products


class Tel2Commrcial_v1(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        patch_size: tuple[int, int] | None = None,
        patch_stride: tuple[int, int] | None = None,
        crop_coordinates: tuple[tuple[int, int], tuple[int, int]] | None = None,
        output_mode: str = "three_channel",
        output_polarizations: Sequence[str] | None = None,
        scene_names: Sequence[str] | None = None,
        product_names: Sequence[str] | None = None,
        raw_scale_factor: float | str | None = "percentile",
        raw_scale_percentile: float = 99.5,
    ) -> None:
        super().__init__()
        if patch_size is None or patch_stride is None:
            raise ValueError("TEL2 dataset requires patch_size and patch_stride.")

        self.root = Path(root)
        self.transform = transform
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.crop_coordinates = crop_coordinates
        self.output_mode = output_mode
        self.output_polarizations = _resolve_output_polarizations(
            output_mode=output_mode,
            output_polarizations=output_polarizations,
        )
        self.channel_names = self.output_polarizations
        self.full_polarizations = FULL_POLARIZATIONS
        self.supports_full_image_reconstruction = True

        self.products = _discover_products(
            root=self.root,
            patch_size=patch_size,
            patch_stride=patch_stride,
            crop_coordinates=crop_coordinates,
            scene_names=scene_names,
            product_names=product_names,
        )
        self._offsets = np.cumsum([0] + [product.patch_count for product in self.products])
        self._memmaps: dict[tuple[str, str], np.memmap] = {}
        resolved_scale, resolved_percentile, scale_source = _resolve_raw_scale_config(
            raw_scale_factor=raw_scale_factor,
            raw_scale_percentile=raw_scale_percentile,
        )
        self.raw_scale_percentile = resolved_percentile
        self.raw_scale_source = scale_source
        self.raw_scale_factor = resolved_scale
        if self.raw_scale_percentile is not None:
            self.raw_scale_factor = self._compute_raw_scale_factor(
                percentile=self.raw_scale_percentile
            )

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def nsamples_per_rows(self) -> int:
        self._require_single_product()
        return self.products[0].patch_rows

    @property
    def nsamples_per_cols(self) -> int:
        self._require_single_product()
        return self.products[0].patch_cols

    def _require_single_product(self) -> None:
        if len(self.products) != 1:
            raise ValueError(
                "Full-image TEL2 reconstruction currently requires exactly one discovered product. "
                "Filter the dataset with scene_names or product_names before calling the test/reassembly path."
            )

    def _locate_patch(self, index: int) -> tuple[Tel2CommercialProduct, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        product_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - int(self._offsets[product_index])
        return self.products[product_index], local_index

    def _get_interleaved_memmap(self, product: Tel2CommercialProduct, pol: str) -> np.memmap:
        cache_key = (str(product.product_root), pol)
        memmap = self._memmaps.get(cache_key)
        if memmap is None:
            memmap = np.memmap(
                product.dat_files[pol],
                dtype=np.float32,
                mode="r",
                shape=(product.rows, product.cols * 2),
            )
            self._memmaps[cache_key] = memmap
        return memmap

    def _compute_raw_scale_factor(self, percentile: float) -> float:
        amplitudes: list[np.ndarray] = []
        for product in self.products:
            patch = self._read_patch(
                product,
                product.row_start,
                product.row_end,
                product.col_start,
                product.col_end,
                apply_raw_scale=False,
            )
            stacked = self._stack_output_channels(patch)
            amplitudes.append(np.abs(stacked).reshape(-1))

        raw_scale_factor = float(np.percentile(np.concatenate(amplitudes), percentile))
        if raw_scale_factor <= 0:
            raise ValueError(
                f"Computed TEL2 raw scale factor {raw_scale_factor!r} is not strictly positive."
            )
        return raw_scale_factor

    def _read_patch(
        self,
        product: Tel2CommercialProduct,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        apply_raw_scale: bool = True,
    ) -> dict[str, np.ndarray]:
        channels: dict[str, np.ndarray] = {}
        for pol in FULL_POLARIZATIONS:
            interleaved = self._get_interleaved_memmap(product, pol)
            raw_patch = interleaved[row_start:row_end, col_start * 2 : col_end * 2]
            complex_patch = raw_patch[:, ::2] + 1j * raw_patch[:, 1::2]
            if apply_raw_scale and self.raw_scale_factor is not None:
                complex_patch = complex_patch / self.raw_scale_factor
            channels[pol] = np.asarray(complex_patch, dtype=np.complex64)
        return channels

    def _stack_output_channels(self, patch_dict: dict[str, np.ndarray]) -> np.ndarray:
        if self.output_mode == "quad":
            return np.stack([patch_dict[pol] for pol in FULL_POLARIZATIONS], axis=0)

        if self.output_polarizations != ALOS_STYLE_POLARIZATIONS:
            return np.stack([patch_dict[pol] for pol in self.output_polarizations], axis=0)

        return np.stack(
            (
                patch_dict["HH"],
                0.5 * (patch_dict["HV"] + patch_dict["VH"]),
                patch_dict["VV"],
            ),
            axis=0,
        )

    def __getitem__(self, index: int) -> torch.Tensor | np.ndarray:
        product, local_index = self._locate_patch(index)
        patch_row = local_index // product.patch_cols
        patch_col = local_index % product.patch_cols

        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.patch_stride
        row_start = product.row_start + patch_row * stride_height
        col_start = product.col_start + patch_col * stride_width
        row_end = row_start + patch_height
        col_end = col_start + patch_width

        patch = self._read_patch(product, row_start, row_end, col_start, col_end)
        if self.transform is not None:
            return self.transform(patch)
        return self._stack_output_channels(patch)

    def iter_products(self) -> Iterable[Tel2CommercialProduct]:
        return iter(self.products)
