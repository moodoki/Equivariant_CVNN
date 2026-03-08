import copy
import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage


def flatten_config(config: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, str]] = []

    for key, value in config.items():
        if sep in key:
            key = key.replace(sep, f"_{sep}_")

        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            if value:
                items.extend(flatten_config(value, new_key, sep=sep).items())
        elif isinstance(value, (list, tuple)):
            items.append((new_key, ", ".join(map(str, value))))
        elif value is None:
            items.append((new_key, "None"))
        else:
            items.append((new_key, str(value)))

    return dict(items)


def generate_tags(config: dict) -> list[str]:
    return [f"{key}: {value}" for key, value in flatten_config(config).items()]


def sanitize_wandb_config(config: dict) -> dict:
    sanitized = copy.deepcopy(config)
    model_class = sanitized["model"]["class"]

    if ("AutoEncoder" and not "WD") or "ResNet" not in model_class:
        sanitized["model"].pop("latent_dim", None)

    if sanitized["model"]["downsampling"] != "LPD":
        sanitized["model"].pop("gumbel_tau", None)

    if sanitized["optim"]["algo"] != "Adam":
        sanitized["optim"].pop("weight_decay", None)

    if sanitized["loss"]["name"] != "FocalLoss":
        sanitized["loss"].pop("gamma", None)

    if sanitized["loss"]["name"] != "ComplexVAELoss":
        sanitized["loss"].pop("kld_weight", None)

    if sanitized["model"]["projection"]["class"] != "NoCtoR":
        sanitized["model"]["projection"].pop("softmax", None)

    sanitized["logging"].pop("logdir", None)
    sanitized.pop("seed", None)
    sanitized.pop("pretrained", None)
    sanitized.pop("world_size", None)

    return sanitized


def _is_iterable_metric(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    if isinstance(value, torch.Tensor):
        return value.ndim > 0
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple)) or hasattr(value, "__iter__")


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flatten_distribution_values(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        array = np.asarray(
            list(value) if not isinstance(value, (list, tuple)) else value
        )

    return np.asarray(array).reshape(-1).astype(float).tolist()


def resolve_logging_backend(config: dict) -> str:
    logging_config = config.setdefault("logging", {})
    backend = os.environ.get(
        "EXPERIMENT_LOGGER_BACKEND", logging_config.get("backend", "aim")
    )
    backend = str(backend).strip().lower()
    if backend in {"none", "disabled", "off", "false"}:
        backend = "disabled"
    if backend not in {"aim", "wandb", "disabled"}:
        raise ValueError(f"Unsupported experiment logging backend: {backend}")
    logging_config["backend"] = backend
    return backend


def resolve_aim_repo(logging_config: dict) -> str:
    aim_config = logging_config.setdefault("aim", {})
    repo = os.environ.get("AIM_REPO", aim_config.get("repo"))
    if repo:
        repo = str(repo)
    elif Path("/data/equiv-cvnn").exists():
        repo = "/data/equiv-cvnn/aimlogs"
    else:
        repo = "./aim"
    aim_config["repo"] = repo
    return repo


class ExperimentLogger:
    backend_name = "disabled"

    def __bool__(self) -> bool:
        return self.is_enabled()

    def is_enabled(self) -> bool:
        return False

    def uses_legacy_wandb_api(self) -> bool:
        return False

    @property
    def run_name(self) -> str | None:
        return None

    @property
    def run_id(self) -> str | None:
        return None

    def log_config(self, config: dict) -> None:
        return None

    def log_summary(self, summary_text: str) -> None:
        return None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        return None

    def log_image(
        self,
        name: str,
        image_or_path: Any,
        caption: str | None = None,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        return None

    def log_figure(
        self,
        name: str,
        figure: Any,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        return None

    def finish(self) -> None:
        return None


class DisabledExperimentLogger(ExperimentLogger):
    backend_name = "disabled"


class WandbExperimentLogger(ExperimentLogger):
    backend_name = "wandb"

    def __init__(
        self,
        config: dict,
        tracked_config: dict | None = None,
        tags: list[str] | None = None,
    ):
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "WandB backend requested but the 'wandb' package is not installed."
            ) from exc

        self._wandb = wandb
        wandb_config = config.setdefault("logging", {}).setdefault("wandb", {})
        project_name = wandb_config["project"]
        entity_name = wandb_config["entity"]
        run_id = wandb_config.get("run_id")

        tracked_config = (
            sanitize_wandb_config(config) if tracked_config is None else tracked_config
        )
        tags = generate_tags(tracked_config) if tags is None else tags

        if config["pretrained"]:
            self._wandb.init(
                project=project_name,
                entity=entity_name,
                resume="must",
                id=run_id,
            )
        else:
            config_cp = copy.deepcopy(tracked_config)
            self._wandb.init(
                project=project_name,
                entity=entity_name,
                config=config_cp,
                tags=tags,
            )

        wandb_config["run_id"] = self.run_id
        logging.info("Will be recording in wandb run name: %s", self.run_name)

    def is_enabled(self) -> bool:
        return True

    def uses_legacy_wandb_api(self) -> bool:
        return True

    @property
    def run_name(self) -> str | None:
        run = getattr(self._wandb, "run", None)
        return getattr(run, "name", None)

    @property
    def run_id(self) -> str | None:
        run = getattr(self._wandb, "run", None)
        return getattr(run, "id", None)

    def log_config(self, config: dict) -> None:
        sanitized = sanitize_wandb_config(config)
        self._wandb.config.update(sanitized, allow_val_change=True)

    def log_summary(self, summary_text: str) -> None:
        self._wandb.log({"summary": summary_text})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        log_data = {}
        for key, value in metrics.items():
            value = _normalize_scalar(value)
            if _is_iterable_metric(value):
                log_data[key] = self._wandb.Histogram(
                    _flatten_distribution_values(value)
                )
            else:
                log_data[key] = value
        self._wandb.log(log_data, step=step)

    def log_image(
        self,
        name: str,
        image_or_path: Any,
        caption: str | None = None,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        image = self._wandb.Image(image_or_path, caption=caption)
        self._wandb.log({name: image}, step=step)

    def log_figure(
        self,
        name: str,
        figure: Any,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        if hasattr(figure, "to_html"):
            html = figure.to_html(include_plotlyjs="cdn")
            self._wandb.log({name: self._wandb.Html(html)}, step=step)
        else:
            self._wandb.log({name: figure}, step=step)

    def finish(self) -> None:
        self._wandb.finish()


class AimExperimentLogger(ExperimentLogger):
    backend_name = "aim"

    def __init__(
        self,
        config: dict,
        tracked_config: dict | None = None,
        tags: list[str] | None = None,
    ):
        try:
            from aim import Distribution, Figure, Image, Run, Text  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Aim backend requested but the 'aim' package is not installed."
            ) from exc

        self._Distribution = Distribution
        self._Figure = Figure
        self._Image = Image
        self._Run = Run
        self._Text = Text

        logging_config = config.setdefault("logging", {})
        aim_config = logging_config.setdefault("aim", {})
        repo = resolve_aim_repo(logging_config)
        experiment = aim_config.get("experiment", config["model"]["class"])
        run_kwargs = {"repo": repo, "experiment": experiment}
        if config.get("pretrained") and aim_config.get("run_hash"):
            run_kwargs["run_hash"] = aim_config["run_hash"]
        try:
            self._run = self._Run(**run_kwargs)
        except TypeError:
            run_kwargs.pop("run_hash", None)
            self._run = self._Run(**run_kwargs)
        aim_config["repo"] = repo
        aim_config["run_hash"] = self.run_id
        logging.info("Will be recording in Aim run hash: %s", self.run_name)
        self._tracked_config = copy.deepcopy(config) if tracked_config is None else copy.deepcopy(tracked_config)
        self._tags = [] if tags is None else list(tags)
        self._run["config"] = copy.deepcopy(self._tracked_config)
        self._run["config_flat"] = flatten_config(self._tracked_config)
        self._run["backend"] = self.backend_name
        self._run["tags"] = list(self._tags)
        add_tag = getattr(self._run, "add_tag", None)
        if callable(add_tag):
            for tag in self._tags:
                add_tag(tag)

    def is_enabled(self) -> bool:
        return True

    @property
    def run_name(self) -> str | None:
        return getattr(self._run, "hash", None)

    @property
    def run_id(self) -> str | None:
        return getattr(self._run, "hash", None)

    def log_config(self, config: dict) -> None:
        flattened = flatten_config(config)
        self._run["config"] = copy.deepcopy(config)
        self._run["config_flat"] = flattened
        self._run["backend"] = self.backend_name

    def _track(
        self,
        payload: Any,
        name: str,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        if context:
            try:
                self._run.track(payload, name=name, step=step, context=context)
                return
            except TypeError:
                pass
        self._run.track(payload, name=name, step=step)

    def log_summary(self, summary_text: str) -> None:
        try:
            self._run.track(self._Text(summary_text), name="summary")
        except Exception:
            self._run["summary"] = summary_text

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        for key, value in metrics.items():
            value = _normalize_scalar(value)
            if _is_iterable_metric(value):
                distribution = self._Distribution(_flatten_distribution_values(value))
                self._track(distribution, name=key, step=step)
            else:
                self._track(value, name=key, step=step)

    def _image_payload(self, image_or_path: Any) -> Any:
        if hasattr(image_or_path, "savefig") and hasattr(image_or_path, "canvas"):
            buffer = io.BytesIO()
            image_or_path.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            with PILImage.open(buffer) as image:
                return self._Image(image.copy())

        if isinstance(image_or_path, (str, os.PathLike, Path)):
            image_path = Path(image_or_path)
            if image_path.exists():
                with PILImage.open(image_path) as image:
                    return self._Image(image.copy())

        return self._Image(image_or_path)

    def log_image(
        self,
        name: str,
        image_or_path: Any,
        caption: str | None = None,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        payload = self._image_payload(image_or_path)
        if caption:
            context = {**(context or {}), "caption": caption}
        self._track(payload, name=name, step=step, context=context)

    def log_figure(
        self,
        name: str,
        figure: Any,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        if hasattr(figure, "savefig") and hasattr(figure, "canvas"):
            payload = self._image_payload(figure)
        else:
            payload = self._Figure(figure)
        self._track(payload, name=name, step=step, context=context)

    def finish(self) -> None:
        close = getattr(self._run, "close", None)
        if callable(close):
            close()


def build_experiment_logger(
    config: dict,
    tracked_config: dict | None = None,
    tags: list[str] | None = None,
) -> ExperimentLogger:
    backend = resolve_logging_backend(config)
    if backend == "aim":
        return AimExperimentLogger(config, tracked_config=tracked_config, tags=tags)
    if backend == "wandb":
        return WandbExperimentLogger(config, tracked_config=tracked_config, tags=tags)
    return DisabledExperimentLogger()
