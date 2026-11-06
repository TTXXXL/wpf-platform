"""Fixed unit-marginal quantile protocol, processing and loss primitives."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEVELS = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
MEDIAN = 4
PROTOCOL = "unit_quantiles_v1"


def check_levels(levels):
    levels = np.asarray(levels, dtype=np.float64)
    if levels.shape != LEVELS.shape or not np.array_equal(levels, LEVELS):
        raise ValueError("quantile_levels must exactly match the ordered nine-level protocol")
    return levels


def check_grid(issue_times, entities, raw):
    times = np.asarray(issue_times)
    if times.dtype.kind != "M" or times.shape != (raw.shape[0],) or np.isnat(times).any():
        raise ValueError("issue_times must be a finite datetime64 vector matching I")
    times = times.astype("datetime64[us]").astype(np.int64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("issue_times must be strictly increasing")
    entities = np.asarray(entities, dtype=str)
    if entities.shape != (raw.shape[2],) or len(set(entities.tolist())) != len(entities):
        raise ValueError("Entity order must be a unique vector matching N")
    return times, entities


def check_quantiles(values, shape=None, monotone=False):
    values = np.asarray(values)
    if values.ndim != 4 or values.shape[-1] != len(LEVELS):
        raise ValueError(f"Quantiles must have shape [I,H,N,9], got {values.shape}")
    if shape is not None and values.shape != tuple(shape):
        raise ValueError(f"Quantiles have shape {values.shape}, expected {tuple(shape)}")
    if not np.isfinite(values).all():
        raise ValueError("Quantiles must be finite at every output position")
    if monotone and np.any(np.diff(values, axis=-1) < 0):
        raise ValueError("Monotone-head output contains crossing quantiles")
    return values


def postprocess(raw, monotone=False):
    raw = check_quantiles(raw, monotone=monotone)
    return np.maximum(np.sort(raw, axis=-1), 0.0)


def masked_pinball(pred, truth, mask, rated_kw):
    """Return [valid cells,Q] losses; index first so masked NaN never participates."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != truth.shape + (len(LEVELS),) or mask.shape != truth.shape:
        raise ValueError("Pinball prediction, truth and mask shapes disagree")
    rated = np.broadcast_to(np.asarray(rated_kw, dtype=np.float64), truth.shape)
    if not np.isfinite(rated).all() or np.any(rated <= 0):
        raise ValueError("rated_kw must be finite and positive")
    valid = mask & np.isfinite(truth)
    if not valid.any():
        raise ValueError("No valid targets for Pinball Loss")
    selected = pred[valid]
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite prediction on a valid target")
    error = (truth[valid, None] - selected) / rated[valid, None]
    return np.maximum(LEVELS * error, (LEVELS - 1) * error)


def torch_pinball(pred, truth, mask):
    """Mean loss on capacity-normalized targets, with an explicit |M| * Q divisor."""
    import torch
    if tuple(pred.shape) != tuple(truth.shape) + (len(LEVELS),) or mask.shape != truth.shape:
        raise ValueError("Pinball prediction, truth and mask shapes disagree")
    valid = mask.bool() & torch.isfinite(truth)
    if not bool(valid.any()):
        raise ValueError("No valid targets for Pinball Loss")
    selected = pred[valid]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Nonfinite prediction on a valid target")
    error = truth[valid].unsqueeze(-1) - selected
    tau = pred.new_tensor(LEVELS)
    return torch.maximum(tau * error, (tau - 1) * error).sum() / (valid.sum() * len(LEVELS))


def ordered_head(raw):
    """Base plus cumulative ReLU increments; exact ties are permitted."""
    import torch
    return torch.cat((raw[..., :1], raw[..., :1] + raw[..., 1:].relu().cumsum(-1)), dim=-1)


def save_quantiles(path, raw, issue_times, entities, metadata, monotone=False):
    """Write raw and scored forecasts together, with a required semantic identity."""
    path = Path(path)
    raw = check_quantiles(np.asarray(raw, dtype=np.float32), monotone=monotone)
    times, entities = check_grid(issue_times, entities, raw)
    required = {"task_sha256", "dataset", "target", "eval_mask", "normalization", "model"}
    if required - set(metadata):
        raise ValueError("Probability identity missing: " + ", ".join(sorted(required - set(metadata))))
    identity = {**metadata, "probabilistic": True, "shape": ["I", "H", "N", "Q"],
                "protocol": PROTOCOL, "lower_clip": 0.0, "upper_clip": None,
                "quantile_levels": LEVELS.tolist(), "raw_predictions_saved": True,
                "monotone_head": bool(monotone)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, predictions_raw=raw, predictions=postprocess(raw, monotone),
                            issue_times=times, entities=entities, quantile_levels=LEVELS,
                            metadata_json=np.asarray(json.dumps(identity, sort_keys=True)))
    return identity


def load_quantiles(path, issue_times, entities, shape, expected_metadata):
    with np.load(path, allow_pickle=False) as saved:
        keys = {"predictions_raw", "predictions", "issue_times", "entities", "quantile_levels", "metadata_json"}
        if keys - set(saved.files):
            raise ValueError("Incomplete probability file: " + ", ".join(sorted(keys - set(saved.files))))
        check_levels(saved["quantile_levels"])
        times = np.asarray(issue_times).astype("datetime64[us]").astype(np.int64)
        if not np.array_equal(saved["issue_times"], times):
            raise ValueError("issue_times differ from the task's scoring grid")
        if saved["entities"].tolist() != list(entities):
            raise ValueError("entities differ from the task entity order")
        metadata = json.loads(str(saved["metadata_json"].item()))
        if (metadata.get("protocol") != PROTOCOL or metadata.get("upper_clip") is not None
                or metadata.get("lower_clip") != 0.0 or not metadata.get("raw_predictions_saved")
                or metadata.get("probabilistic") is not True):
            raise ValueError("Unsupported probability protocol or postprocessing metadata")
        check_levels(metadata.get("quantile_levels"))
        for name, value in expected_metadata.items():
            if metadata.get(name) != value:
                raise ValueError(f"Probability identity differs: {name}")
        raw = check_quantiles(saved["predictions_raw"], shape, metadata.get("monotone_head", False)).copy()
        pred = check_quantiles(saved["predictions"], shape, monotone=True).copy()
        if not np.array_equal(pred, postprocess(raw, metadata.get("monotone_head", False))):
            raise ValueError("Saved scored quantiles do not match the registered postprocessing")
    return raw, pred, metadata
