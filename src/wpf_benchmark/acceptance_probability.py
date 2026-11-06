"""Bounded real-data Toto acceptance: twelve short cases, no research batch or model ranking."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .data.store import load_meta, load_panel, verify_dataset
from .evaluation import write_json
from .models.toto_quantile import TotoNativeQuantiles
from .models import get_model
from .provenance import code_digest, environment, sha256_file
from .runner import TaskData, make_context, seed_all
from .tasks import build_split, resolve_task
from .tasks.probabilistic import probability_task, window_audit


def run_toto_smoke(datasets, model_dir, out, device="cuda"):
    import torch
    from .models.mixture_quantiles import mixture_quantiles
    from .models.neural.trainer import load_checkpoint
    from .probability import LEVELS
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries
    from .models.toto_quantile import pack_toto

    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("The requested smoke device is unavailable")
    if device != ("cuda" if torch.cuda.is_available() else "cpu"):
        raise ValueError("The shared trainer selects the default available device; run smoke on that device")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Smoke output must be empty; existing evidence is preserved")
    identities = {farm: verify_dataset(Path(datasets) / farm) for farm in ("kelmarsh", "penmanshiel")}
    out.mkdir(parents=True, exist_ok=True)
    records, native_records = [], []
    write_json(out / "smoke_status.json", {"status": "running", "data": identities, "formal_experiments": False})
    rng = np.random.default_rng(0)
    for farm in ("kelmarsh", "penmanshiel"):
        directory = Path(datasets) / farm
        meta = load_meta(directory)
        panel = load_panel(directory, ["wind_speed", "wind_dir_sin", "wind_dir_cos", "power"],
                           [m for m in ("observed", "normal_b", "normal_c") if m in meta["masks"]],
                           start="2020-12-01", end="2021-01-04")
        for L in (144, 576):
            cfg = probability_task(meta, "2021-01-01", L=L)
            # Explicit short software-only range, detached from any official testing protocol.
            cfg = replace(cfg, name=f"smoke_{farm}_{L}",
                          splits={"train": ("2020-12-01", "2020-12-18"),
                                  "valid": ("2020-12-18", "2021-01-01"),
                                  "test": ("2021-01-01", "2021-01-04")})
            task = resolve_task(cfg, meta)
            splits = [build_split(task, panel, s) for s in ("train", "valid", "test")]
            for s, n in zip(splits, (8, 4, 2)):
                s.issues = s.issues[:n]
            data = TaskData(task, *splits, identities[farm])
            ctx = make_context(data)
            batch = data.test.batch(data.test.issues)
            if L == 144:
                native = TotoNativeQuantiles(model_dir=str(model_dir), device=device, batch=1)
                native.fit(ctx, data.train, data.valid)
                raw = native.predict_quantiles(batch)
                mean = native.native_mean.copy()
                np.testing.assert_array_equal(raw, native.predict_quantiles(batch))
                xt = torch.as_tensor(batch.x_past, dtype=torch.float32, device=native.device)
                series, observed, ids = pack_toto(xt, native.toto.model.patch_embed.stride)
                native_batch = MaskedTimeseries(series, observed, ids,
                    torch.zeros_like(series, dtype=torch.int64), torch.full(series.shape[:2],600,device=device,dtype=torch.int64),0)
                reference = TotoForecaster(native.toto.model).forecast(native_batch, prediction_length=36,
                                                                         num_samples=None, use_kv_cache=False)
                target = torch.arange(len(ctx.entities), device=device)*4 + 3
                reference = reference.mean[:, target].permute(0,2,1).cpu().numpy()
                np.testing.assert_allclose(mean, reference, rtol=1e-5, atol=1e-3)
                with torch.no_grad():
                    parameters = native._distribution_parameters(xt[:1])
                errors = []
                for h in (0,35):
                    w,df,loc,scale = [a[0,h,0].astype(float) for a in parameters]
                    w /= w.sum()
                    q = mixture_quantiles(w,df,loc,scale)
                    component = rng.choice(len(w),size=10000,p=w)
                    sampled = loc[component]+scale[component]*rng.standard_t(df[component])
                    rank_error = np.abs((sampled[:,None] <= q).mean(0)-LEVELS)
                    if np.max(rank_error) >= 0.03:
                        raise ValueError("Real-subset native CDF sampling check failed")
                    difference = np.abs(np.quantile(sampled,LEVELS)-q)
                    errors.append({"horizon":h+1,"max_CDF_rank_error":float(rank_error.max()),
                                   "absolute_difference_kW_by_tau":difference.tolist(),
                                   "max_absolute_difference_kW":float(difference.max()),
                                   "mean_absolute_difference_kW":float(difference.mean())})
                native_records.append({"farm":farm,"native_mean_max_difference_kW":float(np.max(np.abs(mean-reference))),
                                       "repeated_quantiles_equal":True,"sampling_count":10000,"sampling_seed":0,
                                       "sampling_comparisons":errors,"identity":native.probability_metadata})
                del native,reference,raw,xt,series,observed,ids,native_batch
            for mode in ("head", "last4", "full"):
                seed_all(0)
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                started = time.perf_counter()
                model = get_model("toto1_wpf_q")(model_dir=str(model_dir), adaptation=mode,
                                                epochs=1, batch=1, patience=1)
                model.fit(ctx, data.train, data.valid)
                prediction = model.predict_quantiles(batch)
                if not np.isfinite(prediction).all() or np.any(np.diff(prediction,axis=-1) < 0):
                    raise ValueError("Toto-Q smoke produced invalid quantiles")
                checkpoint_ok = None
                if L == 144 and mode == "head":
                    checkpoint = out / f"{farm}_head_checkpoint.pt"
                    model.save_checkpoint(checkpoint)
                    loaded = load_checkpoint(checkpoint,task,device=device)
                    np.testing.assert_allclose(loaded.predict_quantiles(batch),prediction,atol=1e-3,rtol=1e-5)
                    checkpoint_ok = True
                    del loaded
                if device == "cuda":
                    torch.cuda.synchronize()
                record = {"farm":farm,"L":L,"H":36,"adaptation":mode,"device":device,
                          "epochs":1,"train_windows":8,"valid_windows":4,"prediction_windows":2,
                          "total_seconds_including_load":time.perf_counter()-started,
                          "epoch_seconds":model.probability_metadata["epoch_seconds"],
                          "peak_cuda_allocated_MiB":torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None,
                          "total_parameters":model.n_params,
                          "trainable_parameters":model.probability_metadata["trainable_parameters"],
                          "trainable_names_sha256":model.probability_metadata["trainable_names_sha256"],
                          "shape":list(prediction.shape),"checkpoint_roundtrip":checkpoint_ok,
                          "windows":[window_audit(task,s) for s in splits]}
                records.append(record)
                write_json(out / "smoke_status.json", {"status":"running","cases":records,"native":native_records})
                del model,prediction
    result = {"status":"passed","data":identities,"device":device,"cases":records,"native":native_records,
              "code_sha256":code_digest(),"environment":environment(),
              "checkpoint_sha256":sha256_file(Path(model_dir)/"model.safetensors"),
              "not_started":["B1-M","B1-Z","B1-F","B1-I"],"formal_experiments":False}
    write_json(out / "throughput_smoke.json", result)
    write_json(out / "smoke_status.json", {"status":"passed","cases":len(records)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",required=True)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args = parser.parse_args()
    run_toto_smoke(args.datasets,args.model_dir,args.out,args.device)


if __name__ == "__main__":
    main()
