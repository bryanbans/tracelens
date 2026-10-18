"""CPU-first reproducible benchmark with genuine checkpoints and test predictions."""
from __future__ import annotations

import copy
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import PARSER_VERSION, SOURCE, fit_vocab, read_bgl, save_json, sha256, split_records, windows
from .evaluation import bootstrap_ap, calibrate, metrics
from .models import SequenceModel, event_losses


def score_model(model, x, batch_size=256):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            parts.append(event_losses(model, torch.as_tensor(x[start:start + batch_size])).numpy())
    losses = np.concatenate(parts)
    return losses.mean(axis=1), losses


def fit_model(train, validation, vocab_size, config, seed, shuffled=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SequenceModel(vocab_size, **{k: config[k] for k in ("architecture", "embedding", "hidden", "dropout")})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train = train.copy()
    if shuffled:
        rng = np.random.default_rng(seed)
        for row in train:
            rng.shuffle(row[1:])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.as_tensor(train)), batch_size=config["batch_size"], shuffle=True, generator=generator)
    history, best, best_state, stale = [], float("inf"), None, 0
    for epoch in range(config["epochs"]):
        model.train()
        total, examples, gradient_norms = 0., 0, []
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = event_losses(model, batch).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            gradient_norms.append(float(norm))
            optimizer.step()
            total += loss.item() * len(batch)
            examples += len(batch)
        val_loss = float(score_model(model, validation)[0].mean())
        history.append({"epoch": epoch + 1, "train_loss": total / examples, "validation_loss": val_loss,
                        "gradient_norm_mean_before_clip": float(np.mean(gradient_norms))})
        print(f"  {config['architecture']} seed={seed} epoch={epoch + 1} train={total/examples:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best - 1e-5:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history


def run(data: Path, output: Path, config: dict):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get("threads", 4))
    records = read_bgl(data, config["max_lines"])
    splits = split_records(records)
    vocab = fit_vocab(splits["train"])
    size = config["window_size"]
    prepared = {key: windows(rows, vocab, size) for key, rows in splits.items()}
    xtrain, ytrain, _ = prepared["train"]
    xval, yval, _ = prepared["validation"]
    xtest, ytest, groups = prepared["test"]
    normal_train, normal_val = xtrain[ytrain == 0], xval[yval == 0]
    if not len(normal_train) or not len(normal_val):
        raise ValueError("Train and validation must each contain normal windows.")
    if len(np.unique(ytest)) < 2:
        raise ValueError("Test split needs both classes. Increase --max-lines or use the complete dataset.")
    # All train records inform event identities; only normal train windows fit model weights/counts.
    counts = np.bincount(normal_train[:, 1:].ravel(), minlength=len(vocab)).astype(float) + 1
    probability = counts / counts.sum()
    unigram_losses = -np.log(probability[xtest[:, 1:]])
    unigram_scores = unigram_losses.mean(axis=1)
    unigram_val = -np.log(probability[xval[:, 1:]]).mean(axis=1)
    base_threshold = calibrate(unigram_val, yval, config["normal_quantile"])
    base_metrics = metrics(ytest, unigram_scores, base_threshold)
    base_metrics["ap_block_bootstrap_95"] = bootstrap_ap(ytest, unigram_scores)
    runs = [{"id": "unigram", "name": "Event frequency", "architecture": "unigram", "seed": None,
             "metrics": base_metrics, "parameters": len(vocab), "history": [], "train_seconds": 0,
             "description": "Laplace-smoothed unigram; ignores order.", "checkpoint": None}]
    scores_by_id = {"unigram": unigram_scores}
    normal_calibration = {"unigram": sorted(unigram_val[yval == 0].tolist())}
    for experiment in config["experiments"]:
        for seed in config["seeds"]:
            name = experiment["id"]
            run_id = f"{name}-s{seed}"
            print(f"Training {run_id} on {len(normal_train)} normal windows", flush=True)
            options = {**config, **experiment}
            clock = time.perf_counter()
            model, history = fit_model(normal_train, normal_val, len(vocab), options, seed, experiment.get("shuffle_train", False))
            elapsed = time.perf_counter() - clock
            val_scores, _ = score_model(model, xval)
            scores, _ = score_model(model, xtest)
            threshold = calibrate(val_scores, yval, config["normal_quantile"])
            result = metrics(ytest, scores, threshold)
            result["ap_block_bootstrap_95"] = bootstrap_ap(ytest, scores)
            checkpoint_name = f"{run_id}.pt"
            model_config = {k: options[k] for k in ("architecture", "embedding", "hidden", "dropout")}
            torch.save({"state_dict": model.state_dict(), "model_config": model_config, "vocab_size": len(vocab)}, output / checkpoint_name)
            runs.append({"id": run_id, "name": experiment["name"], "architecture": options["architecture"], "seed": seed,
                         "metrics": result, "parameters": sum(p.numel() for p in model.parameters()),
                         "history": history, "train_seconds": elapsed, "checkpoint": checkpoint_name,
                         "description": experiment["description"], "model_config": model_config})
            scores_by_id[run_id] = scores
            normal_calibration[run_id] = sorted(val_scores[yval == 0].tolist())
    # Choose a representative checkpoint by normal-validation predictive loss only, never test AP.
    candidates = [r for r in runs if r["architecture"] != "unigram" and "shuffle" not in r["id"]]
    selected = min(candidates, key=lambda r: min(h["validation_loss"] for h in r["history"]))["id"]
    archive_hash = sha256(data)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "dataset": {"name": "Loghub · Blue Gene/L", "source": SOURCE, "sha256": archive_hash,
                          "lines": len(records), "sampling": "first N records in file order; not full-dataset benchmark",
                          "parser": PARSER_VERSION, "vocab_size": len(vocab), "window_size": size,
                          "splits": {k: {"records": len(splits[k]), "first_line": splits[k][0]["line"],
                                         "last_line": splits[k][-1]["line"], "windows": len(v[0]),
                                         "anomalous_windows": int(v[1].sum()),
                                         "discarded_tail_records": len(splits[k]) % size,
                                         "unknown_event_fraction": float((v[0] == 0).mean())} for k, v in prepared.items()}},
              "config": config, "selected_model": selected, "runs": runs,
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                              "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()},
              "total_seconds": time.perf_counter() - started,
              "protocol": "60/20/20 contiguous file-order split; non-overlapping windows; train-only vocabulary; normal-only weight fitting; normal-validation loss selects checkpoints; 99th percentile normal-validation threshold; strict > comparison; untouched test scoring; no point adjustment."}
    # Paired input order perturbation: same event counts, changed context; not an independent benchmark.
    rng = np.random.default_rng(1337)
    shuffled_test = xtest.copy()
    for row in shuffled_test:
        rng.shuffle(row[1:])
    for entry in runs:
        if entry["checkpoint"]:
            payload = torch.load(output / entry["checkpoint"], map_location="cpu", weights_only=True)
            model = SequenceModel(payload["vocab_size"], **payload["model_config"])
            model.load_state_dict(payload["state_dict"])
            shifted, _ = score_model(model, shuffled_test)
            entry["order_probe"] = {"mean_score_change": float((shifted - scores_by_id[entry["id"]]).mean()),
                                    "metrics": metrics(ytest, shifted, entry["metrics"]["threshold"])}
    save_json(output / "report.json", report)
    save_json(output / "vocab.json", vocab)
    save_json(output / "baseline.json", {"probability": probability.tolist()})
    save_json(output / "calibration.json", normal_calibration)
    # Derived templates, not raw source logs. Retain ground truth for evaluation, never inference.
    traces = [{"id": i, "start_line": g[0]["line"], "end_line": g[-1]["line"], "time": g[0]["time"],
               "label": int(ytest[i]), "events": [{"line": r["line"], "template": r["template"],
                 "token": int(xtest[i, j]), "known": bool(xtest[i,j] != 0), "label": r["label"]} for j, r in enumerate(g)],
               "scores": {key: float(value[i]) for key, value in scores_by_id.items()},
               }
              for i, g in enumerate(groups)]
    (output / "traces.json").write_text(json.dumps(traces, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected": selected, "lines": len(records), "runs": len(runs),
                      "seconds": report["total_seconds"]}, indent=2), flush=True)
    return report
