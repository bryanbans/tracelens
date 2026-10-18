# TraceLens v2: recall improves; calibration drift remains

## The question

Can TraceLens improve recall without presenting a more permissive threshold as a universal improvement? This release separates representation/model changes from recalibration, locks the procedure before audit predictions, and reports both detections and false alarms.

## What changed

The original event-identity LSTM maps every unfamiliar normalized message to one unknown event. The new content CNN learns word embeddings and phrase filters, allowing parts of previously unseen messages to share useful features with training messages.

The CNN uses 24-dimensional embeddings, 24 filters each at kernel widths 3 and 5, GELU, token-wise max pooling, a 32-unit head, dropout 0.15, and event-wise max pooling. It receives 16 word tokens per event in a 20-event window. It trains against window-level alert labels; no per-event alert label enters the features or event-level training target. Three seeded models average event activations; the maximum averaged activation is the window score.

Training uses lines 1–600,000. Epoch selection uses lines 600,001–1,000,000. Seeds are 17/42/73. AdamW uses learning rate 0.001, weight decay 0.0001, batch size 128, gradient clipping at one, up to six epochs and patience two. Training histories and checkpoint weights are retained in `artifacts/semantic`.

This is supervised content modeling, whereas the original sequence model fitted only normal examples. The complete change includes representation, learning protocol, ensembling, aggregation, and calibration. The controlled recalibration-only comparator helps avoid attributing the entire improvement to a threshold change.

## Selection and freeze

The chronological development procedure is in `configs/validation-v2.json`:

| Stage | Source lines | Use |
| --- | --- | --- |
| Development calibration | 1,000,001–1,500,000 | Fit candidate thresholds |
| Development selection | 1,500,001–2,000,000 | Select family, score rule, and policy |
| Deployment calibration | 2,000,001–2,500,000 | Fit the selected policy's final threshold |
| Audit A | 2,500,001–3,250,000 | Frozen evaluation |
| Audit B | 3,250,001–4,000,000 | Frozen evaluation |
| Confirmation | 4,000,001–4,747,963 | Frozen evaluation; three tail records excluded |

Sequence candidates included mean, top-three mean, or maximum event surprise; word CNN candidates used maximum event activation. Both normal-quantile and labeled F2 policies were considered. The selection objective was highest development-selection F2 subject to at most 1% normal FPR; an explicit fallback would declare infeasibility if no candidate met the selection constraint.

Development first considered sequence-only candidates. When their development recall remained weak, the content CNN was added **before audit predictions were opened**. The initial sequence-only freeze is preserved in `artifacts/development-v2`. The final freeze selected `semantic-max-f2_budget`.

The F2 policy searches exact score boundaries with strict `score > threshold`, maximizing F2 on calibration labels while limiting calibration normal FPR to 1%. The selected deployment threshold was **0.2348803292**. Deployment-calibration recall was 30.65%, precision 62.89%, and normal FPR 0.1452%. Those values are calibration fit, not generalization results.

Before opening the audit, `frozen.json` recorded the procedure, thresholds, archive/vocabulary/checkpoint hashes, and modeling/scoring source hashes. Its SHA-256 is **4728106480449bf6e31b737fa7983f0a85ab3b4c364cd85b35910d9285623fc2**. The audit verifies these hashes and refuses an in-place repeat if a report already exists. Audit predictions were opened once; no thresholds or models were changed afterward.

Coarse class counts across the archive were inspected during v1 development. These are therefore prediction-blind temporal audits, **not pristine label-blind or external holdouts**. The original 10,000-window v1 test is historical development evidence; its old 0.850 AP must not be compared with v2's AP on different time ranges as if the datasets were identical.

## Same-window comparison

The combined audit contains **112,398 non-overlapping windows**, including **7,439 anomalous** and **104,959 normal** windows.

| Detector | Recall | Precision | Normal FPR | Average precision | TP | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original frozen LSTM | 1.01% | 81.52% | 0.0162% | 0.1035 | 75 | 17 |
| Original LSTM, recalibration only | 12.84% | 9.07% | 9.124% | 0.1035 | 955 | 9,577 |
| Word CNN ensemble + selected calibration | 46.35% | 46.59% | 3.765% | 0.3407 | 3,448 | 3,952 |

The full detector finds **3,373 additional anomalous windows** relative to the original, at the cost of **3,935 additional false-positive windows**. It also has a better recall/false-positive trade-off than recalibrating the original detector alone on these ranges.

The 1% calibration target does **not** hold across the audit ranges. That is an observed failure, not something the project hides by changing the threshold after seeing the result.

### Provenance limitation found during release review

The pre-audit freeze hashed the legacy event vocabulary but omitted an independent checksum for the new word vocabulary. The word vocabulary was copied before audit execution; release review verified that it remains byte-identical to the retained training vocabulary. The release manifest now hashes it. This does **not** retroactively provide a pre-audit cryptographic binding for that mapping. The original freeze is preserved unmodified, with the exact scoring-source files matching its recorded hashes under `artifacts/validated/frozen_sources`. Future development freezes now require and verify the word-vocabulary checksum. No audit scores, weights, thresholds, or labels were changed to address this documentation/provenance issue.

## Results over time

| Range | Original recall | CNN recall | CNN precision | CNN normal FPR | Paired recall-gain 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Audit A | 1.98% | 41.04% | 58.15% | 1.38% | +17.5 to +62.7 percentage points |
| Audit B | 1.34% | 19.17% | 17.67% | 7.10% | +11.4 to +27.1 percentage points |
| Confirmation | 0.17% | 74.22% | 69.21% | 2.89% | +60.3 to +86.8 percentage points |

Recall improves in all three ranges, including the confirmation range. False alarms vary substantially; Audit B remains weak. The paired moving-block bootstrap uses 250 consecutive windows per block and 500 resamples with seed 2026. Intervals are conditional/descriptive for these ranges; they do not remove longer-range dependence or historical dataset bias. False-positive differences and all confusion matrices are available in `report.json`.

## What this supports

The measured recall and ranking gains are real on these fixed audit windows, and are not created by tuning to the audit labels. The project demonstrates content representation, pooling, supervised learning, sequence baselines, calibration under a budget, and honest temporal evaluation.

It does **not** support saying the detector meets a 1% production false-positive rate, predicts future outages, or generalizes to all service logs. The next calibration improvement needs new data or a newly reserved external dataset; the current audits are now inspected and must not be reused as an untouched tuning target.

## Reproduction

From the repository root after installing the development dependencies:

```powershell
tracelens download --output data/BGL.zip
# Original and word-CNN checkpoints are bundled. Optionally reproduce semantic fitting;
# --overwrite explicitly replaces artifacts/semantic, not the frozen release copies.
python scripts/train_semantic.py --data data/BGL.zip --overwrite
python -m tracelens.audit develop --data data/BGL.zip --output artifacts/reproduced-audit
python -m tracelens.audit audit --data data/BGL.zip --output artifacts/reproduced-audit
tracelens serve --artifacts artifacts/validated
python -m pytest -q
```

For a fresh audit directory, `develop` and `audit` are separate commands so the freeze is reviewable before opening audit predictions. The release UI packs derived templates and exact scores into `traces.json.gz` without changing numerical results. The raw archive remains separate under Loghub's research/academic terms.
