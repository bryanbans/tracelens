# TraceLens

**Find unusual system-log messages. Inspect the evidence. Measure the false-alarm cost.**

TraceLens is a deep learning workbench for investigating anomalies in system logs. It scores groups of log messages, lets you inspect the messages behind a flagged window, and measures how many labeled anomalies the detector catches or misses. You can compare models, examine false positives, and replay messages through trained models on your own machine.

The project asks a practical question: **can a model catch more abnormal log behavior while keeping false alarms manageable as the logs change?**

Built after D802 Deep Learning in a computer science master's program in AI/ML, TraceLens explores how neural networks learn from machine-generated text and how those models should be evaluated before anyone relies on their alerts.

![TraceLens v2 observatory](docs/images/app-language-overview.jpg)

## The problem

A system log can contain millions of repetitive messages with a small number of unusual events mixed in. Searching for an error keyword is a useful starting point, but it does not tell you how many relevant events you missed, how many alerts were false alarms, or whether a detector still works on later logs.

TraceLens brings detection, inspection, and evaluation into one workspace. It uses real logs from the Blue Gene/L supercomputer to make those questions concrete. Its current scope is historical log analysis and model evaluation; applying it to another system requires training and validation for that system.

## What questions does TraceLens answer?

| Question | How TraceLens helps |
| --- | --- |
| **Which groups of messages does the model flag?** | Browse scores over the log file and open the exact 20-message window behind a decision. |
| **Which messages received the strongest model response?** | Inspect per-message CNN activations alongside the message content. These show model responses, not proven causes. |
| **What did the detector miss, and which alerts were false?** | Filter the labeled evaluation data into flagged windows, missed anomalies, and false positives. |
| **Does changing the model help more than changing the threshold alone?** | Compare the original LSTM, the same LSTM with a recalibrated threshold, and the word CNN ensemble on identical evaluation windows. |
| **Does a threshold keep working on later logs?** | Compare recall, precision, and false-positive rates across three later sections of the log. |
| **How does the trained model respond to a supplied sample?** | Paste or import messages and run local inference, with word-coverage, truncation, and unsupported-input feedback. |

## Try the investigation

Trained models and evaluation results are included, so you can explore the project without training a model first.

1. **Explore logs:** select a flagged window and inspect its messages. One sample contains repeated corrected-parity messages followed by a machine-check interrupt; the inspector shows how differently the CNN responds to each message.
2. **Investigate mistakes:** switch to **Missed anomalies** or **False positives**. Compare the model's response with the dataset labels.
3. **Compare models:** compare the three detectors and see how performance changes across later log ranges.
4. **Try a sample:** load an example or import a sample and run the saved models locally. Pasted logs have no ground truth, so replay reports scores and input checks rather than accuracy.
5. **Method:** inspect the training boundaries, threshold selection, and recorded evaluation procedure.

## What the experiment found

The word CNN catches more labeled anomalies than the original LSTM on the same later log windows, but it also creates more false alarms.

The comparison covers **112,398 non-overlapping windows of 20 messages**, including **7,439 labeled anomalous windows**. A window is labeled anomalous if it contains at least one dataset-labeled abnormal message.

| Detector | Anomalous windows caught (recall) | Flagged windows that were anomalous (precision) | Normal windows incorrectly flagged |
| --- | ---: | ---: | ---: |
| Original LSTM | 1.01% | 81.52% | 0.0162% |
| Same LSTM, recalibrated threshold | 12.84% | 9.07% | 9.124% |
| Word CNN ensemble | 46.35% | 46.59% | 3.765% |

In counts, the CNN finds **3,448 of 7,439 anomalous windows**, misses **3,991**, and incorrectly flags **3,952 normal windows**. Its higher recall comes with a substantial review burden.

**The threshold met its 1% normal false-positive constraint during calibration, then exceeded it in every later evaluation range.** TraceLens makes that change visible. This is the central finding: better detection on one measure does not by itself make a detector reliable enough to deploy.

The recall improvement appears in all three later ranges. A post-hoc sensitivity check finds positive recall-difference intervals across four bootstrap block lengths, while the false-positive increase persists. See the [complete evaluation](docs/CALIBRATION_AUDIT.md) for per-range results, average precision, confusion matrices, and uncertainty estimates, and the [review](docs/REVIEW.md) for the sensitivity analysis and remaining work.

## How the models work

TraceLens compares two ways to learn from logs:

- **Sequence models:** an LSTM and a causal temporal CNN learn to predict the next event identity from normal training windows. Unexpected events receive higher surprise scores.
- **Message-content model:** a word-level CNN learns phrase patterns using labeled training windows. Three independently seeded models average their per-message activations; the highest averaged activation becomes the window score. It can recognize familiar words within a new event template, but does not model the order of messages across a window.

Only message content enters the models. Alert-category labels, severity, timestamps, and node metadata are excluded from input features. Training, model selection, threshold calibration, and later evaluation use separate file ranges. Model weights and thresholds were fixed before opening the later evaluation predictions.

The [deep learning notes](docs/DEEP_LEARNING_NOTES.md) explain the computation graphs, objectives, and design tradeoffs.

## Run locally

Python 3.11+; verified on Python 3.12 / Windows / CPU PyTorch. No cloud model API, GPU, or frontend build is required. Trained checkpoints and derived evaluation artifacts are bundled.

```shell
git clone https://github.com/bryanbans/tracelens.git
cd tracelens
```

On Windows, run `Launch TraceLens.cmd` to create/install a project-local environment on first launch. Or use:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
tracelens serve
```

On macOS/Linux, activate with `source .venv/bin/activate`. Open **http://127.0.0.1:8765** and keep the server running. The default serves `artifacts/validated`; `tracelens serve --artifacts artifacts/portfolio` opens the preserved v1 app/results. The exact verified environment is recorded in `requirements-lock.txt` (Windows CPU wheel resolution may differ elsewhere).

## Reproduce and verify

```powershell
tracelens download --output data/BGL.zip
# Optional: refit content CNNs. This explicit flag replaces artifacts/semantic;
# frozen release checkpoints in artifacts/validated stay separate.
python scripts/train_semantic.py --data data/BGL.zip --overwrite
python -m tracelens.audit develop --data data/BGL.zip --output artifacts/reproduced-audit
# Inspect the freeze before opening audit predictions:
python -m tracelens.audit audit --data data/BGL.zip --output artifacts/reproduced-audit
python scripts/compact_audit.py --artifacts artifacts/reproduced-audit
python -m pytest -q
node --check tracelens/web/v2.js
```

The source BGL archive is approximately 57.5 MB compressed. Raw logs are not bundled. Checkpoints, the frozen procedure, archive hash, all audit scores, paired intervals, and confusion matrices are retained. The release packs templates and exact numerical scores in `traces.json.gz`; packing does not alter evaluation decisions. `scripts/compact_audit.py` reproduces the release packing for the default validated directory from its raw audit export.

The audit command verifies frozen hashes before scoring and refuses an in-place repeat after a report exists. A separate output directory makes reproduction explicit. Subsequent tuning needs new holdouts; these audited ranges are now inspected.

## Deep learning evidence

- Learned embeddings; recurrent LSTM state; causal dilated residual convolutions; word-level phrase convolutions; GELU; token and event max pooling.
- Minibatch backpropagation, AdamW, weight decay, dropout, clipped gradients, seeded training, and held-out epoch selection.
- Event-frequency baselines, three-seed sequence experiments, order/dropout ablations, supervised content modeling, calibration policy selection, and paired temporal evaluation.
- Tests for causality, label exclusion, train-only vocabulary, split separation, a finite-difference gradient check, strict/tied thresholds, budget optimization, packed artifact integrity, and live inference parity.

## Source map

```text
tracelens/
  models.py          # original LSTM / causal temporal CNN
  semantic.py        # word-level content CNN and fitting
  calibration.py     # scoring rules, budgeted thresholds, paired bootstrap
  audit.py           # development freeze, then separate temporal audit
  api.py             # packed artifacts and real ensemble inference
  web/v2.*           # signal observatory UI
artifacts/
  portfolio/         # original experiments, preserved
  semantic/          # three content-CNN checkpoints and training histories
  development-v2/    # initial sequence-only development freeze, never audited
  validated/         # final frozen detector and audit evidence
configs/validation-v2.json
tests/
docs/
```

## Reading and portfolio material

- [Calibration audit and measured v2 findings](docs/CALIBRATION_AUDIT.md)
- [Current model card](docs/MODEL_CARD.md)
- [v2 specification](docs/SPEC_V2.md)
- [Deep learning mechanics](docs/DEEP_LEARNING_NOTES.md)
- [Release validation](docs/VALIDATION.md)
- [Review findings and next experiments](docs/REVIEW.md)
- [Demonstration guide](docs/PORTFOLIO.md)
- [Historical v1 case study](docs/CASE_STUDY.md) and [v1 README](docs/README_V1.md)
- [Data attribution and terms](THIRD_PARTY_NOTICES.md)

## Limits

BGL is historical supercomputer data. File-order windows may mix nodes. Labels indicate retrospective alert messages, not future outages. The word model truncates events to 16 tokens; activations are not calibrated outage probabilities. Coarse class counts across the archive were previously inspected, so the new ranges are prediction-blind temporal audits, not pristine label-blind external replication. The 1% normal-FPR target was missed on every audit range. No audit-driven retuning was used to hide that result.

The CNN uses labeled training windows, while the original sequence models train on normal windows only. The comparison combines changes in representation, supervision, model, and calibration. A similarly supervised simple text-classification baseline is still needed to isolate the value of deep learning. Further tuning needs new evaluation data.

The first CNN freeze omitted an independent word-vocabulary hash. The retained mapping matches the training copy, and release checksums now protect it, but that cannot retroactively supply the missing pre-evaluation commitment. See the full disclosure in the [evaluation report](docs/CALIBRATION_AUDIT.md).

The app binds to loopback and processes input locally. Its inference form is for this learned domain; other log formats require their own training and validation. Project code is MIT licensed; BGL-derived artifacts retain Loghub's separate research/academic terms and attribution requirements.
