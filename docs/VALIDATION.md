# TraceLens v2 release validation

Verified locally on October 18, 2026.

- **40 tests passed**, including all original tests, strict float32/float64 threshold boundary checks, brute-force agreement of budget optimization, word-vocabulary exclusion of future words, event/max score agreement, frozen checkpoint integrity, audit split separation, packed score/metric parity, real ensemble inference parity, and an explicit check that the failed FPR budget remains reported.
- JavaScript syntax checks passed for both the original and v2 interfaces.
- Original 13-variant, 39-real-sequence parity checks remain available through `scripts/check_reference_run.py`. v2 adds reloaded ensemble inference and split-specific count checks.
- Release artifact hashes are available through `python scripts/verify_artifacts.py`.
- Reviewed the new desktop overview, linked incident evidence, temporal range switching, detector comparison, validation page, and replay workflow.
- Live CNN replay reproduces a flagged audit example at score 0.9635; loading a .log file through the browser file chooser succeeds.
- Reviewed the 390x844 phone layouts for Observe and Replay, and checked Validate's contained table scrolling. Document width remains within the viewport. Restored the normal browser viewport.
- Confirmed all audit metrics remain frozen; the failed 1% normal-FPR target is shown prominently in the overview and write-up.

The runner emits a Starlette TestClient/httpx deprecation warning; tests pass. Local results are distinct from the GitHub Actions run; see the repository Checks workflow for its current result. Prior screenshots and v1 artifacts are retained for historical comparison. Narrative display encoding was repaired in the release report without modifying frozen.json, scores, labels, thresholds, or metrics.

Audits were opened once after freezing. They are now inspected data and cannot be reused as untouched holdouts for further optimization. The earlier LinkedIn post was deleted by the user. Publication is on hold; GitHub is the primary project record. See REVIEW.md for this iteration.

Release review also found that the first semantic freeze omitted an independent word-vocabulary hash. The word mapping matches the retained training copy byte-for-byte. The limitation is disclosed in CALIBRATION_AUDIT.md; the original freeze and exact scoring-source snapshots are preserved. Future freezes bind this mapping explicitly. No audit was reopened.

## Application language review

The current interface explains its purpose, window labels, metric denominators, dynamic detection/miss/false-alarm counts, chart aggregation, model comparisons, and sample-scoring limits in plain language. Navigation is Explore logs / Compare models / Try a sample / Method. Technical scoring rules, temporal boundaries, bootstrap assumptions, supervision differences, and the word-vocabulary freeze omission remain accessible. All 40 tests pass; the detector and saved evaluation artifacts are unchanged. Reviewed desktop and 390px phone layouts, switched detectors and ranges, inspected false positives and misses, and scored both a bundled example and unsupported input.


## Reproduce the experiments

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

The audit command verifies frozen hashes before scoring and refuses an in-place repeat after a report exists. A separate output directory makes reproduction explicit. Further tuning needs new evaluation data. These ranges have already been examined.


## Technical skills demonstrated

- Learned embeddings; recurrent LSTM state; causal dilated residual convolutions; word-level phrase convolutions; GELU; token and event max pooling.
- Minibatch backpropagation, AdamW, weight decay, dropout, clipped gradients, seeded training, and held-out epoch selection.
- Event-frequency baselines, three-seed sequence experiments, order/dropout ablations, supervised content modeling, calibration policy selection, and paired temporal evaluation.
- Tests for causality, label exclusion, train-only vocabulary, split separation, a finite-difference gradient check, strict/tied thresholds, budget optimization, packed artifact integrity, and live inference parity.

## Where to find the code

```text
tracelens/
  models.py          # original LSTM / causal temporal CNN
  semantic.py        # word-level content CNN and fitting
  calibration.py     # scoring rules, budgeted thresholds, paired bootstrap
  audit.py           # development freeze, then separate temporal audit
  api.py             # packed artifacts and real ensemble inference
  web/v2.*           # current log analysis interface
artifacts/
  portfolio/         # original experiments, preserved
  semantic/          # three content-CNN checkpoints and training histories
  development-v2/    # initial sequence-only development freeze, never audited
  validated/         # final frozen detector and audit evidence
configs/validation-v2.json
tests/
docs/
```
