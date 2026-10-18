# TraceLens v2 specification

## Objective

Improve recall and study threshold calibration without tuning on the later audit predictions. Give TraceLens a distinct, polished interface that helps a viewer investigate model decisions and understand the cost of improved detection.

## Research and evaluation contract

Preserve `artifacts/portfolio` and the original specification as historical v1 evidence. The authoritative v2 procedure is `configs/validation-v2.json`; modeling details, stage boundaries, disclosures, and measured outcomes are in `CALIBRATION_AUDIT.md`.

Development can iterate before audit predictions are opened. Candidate selection must use development-only metrics. Deployment calibration must be separate from selection. Frozen checksums cover modeling/scoring code, feature vocabulary, checkpoints, archive, and thresholds. Audit A, Audit B, and confirmation use exactly the selected frozen detector. Report adverse results; do not adapt thresholds after observing audits.

The legacy original, recalibrated legacy, and full content-CNN ensemble are evaluated on the same windows. Differences in supervision and feature representation are explicit. Prior aggregate class-count inspection is disclosed; no claim of pristine label-blind holdouts is permitted.

## New model

A word-level multiple-instance CNN uses message content only. Learned embeddings feed convolution widths 3 and 5, GELU, token max pooling, and a dropout head. Event-wise max pooling supports window labels without requiring event-level training labels. Each message contributes its first 16 word tokens. Three seeds average per-event activations before window max pooling.

Epoch selection, F2 threshold optimization, and normal-FPR constraints follow the recorded protocol. CNN activations are scores, not calibrated outage probabilities. Source alert labels, severity, timestamps, and node metadata never enter input features.

## Interface design

Visual identity: graphite instrument surfaces, coral editorial accents, cyan signal traces, large sans-serif headlines with restrained serif emphasis, and monospace evidence. Avoid the previous fixed sidebar and repeating statistic-card grid.

- **Observe:** aggregate recall/precision/FPR, an explicit calibration-drift notice, a large trace canvas, audit-range controls, frozen detector comparisons, and a linked incident rail/event inspector.
- **Validate:** side-by-side same-window metrics, individual time-range results, paired recall-gain intervals, the calibration budget failure, and learning-protocol differences.
- **Replay:** a local log editor/file input, sample loading, real checkpoint selection and inference, event scores, validation errors, and analysis feedback.
- **Protocol:** chronological boundaries, feature/training decisions, freeze provenance, caveats, data attribution, and D802 mapping.

The instrument is fixed to audited thresholds; the release does not offer test-label-driven threshold retuning. Plot bins retain their maximum score and indicate any labeled anomaly in that bin. Exact evidence remains available in the inspector.

## Accessibility and behavior

Keyboard-operable navigation, filters, pagination, incident selection, file import, model selection, and inference. Clear focus indicators, labeled controls, understandable loading/empty/error/success states, reduced-motion support, and readable evidence typography. On phones, navigation wraps and the incident rail scrolls horizontally above the inspector. Wide comparison tables scroll within their own containers. Treat message content as text, not markup.

## Delivery acceptance

1. Models and thresholds are frozen before opening audit predictions; the freeze and unfavorable results are retained.
2. Recall, precision, AP, normal FPR, and confusion matrices are calculated on identical windows for all comparators.
3. Original and new artifact scores agree with live inference within floating-point tolerance.
4. Tests verify threshold boundary semantics, budget optimization, source/checkpoint integrity, split filtering, data packing, and model inference.
5. Review desktop and phone layouts and exercise core interactions in the browser.
6. Update README, model card, audit write-up, screenshots, and LinkedIn copy. Publish only factual claims supported by the audited release.

## Observed release outcome

Recall improves across all three audit ranges, but the 1% calibration normal-FPR target fails on all three. The release must present both findings. Further optimization requires new evaluation data; these audits are now inspected.
