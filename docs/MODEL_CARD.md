# TraceLens v2 model card

## Intended use

Academic/portfolio investigation of retrospective system-log anomalies and calibration drift. Default: three seeded word-level CNNs, selected on development data and calibrated on a separate time range. Not an operational outage predictor.

## Architecture and fitting

24-dimensional word embeddings; 24 Conv1D filters each at widths 3 and 5; GELU and token max pooling; a 32-unit head with dropout 0.15; event max pooling to produce a window logit. Binary cross-entropy uses window-level labels. Input is content only, after stripping the alert category and metadata including severity. Each event retains at most 16 word tokens.

Fit vocabulary and weights on BGL lines 1–600,000; select epochs on 600,001–1,000,000. Seeds 17/42/73; AdamW 0.001, weight decay 0.0001, batch size 128, clip norm 1, up to six epochs with patience two. Average three event activations before max pooling over the 20-event window. This is supervised modeling; source alert labels never enter inference features.

## Calibration and selection

See `configs/validation-v2.json` and `docs/CALIBRATION_AUDIT.md`. Development calibrates/selects models, scoring rules, and threshold policies on lines 1,000,001–2,000,000. Fit the final threshold on 2,000,001–2,500,000. The selected threshold is 0.2348803292, with strict score > threshold. F2 optimization is subject to a 1% normal-FPR calibration budget. This budget is not a temporal generalization guarantee.

## Evaluation

Frozen audit on lines 2,500,001–4,747,963, with three incomplete trailing records excluded: 112,398 windows, 7,439 positives. Combined recall 46.35%, precision 46.59%, normal FPR 3.765%, AP 0.3407; TP=3,448, FP=3,952, FN=3,991, TN=101,007.

On identical windows, the original frozen LSTM has recall 1.01%, precision 81.52%, normal FPR 0.0162%, AP 0.1035. Recalibrating that original model alone yields recall 12.84%, precision 9.07%, normal FPR 9.124%.

Confirmation-range recall is 74.22%, precision 69.21%, normal FPR 2.89%. Recall improves in all audit ranges, but the normal-FPR target fails in all three. Audit B has precision only 17.67%. All range results and paired intervals are retained.

## Limits and provenance

The source archive SHA-256 is d67fd82a711aea0157a9b83175892c6ee60e384a2ddf5bc51f39118453816da8. The freeze SHA-256 is 4728106480449bf6e31b737fa7983f0a85ab3b4c364cd85b35910d9285623fc2.

Coarse archive-wide class counts were inspected in v1; new audits were prediction-blind, not pristine label-blind. No models or thresholds changed after audit predictions were opened. One historical system, mixed-node windows, token truncation, and one deployment-calibration range limit generalization. CNN activations are not calibrated failure probabilities, and an alert is not root-cause evidence.

Use new data or an external dataset before further tuning. Preserve the separate code and data licensing terms. Frozen release copies remain separate from semantic training outputs. The prior model card is retained as MODEL_CARD_V1.md.

The first semantic freeze omitted an independent word-vocabulary checksum. Release review verified byte identity with the retained training mapping, recorded the release checksum, and disclosed that this is not a retroactive pre-audit binding. Original scoring-source snapshots are retained; future freezes require the checksum.
