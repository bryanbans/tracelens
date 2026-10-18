# TraceLens — project specification

**Product:** A research workbench that detects and investigates abnormal system-log windows using deep sequence models.

**Portfolio story:** After D802, I became interested in whether event order helps neural networks distinguish abnormal system behavior from merely unfamiliar messages. I built TraceLens to compare learned sequence models with an order-independent baseline and inspect their successes and failures.

## Research questions

1. Do an LSTM and a causal temporal CNN improve anomaly ranking over a smoothed event-frequency baseline?
2. Does shuffling training order change detection behavior when event counts remain fixed?
3. Does removing dropout change generalization under the same seed and training budget?
4. How stable is the alert operating point when event distributions change over time?

## Scope and deliverables

- A local web application with test-stream visualization, adjustable validation-based alert calibration, searchable/paginated windows, ground-truth comparison, and per-event surprise decomposition.
- A model lab showing every seed and experiment; a live inference page accepting log text/files; a methodology and provenance page.
- A Python package, reproducible CLI, configuration files, saved neural checkpoints, report JSON, tests, CI, and a documented data acquisition path.
- A technical case study and honest LinkedIn/GitHub copy grounded in the measured results.

The initial bounded experiment uses the first 1,000,000 records from Loghub's BGL archive. It is not a claim of full-dataset benchmark performance. Actual model inference runs locally; no third-party inference API is required.

## Data contract

BGL alert tags provide retrospective labels. `-` is non-alert; any other first-column value is an alert. The parser strips that label and all metadata, including severity, before deterministic content masking. Input features are event identities, not label tags, timestamps, node IDs, or severity.

Split raw records in file order into 60% train, 20% validation, and 20% test before windowing. Create disjoint windows of 20 events inside each split. Discard and record incomplete tails. A positive window contains at least one labeled alert event. Do not apply point adjustment or expand predictions using ground-truth ranges.

The event vocabulary is fitted on training content only, with frequency >=2; rare/unseen templates map to token zero. Vocabulary discovery includes all training records; normal-only filtering applies to neural weights and baseline frequencies. This use of normal labels makes the pipeline semi-supervised. Data labels do not enter inference features.

Training windows mix node streams because the source file interleaves messages. This is an initial file-order benchmark, not a node/session-aware operational deployment.

## Model and optimization contract

For a sequence e₁…e₂₀, predict e₂…e₂₀ from preceding events. The training loss and anomaly score are mean next-event negative log likelihood over these 19 targets:

`score = (1/19) Σ[t=2…20] −log p(e_t | e_1…e_(t−1))`

- **Event frequency:** Laplace-smoothed probabilities estimated from normal training targets. Ignores order.
- **LSTM:** 32-dimensional learned embeddings; 64 hidden units; linear vocabulary head; input/output dropout 0.15.
- **TCN:** Embeddings and a pointwise projection, followed by three residual causal Conv1D blocks with dilations 1/2/4, kernel size 3, GELU, and dropout. Left padding prevents future-event leakage.
- **Order ablation:** LSTM with positions 2–20 shuffled once within each normal training window per seed. Keep position 1 fixed, preserving both initial context and the multiset of scored targets. Validation/test inputs remain ordered.
- **Dropout ablation:** LSTM with dropout zero; hold other hyperparameters fixed.

Run neural experiments with seeds 17/42/73, AdamW learning rate 0.002, weight decay 0.0001, batch size 128, gradient clipping at norm 1, up to 8 epochs, and patience 3. Select each checkpoint by normal-validation cross-entropy. Select the default non-shuffle checkpoint by the same criterion, not test performance. Hardware/time and pre-clipping gradient norms are recorded.

## Evaluation contract

Calibrate each alert threshold at the 99th percentile of normal-validation scores using the higher order statistic. A window alerts only if score > threshold. This targets a validation operating point; it does not promise a 1% false-positive rate on test data.

Report average precision (AP), ROC AUC, precision, recall, F1, the confusion matrix, false positives per 1,000 total windows, normal-window false-positive rate, class prevalence, and unknown-template fractions. Show individual seeds, descriptive seed summaries, and per-run AP intervals from a moving-block bootstrap (50-window blocks, 200 resamples). AP is not accuracy or trapezoidal PR-AUC.

A test-time order permutation probe preserves the initial context and scored-target counts. Reuse the original threshold and report changed metrics. This is a controlled perturbation, not a new independent dataset.

The UI threshold slider is exploratory. The exported report retains the frozen 99th-percentile benchmark metrics. Do not tune reported settings against this test set; future model decisions need a new holdout or rolling-origin evaluation.

## Interface specification

Visual direction: quiet observability workspace; deep evergreen navigation, warm off-white canvas, mint accents, restrained amber warnings, readable typography, clear component boundaries, and dense but legible evidence tables.

Core journeys:

1. Open the overview → see real test metrics → inspect an alert or missed anomaly → identify surprising/unknown events.
2. Open Model lab → compare architectures and seed variability → understand the baseline and ablations.
3. Open Analyze logs → load a held-out example or paste/upload text → run the active checkpoint → inspect surprise scores and out-of-vocabulary warnings.
4. Open Method & evidence → inspect splits, provenance, protocol, limits, and course mapping.

Responsive layouts must support desktop and narrow mobile screens. Navigation, model selection, filters, pagination, dialogs, and inference must work by keyboard. Provide loading, empty, validation-error, inference-success, and missing-artifact states. Never render uploaded log text as HTML.

## D802 mapping

| Supplied course competency | Demonstrated evidence |
| --- | --- |
| Applies neural network architectures | Implemented LSTM and causal CNN, embeddings, linear layers, activations, residual connections |
| Analyzes deep learning models | Baselines, seed comparisons, AP and operating-point metrics, error inspection, order and dropout ablations |
| Applies techniques to optimize deep learning | AdamW, minibatches, weight decay, dropout comparison, clipped gradients, validation checkpointing, training curves |

The mapping follows the supplied course overview. No official assessment rubric was provided; this is an independent portfolio project.

## Acceptance criteria

- A fresh Python environment can install the package, fetch data, run training, and launch the application with documented commands.
- All displayed benchmark values come from exported experiment artifacts; no fabricated model results or fallback mock metrics.
- Saved checkpoint inference reproduces exported scores for the same event sequence.
- Tests establish causality, train-only vocabulary, split separation, label exclusion, threshold tie handling, a finite-difference gradient check, and end-to-end training/API behavior.
- Review desktop and mobile layouts and exercise important browser interactions.
- README/case study disclose dataset scope, distribution shift, semi-supervised fitting, and detection-versus-prediction limits.

## Deliberate limits and follow-up research

This version does not prove root cause, forecast future outages, learn arbitrary log formats, ingest production streams, or provide production access control. Scores are not calibrated failure probabilities. Add session/node grouping, a stronger adaptive parser, multiple untouched time splits, full-dataset evaluation, and a second dataset before making broader reliability claims.

## Sources

- [Loghub](https://github.com/logpai/loghub), dataset distribution, research/academic terms, and citations.
- [BGL documentation](https://github.com/logpai/loghub/tree/master/BGL), source system and alert-label interpretation.
- [BGL archive](https://zenodo.org/records/8196385/files/BGL.zip?download=1), downloaded original input; SHA-256 is recorded in the report.
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), recurrent layer contract.
- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html), state dictionaries and weights-only checkpoint loading.
