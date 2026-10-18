> Historical v1 results. See [the v2 calibration audit](CALIBRATION_AUDIT.md) for the current detector and later-time evaluation.

# TraceLens: when good anomaly ranking makes a weak alert

## Motivation

D802 introduced the mechanics of neural architectures and their optimization. TraceLens turns that foundation into an engineering investigation: can a sequence model detect abnormal system behavior from log events, and does the model actually use event order?

System logs are appealing because the observable sequence matters. A rare event can be harmless, a common event can be suspicious in context, and software changes can make normal events unfamiliar. The project connects deep learning to infrastructure reliability while keeping evaluation failures visible.

## Implementation

I implemented an order-independent event-frequency baseline, an LSTM, and a residual causal temporal CNN in PyTorch. All methods receive normalized event content; the BGL alert label and metadata, including severity, are excluded from input features. Neural models learn event embeddings and predict the next event. Mean target surprise provides the window anomaly score.

The LSTM has 36,146 trainable parameters. The CNN uses left-padded convolutions with dilation 1, 2, and 4. Tests verify that changing future tokens cannot change earlier predictions. A finite-difference test checks a gradient against backpropagation.

The UI supports real checkpoint inference, not a replay presented as inference. Saved test scores power the overview, while opening a window recomputes its event-level surprise from the selected checkpoint. The same scorer handles pasted logs.

## Experimental setup

Source: [Loghub's BGL archive](https://github.com/logpai/loghub/tree/master/BGL), derived from Blue Gene/L supercomputer logs. Original labels identify alert messages; they do not establish future outages.

This exploratory run uses the first 1,000,000 records in source-file order. Raw records are split before windowing:

| Partition | Source lines | Windows | Anomalous windows | Unknown-event fraction |
| --- | --- | ---: | ---: | ---: |
| Train | 1–600,000 | 30,000 | 11,919 | 0.0113% |
| Validation | 600,001–800,000 | 10,000 | 199 | 0.0125% |
| Test | 800,001–1,000,000 | 10,000 | 212 | 15.5495% |

Each non-overlapping window has 20 events. Its label is positive if any event is alert-tagged. The vocabulary has 114 entries including the unknown token. Only normal training windows fit model weights and frequency counts; labels also identify normal calibration windows. This is semi-supervised fitting.

The optimizer is AdamW with learning rate 0.002 and weight decay 0.0001. Each neural experiment runs seeds 17, 42, and 73 for up to eight epochs. Gradient clipping limits norm to one. Checkpoints are selected by normal-validation cross-entropy; patience is three epochs. The standard dropout rate is 0.15.

Two controlled comparisons extend the architecture comparison. The shuffled-training ablation permutes positions 2–20 once per window/seed, keeping initial context and scored-target counts fixed. The no-dropout ablation changes dropout alone, with other training settings held fixed.

## Results

| Experiment | Mean test AP | Sample SD across seeds | Test recall range at calibrated thresholds |
| --- | ---: | ---: | ---: |
| Event frequency | 0.8031 | — | 90.6% |
| LSTM | 0.8556 | 0.0062 | 1.4–66.0% |
| Temporal CNN | 0.8310 | 0.0497 | 1.9–69.8% |
| LSTM, shuffled training | 0.8506 | 0.0052 | 0.5–73.1% |
| LSTM, no dropout | 0.8538 | 0.0012 | 2.8–73.1% |

AP means average precision, a threshold-independent ranking metric. It is not classification accuracy. The prevalence of anomalous test windows is 2.12%. Seed standard deviations describe training variability on this same split; they do not measure dataset generalization.

The LSTM's mean AP is higher than the frequency baseline on this prefix. The ordered-versus-shuffled difference is small (about 0.0050 AP), and the no-dropout difference is smaller (about 0.0018 AP). These experiments do not justify a strong claim that order or dropout materially improves anomaly detection in general. Several shuffled-training runs also use their full epoch budget, so comparisons reflect this bounded training procedure, not necessarily fully converged models.

The CNN's larger seed variability is another reason to avoid showcasing only the strongest single run. Its best observed AP is similar to the LSTM's, but one seed performs below the frequency baseline.

## The most useful finding: calibration can fail independently of ranking

The default checkpoint is LSTM seed 17, chosen by normal-validation predictive loss, not by the best test score. It has AP **0.8500**. At the frozen 99th-percentile normal-validation threshold of approximately **6.0703**, it flags just **13 of 212** anomalous windows: **6.13% recall**, with **zero observed false positives** among 9,788 normal test windows.

The count baseline detects **192 of 212** anomalous windows but also flags **1,538** normal windows. Its precision is **11.10%**, and it produces **153.8 false alerts per 1,000 total test windows**.

A single AP number would hide these very different operational behaviors. The sequence explorer therefore makes missed anomalies and false positives first-class views. Its slider demonstrates how an operating point changes observed behavior; slider exploration does not rewrite the frozen benchmark metrics.

Unknown-event frequency rises sharply between validation and test. That is evidence of event-vocabulary shift. It is a plausible contributor to unstable scores, but these experiments do not isolate it as the sole cause of threshold behavior. Validation loss also measures next-event prediction on normal windows, which is not the same objective as maximizing anomaly recall.

## What I would improve next

1. Define node/session-aware sequences instead of windows that can mix interleaved node messages.
2. Reserve new time ranges and use rolling-origin evaluation before tuning thresholds or architecture further.
3. Add a stronger log parser and explicit unknown-template handling; evaluate the resulting trade-offs.
4. Compare calibration policies under a specified alert budget without selecting them on test labels.
5. Replicate on the complete BGL dataset and another labeled system-log dataset.

## Reproducibility and limitations

`artifacts/portfolio/report.json` contains the exact configuration, original archive SHA-256, recorded environment, individual seeds, losses, gradient norms, thresholds, confusion matrices, AP intervals, and a paired test-time order-permutation probe. AP intervals use a 50-window moving-block bootstrap with 200 resamples. They are descriptive and may underrepresent long-range dependence.

This is an exploratory portfolio release. It has one inspected holdout from one bounded data prefix, not a prospective operational validation. BGL labels are retrospective alerts; the project does not establish root cause, early outage prediction, or transfer to arbitrary service logs. The code, data protocol, and tests make these limits inspectable.

## Course evidence

- **Architectures:** embeddings, recurrent state and LSTM gates, causal convolutions, dilation, residual paths, GELU, and a vocabulary projection head.
- **Analysis:** a strong simple baseline, individual seeds, precision–recall evaluation, class balance, error inspection, and controlled ablations.
- **Optimization:** minibatch backpropagation, AdamW, weight decay, dropout comparison, clipped gradients, and validation-based checkpoint selection.

The most defensible outcome is a complete experiment and investigation tool: I can explain how the models learn, reproduce their measured behavior, and show why apparently strong metrics still require careful engineering judgment.
