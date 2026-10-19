# TraceLens

[![Application and evaluation checks](https://github.com/bryanbans/tracelens/actions/workflows/checks.yml/badge.svg)](https://github.com/bryanbans/tracelens/actions/workflows/checks.yml)

**Understand what a log detector catches—and what it misses.**

Finding unusual messages in a system log is only part of the problem. An engineer also needs to know which alerts deserve attention, which abnormal messages went unnoticed, and how many false alarms a detector creates.

TraceLens brings model scores, log messages, and evaluation results into one place. It helps you investigate flagged messages, examine mistakes, and compare whether a different model or alert threshold improves detection on later logs.

![TraceLens workspace showing model results and the messages behind an alert](docs/images/app-language-overview.jpg)

## What questions does TraceLens help answer?

| Question | What you can do today |
| --- | --- |
| **Which parts of the log does the model flag?** | Follow scores across the file and open the messages behind a flagged result. |
| **What did it miss, and which alerts were false alarms?** | Use the dataset labels to inspect missed anomalies and normal windows that triggered alerts. |
| **Which messages received the strongest response?** | Compare message scores with the text and the dataset's labels. |
| **Would changing the alert threshold be enough?** | Compare the original model, that same model with a new threshold, and a model that learns message phrases. |
| **Does the detector keep working on later logs?** | Compare detection and false-alarm rates across three later sections of the file. |
| **How does it respond to a log sample?** | Paste or import messages and run the saved models locally, with feedback about recognized words and input limits. |

The app analyzes **20 messages at a time**, called a window. It flags a window when its score exceeds the selected model's saved threshold. For evaluation, a window is labeled anomalous if any message in it has a dataset alert label. Message scores show model responses; they do not establish the cause of a system problem.

## The challenge: an error message is not the whole story

An included example contains these messages, with other lines between them:

```text
instruction cache parity error corrected
...
machine check interrupt
```

In this window, the correction message is labeled normal and the interrupt is labeled abnormal. TraceLens shows those labels alongside the model responses.

A keyword search can find the word “error.” It cannot tell you how many abnormal windows a detector missed, how many ordinary windows it flagged, or whether its alert threshold still works on the next section of the log.

TraceLens lets you move from an overall detection score to the messages behind it. The model's decision and the dataset label are displayed separately, so you can examine **what was flagged, what was labeled abnormal, and where they disagree**.

## Current project status

**Working local application with trained deep learning models and a reproducible evaluation.**

- The app includes real evaluation examples from the Blue Gene/L (BGL) supercomputer logs. You can explore the results without training a model first.
- Three model configurations can be compared on the same later log windows. Sample scoring uses the saved models on your own machine; it does not call a cloud AI service.
- The message-content model catches more labeled anomalies, but also creates more false alarms. Its 1% false-positive target was missed in every later evaluation range.
- Pasted samples have no verified labels, so the app reports scores and input checks rather than claiming accuracy. Unusable input receives no alert decision.
- A comparison with a simple text classifier trained using the same labels, and evaluation on new data, remain unfinished. This is a research prototype; it has not been validated as an operational outage predictor.

## The deep learning study

The study follows from the practical question: **does changing how a model learns from logs help more than changing when it raises an alert?**

The original **long short-term memory network (LSTM)** learns to predict the next message template from normal log windows. A message template replaces variable values, such as numbers, with placeholders. Unexpected templates receive higher scores.

The newer **convolutional neural network (CNN)** learns phrase patterns from windows labeled normal or anomalous. Three separately trained CNNs average their scores for each message; the strongest averaged message response becomes the window score. Unlike the LSTM, this model does not learn the order of messages across a window.

The results show why detection and false alarms need to be reported together. The CNN catches substantially more labeled anomalies, but many of its alerts still need to be ruled out.

| Model | Anomalous windows caught | Normal windows incorrectly flagged | Flagged windows that were anomalous |
| --- | ---: | ---: | ---: |
| Original LSTM | 75 of 7,439 (1.01%) | 17 | 81.52% |
| Same LSTM, new alert threshold | 955 of 7,439 (12.84%) | 9,577 | 9.07% |
| Word CNN ensemble | **3,448 of 7,439 (46.35%)** | **3,952** | **46.59%** |

These results cover **112,398 non-overlapping windows**, including 7,439 labeled anomalous and 104,959 normal windows. The CNN's 7,400 flagged windows include 3,448 labeled anomalous and 3,952 labeled normal. It misses another 3,991 anomalous windows.

When the CNN threshold was set on an earlier calibration range, fewer than 1% of normal windows were flagged. On later logs, that rate rose to **3.77% overall** and exceeded the 1% target in all three ranges. The model caught more anomalies in each range, with more false alarms too. The app keeps that tradeoff visible rather than presenting higher recall alone as success.

The CNN also learns from anomaly labels during training, while the original sequence models learn from normal windows only. The comparison therefore changes more than the network architecture. It does not establish that deep learning beats a simple text classifier trained with the same labels.

[Full evaluation and uncertainty estimates](docs/CALIBRATION_AUDIT.md) · [Model card](docs/MODEL_CARD.md) · [Deep learning methods](docs/DEEP_LEARNING_NOTES.md) · [Review and next experiments](docs/REVIEW.md)

## Try the app

Requires Python 3.11 or newer. The models run on CPU; no GPU or paid AI service is required.

```shell
git clone https://github.com/bryanbans/tracelens.git
cd tracelens
```

On Windows, run **Launch TraceLens.cmd** to install dependencies and start the app. For manual setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
tracelens serve
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead. Open **http://127.0.0.1:8765** and keep the server running.

Start in **Explore logs** and inspect a flagged window. Switch to **Missed anomalies** or **False positives**, then open **Compare models** to see the tradeoff across later log ranges. In **Try a sample**, load the included example and select **Score messages**. **Method** explains how the models and thresholds were chosen. Sample input is processed by the local server.

## Data and evaluation design

TraceLens uses historical BGL logs. Consecutive messages can come from different machines, and the labels describe abnormal messages that have already occurred. These results do not establish advance warning of an outage or performance on another system.

Training, model selection, threshold calibration, and later evaluation use separate file ranges. Alert labels and metadata such as severity and timestamps are excluded from model inputs. The evaluated model weights and thresholds were fixed before later predictions were examined; those results are preserved rather than adjusted after seeing the mistakes.

There are limits to that record. Archive-wide counts of normal and abnormal records had been inspected earlier, so this was not an independent test on a fully unseen dataset. The original evaluation record also omitted a separate file fingerprint for the CNN word-to-number mapping. The retained mapping matches the training copy, but later checks cannot retroactively supply that missing record. Both limitations are documented in the [evaluation report](docs/CALIBRATION_AUDIT.md).

The CNN reads only the first 16 word tokens per message. Its scores are not outage probabilities, and recognized words do not prove that the model is suitable for another log format. Further model or threshold selection needs new evaluation data; the current evaluation ranges have already been examined.

Built with Python, PyTorch, FastAPI, NumPy, scikit-learn, and HTML/CSS/JavaScript. Saved models, training histories, evaluation scores, and file checksums are included. The raw BGL archive is downloaded separately for reproduction.

## Explore the project

- [Reproduce the experiments and run checks](docs/VALIDATION.md#reproduce-the-experiments)
- [Evaluation results and limitations](docs/CALIBRATION_AUDIT.md)
- [Model architecture, training, and intended use](docs/MODEL_CARD.md)
- [Deep learning concepts and implementation](docs/DEEP_LEARNING_NOTES.md)
- [Review findings and remaining work](docs/REVIEW.md)
- [Short demonstration guide](docs/PORTFOLIO.md)
- [Preserved original experiments](docs/README_V1.md)
- [Data attribution and terms](THIRD_PARTY_NOTICES.md)

Original code is MIT licensed. BGL-derived artifacts retain the separate research/academic terms and attribution requirements described in [Third-party notices](THIRD_PARTY_NOTICES.md).
