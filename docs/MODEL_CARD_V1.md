# TraceLens model card

## Identity and purpose

Version 0.1.0. The included artifacts contain 12 neural checkpoints across four experiments and three seeds, plus one deterministic frequency baseline. Default checkpoint: `lstm-s17`, selected by minimum normal-validation predictive loss among non-shuffle candidates.

Intended use: academic/portfolio research into retrospective log-window anomaly detection, model comparison, and evaluation. The local interface is an investigation aid.

## Training data

First 1,000,000 records from the public Loghub BGL archive. Splits are contiguous in file order: 600,000 train, 200,000 validation, 200,000 test records. The original archive SHA-256 is `d67fd82a711aea0157a9b83175892c6ee60e384a2ddf5bc51f39118453816da8`.

Features are deterministic normalized content templates. Source label and metadata, including severity, are stripped. A train-only vocabulary maps rare/new templates to token zero. Normal-only weight fitting and calibration use dataset labels; therefore the protocol is semi-supervised.

## Outputs

Next-event negative log likelihood, averaged over 19 target events in each 20-event window. Thresholds are the higher-order-statistic 99th percentile of normal-validation scores. Strictly greater scores trigger an alert. Scores are not probabilities of failure. Per-event surprise is a loss decomposition, not a causal attribution or root-cause diagnosis.

## Evaluation summary

LSTM seed-mean AP: 0.8556, sample SD 0.0062. Default checkpoint AP: 0.8500. Default frozen operating point: precision 1.0000, recall 0.0613, F1 0.1156, TP=13, FP=0, FN=199, TN=9,788. These are the specific prefix/split results. No broad reliability guarantee follows from zero observed false positives.

See `report.json` for every model, threshold, AP interval, and experiment setting. UI threshold changes are exploratory and leave the saved benchmark unchanged.

## Known limitations

- Alert categories are not guaranteed future outages. Labels may be imperfect.
- Older supercomputer data may not represent contemporary applications.
- File-order windows can mix node messages; no node/session-aware grouping is performed.
- Unknown-event frequency is 15.55% on test, compared with 0.0125% on validation.
- Strong AP can coexist with poor thresholded recall; operating points vary across seeds.
- Mean loss can dilute a single surprising event, and the first event is context only.
- Prefix selection, temporal dependence, and a single inspected holdout limit generalization claims.
- Pasted logs from a new format can become entirely unknown; an alert then indicates unfamiliarity, not demonstrated malfunction.

## Operational boundaries

Designed for loopback/local use. No production identity/access system, multi-tenant isolation, streaming ingestion, automated remediation, or durable log storage. The app does not deliberately persist uploaded/pasted logs; source files remain on the user's device and input is processed in the local server. Browser form history/memory and the running process remain outside that storage claim.

Use checkpoints bundled with a trusted copy of this project. Loading uses PyTorch state dictionaries with `weights_only=True`. Dataset usage terms are separate from the code license.

## Maintenance

Further tuning requires a fresh holdout or rolling-origin protocol. Retrain event vocabulary and models for another log domain, document the new data contract, and revalidate calibration before relying on alerts.
