# TraceLens v2 portfolio material

The current post text is in [LINKEDIN_POST.txt](LINKEDIN_POST.txt). The technical account is [CALIBRATION_AUDIT.md](CALIBRATION_AUDIT.md). Historical v1 results are retained separately and must not be presented as the new detector's audit performance.

## Two-minute demonstration

1. Observe: show combined recall, precision, and normal FPR together, including the calibration drift notice.
2. Switch to Confirmation: explain the different same-range recall and false-positive values.
3. Select Original and Recalibrated to distinguish recalibration alone from the complete representation/model change.
4. Inspect a flagged window, a false positive, and a missed anomaly. Explain event activations and the independent dataset labels.
5. Validate: show the three time ranges and paired block-bootstrap intervals. Explicitly acknowledge Audit B's weak precision.
6. Replay: load a real audit example, import a log file, and run the CNN ensemble locally.
7. Protocol: explain temporal boundaries, pre-audit freezing, source/checkpoint hashes, and the prior coarse label-count inspection disclosure.

## Questions to prepare for

- Why do word embeddings help when event templates change?
- How do token-wise and event-wise max pooling relate to the window label?
- What is the difference between supervised content modeling and the original normal-only sequence fitting?
- Why does a 1% calibration budget fail to guarantee 1% FPR under temporal shift?
- What can a paired moving-block interval establish, and what dependence/bias can it miss?
- Why do further calibration improvements need new data instead of these now-inspected audit ranges?

## Publication record

The user deleted the earlier LinkedIn post. Publication is on hold; LINKEDIN_PUBLICATION.json preserves the historical record. GitHub repository: https://github.com/bryanbans/tracelens.
