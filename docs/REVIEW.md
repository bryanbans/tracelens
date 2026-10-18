# Holistic review and refinement — 2026-10-18

This review leaves the original detector weights, mapping, thresholds, frozen plan, audit scores, labels and reported metrics intact. LinkedIn publication is on hold. GitHub is the primary project record.

## Findings fixed

- **Unsupported replay inputs:** punctuation-only messages can produce raw activation 1.0 under the frozen CNN's masked pooling. The application now withholds the alert decision for semantic windows containing unencodable messages or no known retained words, and labels raw scores diagnostic. Known-word coverage, token truncation, blank lines, incomplete tails and original source line numbers are visible. Coverage is not a validated domain detector. Arbitrary pasted logs remain domain-unverified.
- **Frozen threshold bypass:** schema-v2 audit listings reject threshold overrides, unknown filters and unknown ranges. Previews use the first event, independent of ground truth.
- **Reproducibility:** provenance checks now raise explicit exceptions even under optimized Python; UTF-8 is explicit in audit reads. Startup checks bundled artifact hashes. Truncated archives cannot silently shorten requested audit ranges. The ordinary parser now accepts metadata-only BGL records consistently with the audit parser.
- **Request handling:** actual request bytes are bounded, including requests without Content-Length.
- **Interface correctness:** stale inspector requests are invalidated when filters change, stale errors cannot replace newer evidence, and initial detector-load failures surface as failures. Replay includes clear no-decision and domain/coverage states.
- **Publication provenance:** the earlier LinkedIn post is recorded as deleted by the user. Its former screenshot is excluded from GitHub and release packaging.

## Post-hoc uncertainty sensitivity

Using the existing frozen predictions only, repeat the same paired moving-block bootstrap with seed 2026, 500 repetitions and four block lengths. No model or policy selection follows from this analysis. Entries are 95% recall-difference intervals in percentage points (CNN minus original LSTM).

| Range | 100 windows | 250 windows | 1,000 windows | 2,500 windows |
| --- | --- | --- | --- | --- |
| audit_a | +22.6 to +55.3 | +17.5 to +62.7 | +13.9 to +70.8 | +14.4 to +63.5 |
| audit_b | +11.6 to +25.4 | +11.4 to +27.1 | +9.9 to +31.3 | +10.7 to +34.5 |
| confirmation | +63.8 to +85.0 | +60.3 to +86.8 | +54.6 to +88.6 | +45.7 to +92.4 |

Recall-difference intervals stay positive for all twelve range/block combinations. Normal-FPR-difference intervals also remain positive in all combinations. This checks one uncertainty assumption; it does not establish independence, transportability, future performance or external replication. Block lengths were examined after seeing the audits. These are conditional, descriptive intervals, not a new confirmatory experiment.

Reproduce with `python scripts/audit_sensitivity.py`. Exact results and source-report/freeze hashes are in [audit-sensitivity.json](audit-sensitivity.json).

## Remaining scientific work

1. Add a same-supervision lexical baseline (TF-IDF/logistic regression), and a clean ablation separating representation, supervision, ensemble and calibration effects. The existing comparison does not isolate the benefit of depth.
2. Use fresh evaluation data after pre-registering the next selection and calibration plan. These audit ranges are already inspected. Do not tune to their errors and describe a subsequent score as untouched validation.
3. Investigate calibration drift on development data. The frozen CNN still misses the 1% normal-FPR target in every audit range; this release does not claim to repair it.
4. Validate domain screening and longer-message handling separately. The new coverage checks detect obvious unsupported input; they cannot establish that a log belongs to BGL or justify production use.
5. Preserve the disclosed first-freeze omission of the independent word-vocabulary hash. Later integrity checks cannot retroactively prove that missing pre-audit commitment.

## Verification

40 automated tests pass locally, including real-checkpoint parity against packed audit scores, full packed-metric reproduction, malformed calibration data, unsupported replay, source-line retention, token truncation, chunked request limits, immutable audit thresholds, artifact corruption and short archive rejection. Both frontend scripts pass syntax checks. Desktop replay was exercised through the browser, including the no-decision case; desktop and phone layouts were visually reviewed. The original freeze and all 19 release artifact checksums still verify.
