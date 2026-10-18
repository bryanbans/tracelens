# Deep learning mechanics in TraceLens

These notes connect the implementation to the mathematical ideas behind it. Use them to prepare an explanation of the project, then inspect the corresponding code and training curves.

## Supervised content CNN (current release)

The input is `[B, 20, 16]`: 20 messages, each truncated to its first 16 word tokens. A 24-dimensional embedding feeds two 24-channel convolutions of widths 3 and 5, followed by GELU and masked max pooling over token positions. Concatenating the pooled vectors produces 48 features per event. A 32-unit GELU head, dropout, and a scalar logit complete the event model.

Training uses the maximum event logit as the window logit and weighted binary cross-entropy with positive weight `normal_windows / anomalous_windows`, fitted on training labels only. This is multiple-instance learning: an anomalous window label does not identify the responsible event. The hard maximum routes the window loss through the winning event; it can learn a useful phrase detector but does not justify causal event attribution. Class weighting also means the sigmoid output is not a calibrated failure probability.

At inference, average the three seeded models' sigmoid activations separately for each event, then take the maximum across events. Averaging window maxima instead would be a different scoring rule. The event max makes this content CNN invariant to permutation of messages inside a window; it does not model cross-message temporal order. The historical LSTM/TCN experiments below supply the sequence-model evidence.

Padding is excluded from token pooling. An event with no encodable tokens nevertheless produces an extreme raw activation under the frozen network. The replay application therefore withholds a decision for any semantic window containing such an event, or with no known retained words. This is an application-level validity check, not a retrained model or a validated out-of-distribution detector. Audited predictions are preserved.

The next scientific comparison should include a supervised TF-IDF/logistic baseline under the same training labels and splits. Comparisons with normal-only LSTM models confound representation, objective, and supervision. The current audit cannot establish that deep learning beats a similarly supervised lexical baseline. Any subsequent model selection needs genuinely new evaluation data.

## Historical sequence representation and computation graph

A window contains 20 integer event identities. An embedding matrix E ∈ R^(114×32) maps each identity to a learned 32-dimensional vector. The model receives events 1–19 and predicts events 2–20. With batch size B, its input is `[B, 19]` and output logits are `[B, 19, 114]`.

No event identity has an inherent numerical distance from another. Embeddings learn useful relationships through the predictive objective. Their parameters receive gradients when the corresponding identities occur as inputs. Vocabulary discovery alone does not train an embedding.

The graph is: event lookup → sequence encoder → dropout → linear vocabulary projection → cross-entropy. Autograd follows this graph backward; the finite-difference test independently checks one output-head derivative.

## Objective and anomaly score

At target position t, logits z_t are converted to p_t = softmax(z_t). For observed identity e_t, loss is −log p_t[e_t]. The training objective averages over every target in every normal training window.

For softmax cross-entropy, the derivative with respect to logit z_(t,k) is `p_(t,k) − 1[k=e_t]`, divided by the number of averaged target terms. This gradient passes through the linear head, encoder, and embedding lookup.

At evaluation, the same mean loss is used as a window anomaly score. This makes interpretation straightforward: high score means poor prediction of observed events under the learned normal model. It does not imply a calibrated failure probability. Mean scoring can dilute isolated surprises; a separate future experiment could compare alternative aggregation rules using fresh validation/holdout data.

## LSTM

For embedding x_t and prior hidden state h_(t−1), the LSTM computes input, forget, and output gates with sigmoid activations, and a candidate update with tanh. Its cell-state recurrence is:

`c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t`

`h_t = o_t ⊙ tanh(c_t)`

The additive cell-state path and learned gates help manage information and gradient flow over time. They do not guarantee perfect memory or remove every optimization difficulty. Hidden state is reset between independent windows in this implementation.

For the included 114-event vocabulary:

| Component | Parameter calculation | Parameters |
| --- | --- | ---: |
| Embedding | 114 × 32 | 3,648 |
| LSTM weights/biases | 4 × 64 × (32 + 64) + 8 × 64 | 25,088 |
| Output projection | 64 × 114 + 114 | 7,410 |
| **Total** | | **36,146** |

PyTorch has separate input-hidden and hidden-hidden bias vectors, which accounts for the two sets of gate biases.

## Causal temporal CNN

The CNN projects the embeddings to 64 channels and applies three residual Conv1D blocks. Each uses a kernel of size three and dilation 1, 2, or 4. Left padding ensures that the output at time t cannot inspect input events after t.

The receptive field is `1 + (3−1) × (1+2+4) = 15` input positions. Predicting event 20 therefore has up to 15 preceding input positions in this encoder. The LSTM can, in principle, carry information through all 19 input positions. These different context assumptions are part of the architecture comparison.

Residual additions provide a direct path around each nonlinear transformation. GELU supplies nonlinearity. The encoder retains the time dimension so its outputs stay aligned with next-event targets; temporal pooling is not required for this objective.

Parameter count: embedding 3,648 + projection 2,112 + convolution blocks 37,056 + output head 7,410 = **50,226**.

## Optimization and regularization

- **AdamW** combines adaptive first/second-moment estimates with decoupled weight decay. The saved config records learning rate 0.002 and weight decay 0.0001.
- **Minibatches** provide stochastic gradient estimates. Seeded batch shuffling changes training order while retaining reproducibility on the recorded runtime.
- **Dropout** perturbs activations during training and is disabled by evaluation mode. The zero-dropout experiment changes this setting alone. Its small AP difference is descriptive evidence, not proof that regularization is unnecessary.
- **Gradient clipping** limits the aggregate gradient norm to one. Logged norms are measured before clipping; this distinction matters when inspecting optimization behavior.
- **Checkpoint selection** minimizes normal-validation cross-entropy. The training curve includes all completed epochs, even when an earlier checkpoint is restored.

Good predictive loss on normal validation data does not necessarily select the best anomaly detector. The default checkpoint's ranking/threshold discrepancy is a concrete example of the mismatch between a training objective and an operational objective.

## What the ablations establish

The frequency baseline tests performance without learned temporal context. Shuffling positions 2–20 during LSTM training preserves initial context and the multiset of scored targets, changing transition structure. Keeping ordered validation/test inputs makes the comparison interpretable under this specific training budget.

The test-time permutation probe is a separate sensitivity check. A score change shows that the encoder responds to altered context; it does not by itself prove useful contextual reasoning on real incidents. Seeds and the baseline are necessary to avoid attributing incidental variation to an architectural benefit.

## Evaluation questions to answer aloud

1. Why can AP stay high while recall at a particular threshold falls?
2. Why must the source alert label and severity metadata be removed before feature construction?
3. Why fit vocabulary before seeing validation/test, and what happens when test templates are unknown?
4. What changes if the window contains messages from multiple nodes?
5. Why does a 99th-percentile calibration on validation not guarantee 1% false positives on test?
6. Why do the small ordered/shuffled and dropout differences need more data splits before stronger claims?
