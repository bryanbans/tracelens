"""Calibration policies use calibration labels only; scoring does not consume labels."""
import numpy as np


def aggregate(losses, mode="mean"):
    losses = np.asarray(losses, dtype=np.float64)
    if mode == "mean":
        return losses.mean(axis=1)
    if mode == "max":
        return losses.max(axis=1)
    if mode == "top3":
        return np.sort(losses, axis=1)[:, -min(3, losses.shape[1]):].mean(axis=1)
    raise ValueError(f"Unknown score aggregation: {mode}")


def f2_from_metrics(m):
    p, r = m["precision"], m["recall"]
    return 5 * p * r / (4 * p + r) if 4 * p + r else 0.


def choose_threshold(scores, labels, policy, budget=.01):
    """Strict score > threshold, including ties. No randomized interpolation."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    normal = scores[labels == 0]
    if not len(normal):
        raise ValueError("Calibration needs normal examples.")
    if policy == "normal_quantile_99":
        return float(np.quantile(normal, .99, method="higher"))
    if policy != "f2_budget":
        raise ValueError("Unknown threshold policy.")
    if not np.any(labels == 1):
        raise ValueError("F2 calibration requires positive examples.")
    # Sort once and evaluate decisions only after complete equal-score groups.
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    end = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y)[end].astype(float)
    fp = (end + 1) - tp
    recall = tp / y.sum()
    precision = tp / (tp + fp)
    f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                   out=np.zeros_like(recall), where=(4*precision+recall)>0)
    feasible = fp / len(normal) <= budget
    if not feasible.any():
        return float(scores.max())  # No alerts; an explicit conservative fallback.
    idx = max(np.flatnonzero(feasible), key=lambda i:(f2[i], -fp[i], s[end[i]]))
    return float(np.nextafter(s[end[idx]], -np.inf))


def select_candidate(candidates, budget=.01):
    feasible = [c for c in candidates if c["selection_metrics"]["normal_false_positive_rate"] <= budget]
    if feasible:
        return sorted(feasible, key=lambda c:(-f2_from_metrics(c["selection_metrics"]),
                     -c["selection_metrics"]["average_precision"], c["id"]))[0], True
    return sorted(candidates, key=lambda c:(c["selection_metrics"]["normal_false_positive_rate"],
                  -f2_from_metrics(c["selection_metrics"]), c["id"]))[0], False


def paired_block_intervals(labels, old_predictions, new_predictions, block_size=250, repetitions=500, seed=2026):
    """Paired moving-block bootstrap; retain temporal dependence within sampled blocks."""
    labels=np.asarray(labels); old=np.asarray(old_predictions); new=np.asarray(new_predictions)
    rng=np.random.default_rng(seed); n=len(labels); b=min(n,block_size)
    values=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n-b+1,size=int(np.ceil(n/b)))
        idx=np.concatenate([np.arange(s,s+b) for s in starts])[:n]
        y=labels[idx]; pos=y==1; neg=~pos
        if pos.any() and neg.any():
            values.append([float(new[idx][pos].mean()-old[idx][pos].mean()),
                           float(new[idx][neg].mean()-old[idx][neg].mean())])
    if not values:
        return None
    lo,hi=np.quantile(values,[.025,.975],axis=0)
    return {"recall_difference_95":[float(lo[0]),float(hi[0])],
            "normal_fpr_difference_95":[float(lo[1]),float(hi[1])],
            "block_size":b,"repetitions":repetitions,"valid_resamples":len(values)}
