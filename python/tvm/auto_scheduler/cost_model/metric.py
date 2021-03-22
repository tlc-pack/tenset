"""Evaluation metric for the cost models"""
import numpy as np

def max_curve(trial_scores):
    """Make a max curve f(n) = max([s[i] fo i < n]) """
    ret = np.empty(len(trial_scores))
    keep = -1e9
    for i, score in enumerate(trial_scores):
        keep = max(keep, score)
        ret[i] = keep
    return ret


def metric_r_squared(preds, labels):
    """Compute R^2 value"""
    s_tot = np.sum(np.square(labels - np.mean(labels)))
    s_res = np.sum(np.square(labels - preds))
    if s_tot < 1e-6:
        return 1
    return 1 - s_res / s_tot


def metric_rmse(preds, labels):
    """Compute RMSE (Rooted Mean Square Error)"""
    return np.sqrt(np.mean(np.square(preds - labels)))


def vec_to_pair_com(vec):
    return (vec.reshape((-1, 1)) - vec) > 0


def metric_pairwise_comp_accuracy(preds, labels):
    """Compute the accuracy of pairwise comparision"""
    n = len(preds)
    if n <= 1:
        return 0.5
    preds = vec_to_pair_com(preds)
    labels = vec_to_pair_com(labels)
    correct_ct = np.triu(np.logical_not(np.logical_xor(preds, labels)), k=1).sum()
    return correct_ct / (n * (n-1) / 2)


def metric_top_k_recall(preds, labels, top_k):
    """Compute recall of top-k@k = |(top-k according to prediction) intersect (top-k according to ground truth)| / k."""
    real_top_k = set(np.argsort(-labels)[:top_k])
    predicted_top_k = set(np.argsort(-preds)[:top_k])
    recalled = real_top_k.intersection(predicted_top_k)
    return 1.0 * len(recalled) / top_k


def metric_peak_score(preds, labels, top_k):
    """Compute average peak score"""
    trials = np.argsort(preds)[::-1][:top_k]
    trial_scores = labels[trials]
    curve = max_curve(trial_scores) / np.max(labels)
    return np.mean(curve)


def random_mix(values, randomness):
    random_values = np.random.uniform(np.min(values), np.max(values), len(values))
    return randomness * random_values + (1 - randomness) * values

