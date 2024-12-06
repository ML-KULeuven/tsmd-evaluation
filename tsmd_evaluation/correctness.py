import numpy as np
import scipy.optimize

from . import prom

# Correctness
def correctness(gt, discovered_sets, threshold=0.5, corrected=False):
    if type(gt) is list:
        gt_sets   = gt
    elif type(gt) is dict:
        gt_sets   = list(gt.values())
    
    g, d = len(gt_sets), len(discovered_sets)
    if d == 0:
        return 0.0
    
    M = np.zeros((g, d))
    for i in range(g):
        for j in range(d):
            gt_set, discovered_set = gt_sets[i], discovered_sets[j]
            M[i, j] = np.sum([prom.overlap_rate(s, e, s_gt, e_gt) for (s, e) in discovered_set for (s_gt, e_gt) in gt_set if prom.overlap_rate(s, e, s_gt, e_gt) > threshold]) / len(gt_set)
    
    # I do this optimally because the authors do not specify how to.
    r, c = scipy.optimize.linear_sum_assignment(M, maximize=True)
    correctness = np.sum(M[r, c])

    m = min(g, d)
    assert len(r) == len(c) == min(g, d)
    return correctness / (m if not corrected else g)