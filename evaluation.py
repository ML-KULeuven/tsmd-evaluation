import numpy as np
import time
import pandas as pd
import scipy.optimize
import sklearn.metrics as skmetrics

## MATCH MATRIX
def overlap_union_ratio(s, e, s_gt, e_gt):
    return max(0, (min(e, e_gt) - max(s, s_gt)) / (max(e, e_gt) - min(s, s_gt)))


def match_matrix(gt, discovered_sets, threshold=0.5):
    assert (threshold >= 0.5)
    assert gt
        
    g, d = len(gt), len(discovered_sets)
    column_names = np.arange(1, d+1)
    if type(gt) is list:
        row_names = np.arange(1, g+1)
        gt_sets   = gt
    elif type(gt) is dict:
        row_names = np.array(list(gt.keys())) 
        gt_sets   = list(gt.values())
    
    if not discovered_sets:
        # only return missed detection column
        match_matrix = np.zeros((g+1, 1), dtype=int)
        match_matrix[:-1, 0] = [len(gt_set) for gt_set in gt_sets]
        match_matrix[-1, 0]  = 0
        return match_matrix, row_names, np.array([])
    
    # inner match matrix
    mm = np.zeros((g, d), dtype=int)
    for i, gt_set in enumerate(gt_sets):
        for (s_gt, e_gt) in gt_set:
            # find the best match in any motif set, greater than threshold
            best     = None
            best_our = 0.0 
            for j, discovered_set in enumerate(discovered_sets):
                for (s, e) in discovered_set:
                    our = overlap_union_ratio(s, e, s_gt, e_gt)
                    if our > threshold and our > best_our:
                        best = j
                        best_our = our
            
            if best is not None:
                mm[i, best] += 1
    
    # r and c are of length min(n, m)
    r, c = scipy.optimize.linear_sum_assignment(mm, maximize=True)
    
    r_perm = np.hstack((r, np.setdiff1d(range(g), r))) 
    c_perm = np.hstack((c, np.setdiff1d(range(d), c)))
    
    # compute number of missed detections
    md = np.array([len(gt_set) for gt_set in gt_sets])
    md = md - np.sum(mm, axis=1)
    
    # compute number of false discoveries 
    fd = np.array([len(discovered_set) for discovered_set in discovered_sets])
    fd = fd - np.sum(mm, axis=0)
    
    # permute M
    mm = mm[:, c_perm]
    mm = mm[r_perm, :]
    
    # permute false discoveries and missed detections
    fd = fd[c_perm]
    md = md[r_perm]
    
    md = np.expand_dims(md, axis=0).T
    M = np.block([
        [mm, md],
        [fd, 0 ] 
    ])
    if possible_tie(M):
        M, r_perm, c_perm = break_ties(M, r_perm, c_perm)
    
    return M, row_names[r_perm], column_names[c_perm]


def possible_tie(M):
    (g, d) = M.shape[0] - 1, M.shape[1] - 1 
    if g >= d:
        return False
        
    for i in range(g):
        tp = M[i, i]
        if np.sum(M[i, :d] == tp) > 1: return True
    return False

    
def break_ties(M, r, c):
    (g, d) = M.shape[0] - 1, M.shape[1] - 1 
    mm = np.zeros((g, d))
    
    for i in range(g):
        tp = M[i, i]
        fp = np.sum(M[:, :d], axis=0, dtype=float) - M[i, :d]
        fp[~(M[i, :d] == tp)] = np.inf
        mm[i, :] = fp

    r_, c_ = scipy.optimize.linear_sum_assignment(mm, maximize=False)

    r_ = np.concatenate((r_, [g]))
    c_ = np.concatenate((c_, np.setdiff1d(range(d), c_), [d]))

    M = M[r_, :]
    M = M[:, c_]
    return M, r[r_[:g]], c[c_[:d]]


## MICRO-AVERAGED METRICS
def micro_averaged_recall(match_matrix):
    g, d = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    if d == 0:
        return 0.0
    m = min(g, d)
    diag = np.diag(match_matrix[:m, :m])
    return np.sum(diag) / np.sum(match_matrix[:g, :])


def micro_averaged_precision(match_matrix, penalize_additional=False):
    g, d = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    # Should be nan.
    if d == 0:
        return 0.0
        
    m = min(g, d)
    diag = np.diag(match_matrix[:m, :m])

    if penalize_additional:
        return np.sum(diag) / np.sum(match_matrix[:, :d])
    else:
        return np.sum(diag) / np.sum(match_matrix[:, :m])

def micro_averaged_f1(match_matrix, penalize_additional=False):
    p = micro_averaged_precision(match_matrix, penalize_additional=penalize_additional)
    r  = micro_averaged_recall(match_matrix)
    if (p == 0.0 and r == 0.0) or np.isnan(p):
        return 0
    return 2 * r * p / (p + r)

## MACRO-AVERAGED METRICS
def macro_averaged_precision(match_matrix, penalize_additional=False):
    g, d = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    m = min(g, d)    

    tps = np.diag(match_matrix[:m, :m])
    ps = np.divide(tps, np.sum(match_matrix[:, :m], axis=0))
    p = np.sum(ps) / d if penalize_additional else np.sum(ps) / m
    return p

def macro_averaged_recall(match_matrix):
    g, d = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    m = min(g, d)

    tps = np.diag(match_matrix[:m, :m])
    rs = np.divide(tps, np.sum(match_matrix[:m, :], axis=1))
    r  = np.sum(rs) / g
    return r

def macro_averaged_f1(match_matrix):
    g, d = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    assert d >= g
    m = min(g, d)
    
    tps = np.diag(match_matrix[:m, :m])
    rs = np.divide(tps, np.sum(match_matrix[:m, :], axis=1))
    ps = np.divide(tps, np.sum(match_matrix[:, :m], axis=0))
    f1s = 2 * np.divide(np.multiply(rs, ps), (rs + ps))
    f = np.sum(f1s) / m
    return f


## VISUALIZATION
def pretty_print_match_matrix(match_matrix, row_names, col_names):
    from tabulate import tabulate
    table, row_names, col_names = match_matrix.tolist(), row_names.tolist(), col_names.tolist()
    table.insert(0, [""] + col_names + ["MD"])
    
    for i, row_name in enumerate(row_names):
        table[i+1].insert(0, row_name)
        
    table[-1].insert(0, "FD")
    table[-1][-1] = "-"
    return tabulate(table, [], tablefmt="grid")


## OTHER METRICS
# Score
def score_metric(gt_sets, discovered_sets, penalize_additional=False):
    if type(gt_sets) is dict:
        gt_sets   = list(gt_sets.values())
    g, d = len(gt_sets), len(discovered_sets)
    m = max(g, d)
    
    matrix = np.zeros((g, d))
    for i in range(g):
        for j in range(d):
            gt_set = gt_sets[i]
            discovered_set = discovered_sets[j]
            matrix[i, j] = optimal_score(gt_set, discovered_set)

    r, c = scipy.optimize.linear_sum_assignment(matrix, maximize=False)
    score = np.sum(matrix[r, c])

    if d < g:
        i_unmatched = np.setdiff1d(range(g), r)
        for i in i_unmatched:
            score += sum([e-s for (s, e) in gt_sets[i]])
    elif d > g:
        if penalize_additional:
            j_unmatched = np.setdiff1d(range(d), c) 
            for j in j_unmatched:
                score += sum([e-s for (s, e) in discovered_sets[j]])
    
    return score
    
def optimal_score(gt_set, discovered_set):    
    k, k_gt = len(discovered_set), len(gt_set)
    
    matrix = np.full((k_gt + k, k + k_gt), np.inf)
    for i in range(k_gt):
        (s_gt, e_gt) = gt_set[i]
        for j in range(k):
            (s, e) = discovered_set[j]
            # if overlapping
            if s_gt < e and s < e_gt:
                matrix[i, j] = abs(s_gt - s) 
        
    for i in range(k_gt):
        (s_gt, e_gt) = gt_set[i]
        matrix[i, k:] = e_gt - s_gt
        
    for j in range(k):
        (s, e) = discovered_set[j]
        matrix[k_gt:, j] = e - s
        
    matrix[k_gt:, k:] = 0

    r, c = scipy.optimize.linear_sum_assignment(matrix, maximize=False)
    score = np.sum(matrix[r, c])
    return score

# Correctness
def correctness(gt_sets, discovered_sets, threshold=0.5):
    if type(gt_sets) is dict:
        gt_sets   = list(gt_sets.values())
    
    g, d = len(gt_sets), len(discovered_sets)
    
    matrix = np.zeros((g, d))
    for i in range(g):
        for j in range(d):
            gt_set, discovered_set = gt_sets[i], discovered_sets[j]
            matrix[i, j] = np.sum([overlap_union_ratio(s, e, s_gt, e_gt) for (s, e) in discovered_set for (s_gt, e_gt) in gt_set if overlap_union_ratio(s, e, s_gt, e_gt) > threshold]) / len(gt_set)
    
    # I do this optimally because the authors do not specify how to.
    r, c = scipy.optimize.linear_sum_assignment(matrix, maximize=True)
    correctness = np.sum(matrix[r, c])

    m = min(g, d)
    assert len(r) == len(c) == min(g, d)
    if m == 0:
        return 0.0
    return correctness / m