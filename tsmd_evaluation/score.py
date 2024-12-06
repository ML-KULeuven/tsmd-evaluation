import numpy as np
import scipy.optimize

def score(gt_sets, discovered_sets, penalize_off_target=False):
    """
    Calculate the score between ground truth sets and discovered sets.

    Parameters:
    gt_sets (list or dict): Ground truth motif sets.
    discovered_sets (list): Discovered motif sets.
    penalize_off_target (bool): Whether to penalize off-target motif sets.

    Returns:
    float: The score value.
    """

    if type(gt_sets) is dict:
        gt_sets   = list(gt_sets.values())
    g, d = len(gt_sets), len(discovered_sets)
    
    # Calculate optimal score between each pair of a GT and discovered motif set.
    M = np.zeros((g, d))
    for i in range(g):
        for j in range(d):
            gt_set = gt_sets[i]
            discovered_set = discovered_sets[j]
            M[i, j] = optimal_score(gt_set, discovered_set)

    # Find the matching between sets with the lowest possible score.
    r, c = scipy.optimize.linear_sum_assignment(M, maximize=False)
    score = np.sum(M[r, c])

    # Penalize unmatched motif sets
    if d < g:
        unmatched = np.setdiff1d(range(g), r)
        for i in unmatched:
            score += sum([e-s for (s, e) in gt_sets[i]])
    elif d > g:
        if penalize_off_target:
            unmatched = np.setdiff1d(range(d), c) 
            for j in unmatched:
                score += sum([e-s for (s, e) in discovered_sets[j]])
    
    return score
    
def optimal_score(gt_set, discovered_set):  
    """
    Calculate the optimal score between a ground truth motif set and a discovered motif set.

    Parameters:
    gt_set (list): Ground truth motif set.
    discovered_set (list): Discovered motif set.

    Returns:
    float: The optimal score between them.
    """  

    k, k_gt = len(discovered_set), len(gt_set)
    
    M = np.full((k_gt + k, k + k_gt), np.inf)
    for i in range(k_gt):
        (s_gt, e_gt) = gt_set[i]
        for j in range(k):
            (s, e) = discovered_set[j]
            # If overlapping, M_i,j is the difference in start indices. Else np.inf
            if s_gt < e and s < e_gt:
                M[i, j] = abs(s_gt - s) 
    
    # Each GT motif is either matched, or adds its length to the score
    for i in range(k_gt):
        (s_gt, e_gt) = gt_set[i]
        M[i, k:] = e_gt - s_gt
        
    # Each discovered motif is either matched, or adds its length to the score
    for j in range(k):
        (s, e) = discovered_set[j]
        M[k_gt:, j] = e - s
        
    M[k_gt:, k:] = 0
    r, c = scipy.optimize.linear_sum_assignment(M, maximize=False)
    score = np.sum(M[r, c])
    return score