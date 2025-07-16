import pandas as pd
import numpy as np

def add_noise(ts, noise_std=0.1):
    noise = np.random.normal(loc=0, scale=noise_std, size=ts.shape)
    return ts + noise
    
def generate_tsmd_benchmark_ts(df, g=2, noise_std=0.0, noise_on='none'):
    assert noise_on in ['none', 'motifs', 'non_motifs', 'all']
    
    freqs   = df['label'].value_counts()
    
    classes = freqs.index
    classes = classes[freqs > 1]

    if len(classes) < g:
        ValueError("TODO")
    
    # Pick the repeating classes randomly
    repeating_classes = np.random.choice(classes, size=g, replace=False)  
    # Sample one instance of every non-repeating class. Then randomly order them
    X_non_repeating = df[~df['label'].isin(repeating_classes)].copy()
    X_non_repeating = X_non_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample())
    X_non_repeating = X_non_repeating.sample(frac=1).reset_index(drop=True)
            
    # Sample at least two motifs for each repeating class
    X_repeating = df[df['label'].isin(repeating_classes)].reset_index(drop=True)
    motifs = X_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample(n=2))
    
    # Then complete the GT motifs by randomly sampling other motifs from repeating classes
    other_motifs = X_repeating[~X_repeating.apply(tuple, 1).isin(motifs.apply(tuple, 1))]
    other_motifs = other_motifs.sample(n=min(len(other_motifs), max(0, len(X_non_repeating)+1-len(motifs))), replace=False)

    all_motifs = pd.concat((motifs, other_motifs))    
    all_motifs = all_motifs.sample(frac=1).reset_index(drop=True)

    if noise_on in ['motifs', 'all']:
        all_motifs['ts'] = all_motifs['ts'].apply(lambda ts: add_noise(ts, noise_std))
        other_motifs['ts'] = other_motifs['ts'].apply(lambda ts: add_noise(ts, noise_std))
    if noise_on in ['non_motifs', 'all']:
        X_non_repeating['ts'] = X_non_repeating['ts'].apply(lambda ts: add_noise(ts, noise_std))
    
    gt = {c: [] for c in repeating_classes}
    ts = []

    # Concatenate instances, alternating between a non-repeating and a repeating class, until no instances left
    curr   = 0
    for i in range(min(len(X_non_repeating), len(all_motifs) - 1)):
        # Motif
        motif, label, l = all_motifs.iloc[i]
        ts.append(motif)
        gt[label].append((curr, curr+l))
        curr += l

        # Non-motif
        instance, _, l = X_non_repeating.iloc[i]
        ts.append(instance)
        curr += l
    
    motif, label, l = all_motifs.iloc[-1]
    ts.append(motif)
    gt[label].append((curr, curr+l))    
    return np.vstack(ts), gt

def convert_X_y_to_df(X, y):
    time_series = [x.T for x in X]
    lengths = [len(ts) for ts in time_series]
    df = pd.DataFrame({"ts": time_series, "label": y, "length": lengths})
    return df

def generate_tsmd_benchmark_dataset(df, N, g_min, g_max):    
    # Generate time series
    benchmark_ts = []
    gts = []
    for _ in range(N):
        
        # Sample a number of motif sets
        g = np.random.randint(g_min, g_max+1)
    
        ts, gt = generate_tsmd_benchmark_ts(df, g=g)
        benchmark_ts.append(ts)
        gts.append(gt)
        
    benchmark_dataset = pd.DataFrame({'ts': benchmark_ts, 'gt': gts})
    return benchmark_dataset


def generate_random_walk(n, start_value=0, loc=0, scale=1):
    # Generate random steps (e.g., from a normal distribution with mean 0 and standard deviation 1)
    steps = np.random.normal(loc=loc, scale=scale, size=n)
    # Compute the random walk by taking the cumulative sum of the steps
    random_walk = np.cumsum(steps)
    # Add the starting value
    random_walk = start_value + random_walk
    return random_walk
    

def generate_benchmark_ts_with_random_walk(n, ts_instances, loc=0, scale=1):

    total_motifs = sum([len(instances)  for instances in ts_instances.values()])
    total_length = sum([len(instance)-1 for instances in ts_instances.values() for instance in instances])
    
    n_random = n - total_length

    # Generate random walk of length n
    t = generate_random_walk(n_random, start_value=0, loc=loc, scale=scale)
    # Generate a set of unique random indices
    indices = np.random.choice(n_random, size=total_motifs, replace=False)
        
    i = 0
    for label, instances in ts_instances.items():
        for instance in instances:
            l = len(instance)
            index = indices[i]

            t = np.concatenate((t[:index], instance + t[index], t[index+1:]))
            indices[index<indices] += (l-1)
            
            i += 1

    gt = {label: [] for label in ts_instances.keys()}
    i = 0
    for label, instances in ts_instances.items():
        for instance in instances:
            gt[label].append((indices[i], indices[i]+len(instance)))
            i += 1

    return t, gt