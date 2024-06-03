import pandas as pd
import numpy as np

def generate_tsmd_ts(df, kappa):
    freqs   = df['label'].value_counts()
    
    classes = freqs.index
    classes = classes[freqs > 1]  
    # Pick the repeating classes randomly
    repeating_classes = np.random.choice(classes, size=kappa, replace=False)  
    # Sample one instance of every non-repeating class. Then randomly order them
    non_repeating = df[~df['label'].isin(repeating_classes)].copy()
    non_repeating = non_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample())
    non_repeating = non_repeating.sample(frac=1).reset_index(drop=True)
            
    # Sample at least two motifs for each repeating class
    repeating = df[df['label'].isin(repeating_classes)].reset_index(drop=True)
    motifs = repeating.groupby('label', group_keys=False).apply(lambda x: x.sample(n=2))
    
    # Then complete the GT motifs by randomly sampling other motifs from repeating classes
    other_motifs = repeating[~repeating.apply(tuple, 1).isin(motifs.apply(tuple, 1))]
    other_motifs = other_motifs.sample(n=min(len(other_motifs), max(0, len(non_repeating)+1-len(motifs))), replace=False)

    all_motifs = pd.concat((motifs, other_motifs))    
    all_motifs = all_motifs.sample(frac=1).reset_index(drop=True)
            
    gt = {c: [] for c in repeating_classes}
    ts = []

    # Concatenate instances, alternating between a non-repeating and a repeating class, until no instances left
    length = 0
    curr   = 0
    for i in range(min(len(non_repeating), len(all_motifs) - 1)):
        # motif
        motif, label, l = all_motifs[['series', 'label', 'length']].iloc[i]
        ts.append(motif)
        gt[label].append((curr, curr+l))
        curr += l

        # non-motif
        instance, _, l = non_repeating[['series', 'label', 'length']].iloc[i]
        ts.append(instance)
        curr += l
    
    motif, label, l = all_motifs[['series', 'label', 'length']].iloc[-1]
    ts.append(motif)
    gt[label].append((curr, curr+l))    
    return np.vstack(ts), gt

def generate_tsmd_benchmark(df, nb_series, kappa_min, kappa_max):    

    # Generate time series
    tss = []
    gts = []
    for _ in range(nb_series):
        
        # Sample a number of motif sets
        kappa = np.random.randint(kappa_min, kappa_max+1)
    
        ts, gt = generate_tsmd_ts(df, kappa)
        tss.append(ts)
        gts.append(gt)
        
    benchmark = pd.DataFrame({'series': tss, 'gt': gts})
    return benchmark