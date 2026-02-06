import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
from .delayer import Delayer

def preprocess_features(
    stories,
    feat_dict,          # dict: {story_id: array (T_s, n_features)}
    trim,
    ndelays,
    use_pca=False,
    n_comps=0.90
):
    """
    Preprocess stimulus features for encoding models:
    - Trim start/end of each story
    - Global z-scoring across all stories
    - Optional global PCA
    - Apply temporal delays **per story** (no cross-story leakage)
    
    Args:
        stories (list): List of story IDs (must match keys in feat_dict)
        feat_dict (dict): {story_id: np.ndarray (T_s, n_features)}
        trim (int): Number of samples to trim from start AND end of each story
        ndelays (int): Number of positive delays (delays = 1, 2, ..., ndelays)
        use_pca (bool): Whether to apply PCA dimensionality reduction
        n_comps (float or int): 
            - float (0 < n_comps <= 1): target explained variance ratio
            - int: exact number of components to keep
    
    Returns:
        tuple: (delayed_features, story_onsets)
            - delayed_features : np.ndarray (total_timepoints, n_features × ndelays)
            - story_onsets     : np.ndarray of starting time indices for each story
    """
    # Filter stories that exist in the dictionary
    valid_stories = [s for s in stories if s in feat_dict]
    if not valid_stories:
        raise ValueError("No valid stories found in feature dictionary.")
    
    skipped = set(stories) - set(valid_stories)
    if skipped:
        print(f"Warning: stories {skipped} not found in feat_dict → skipped.")

    delays = list(range(1, ndelays + 1))

    # === 1. Trim each story ===
    trimmed_blocks = []
    story_lengths = []

    for story_id in valid_stories:
        arr = feat_dict[story_id]
        if arr.shape[0] < 2 * trim + 1:
            raise ValueError(
                f"Story {story_id}: length {arr.shape[0]} too short "
                f"for trim={trim} (need at least {2*trim + 1} samples)"
            )
        
        trimmed = arr[5 + trim : -trim] if trim > 0 else arr.copy()
        trimmed_blocks.append(trimmed)
        story_lengths.append(trimmed.shape[0])

    # === 2. Concatenate for global normalization ===
    all_features = np.vstack(trimmed_blocks)  # (total_T, n_features)

    # === 3. Global z-scoring ===
    all_features_z = zscore(all_features, axis=0, ddof=0)

    # === 4. Optional global PCA ===
    if use_pca:
        if isinstance(n_comps, float) and 0 < n_comps <= 1:
            pca = PCA(n_components=n_comps)
        else:
            pca = PCA(n_components=int(n_comps))

        print(f"Running PCA (target: {n_comps})...")
        all_features_z = pca.fit_transform(all_features_z)
        print(f"→ kept {all_features_z.shape[1]} components "
              f"({pca.explained_variance_ratio_.sum():.3f} variance explained)")

    # === 5. Apply delays per story ===
    delayer = Delayer(delays=delays)
    # Fit on first story shape (just needs feature dimension)
    delayer.fit(all_features_z[:story_lengths[0]])

    delayed_blocks = []
    story_onsets = [0]
    current_position = 0

    for length in story_lengths:
        block = all_features_z[current_position : current_position + length]
        delayed_block = delayer.transform(block)
        delayed_blocks.append(delayed_block)
        
        current_position += length
        story_onsets.append(current_position)

    # Final output
    delayed_features = np.vstack(delayed_blocks)
    story_onsets = np.array(story_onsets[:-1])  # remove final (total length)

    return delayed_features, story_onsets



