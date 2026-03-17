import os
import sys 
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import DATA_DIR, REPO_DIR, DER_DIR
import h5py
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_masked_responses(stories, subject, mask):
    subject_dir = os.path.join(DATA_DIR, subject)
    resp = {}
    
    for story in stories:
        path = os.path.join(subject_dir, f"{story}.hf5")
        if not os.path.isfile(path):
            print(f"    ⚠️  Missing: {story}.hf5 → skipping")
            continue
        
        print(f"    📂 Loading {story}.hf5...", end=" ")
        with h5py.File(path, "r") as hf:
            data = hf["data"][:]
                
        if data.shape[0] < data.shape[1]:  # likely already (n_TRs, n_vox)
            data = data.T                      # → (n_vox, n_TRs)
        
        if mask is not None:
            data = data[mask, :]
        
        resp[story] = data.T  # Return as (n_TRs, n_voxels)
        print(f"✓ ({data.T.shape[0]} TRs, {data.T.shape[1]} voxels)")
    
    return resp


def load_encoding_scores(subject):
    subject_dir = os.path.join(REPO_DIR, "encoding/results/opensmile_all_stories", subject)
    scores_path = os.path.join(subject_dir, "corrs.npy")
    scores = np.load(scores_path)
    return scores


def get_top_voxel_mask(subject, percentile=95):
    scores = load_encoding_scores(subject)
    voxel_scores = scores.ravel()
    threshold = np.percentile(voxel_scores, percentile)
    mask = voxel_scores >= threshold  
    n_voxels = len(voxel_scores)
    n_selected = mask.sum()
    print(f"  📊 Selected: {n_selected:,} / {n_voxels:,} voxels "
          f"({n_selected/n_voxels*100:.1f}%) | threshold r = {threshold:.4f}")
    
    return mask, threshold


def extract_pca_targets(percentile=99, n_pca=3):
    """
    Extract brain targets using PCA for wav2vec finetuning.
    
    Args:
        percentile: Top percentile of voxels to use (by encoding score)
        n_pca: Number of principal components (recommended: 3-4)
    
    Returns:
        Dictionary with PCA components per subject/story
    """
    json_path = os.path.join(DER_DIR, "all_stories.json")    
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    subject_to_stories = data.get("participants", {})
    output = {}
    
    total_subjects = len(subject_to_stories)
    
    for subj_idx, (subject, stories) in enumerate(subject_to_stories.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{subj_idx}/{total_subjects}] Processing subject: {subject}")
        print(f"{'='*70}")
        
        mask, threshold = get_top_voxel_mask(subject, percentile=percentile)
        
        # Get correlation scores for selected voxels
        scores = load_encoding_scores(subject).ravel()
        
        print(f"\n  📖 Loading {len(stories)} stories for {subject}:")
        responses = get_masked_responses(stories, subject, mask)
        
        output[subject] = {}
        
        print(f"\n  🔬 Computing PCA for {len(responses)} stories:")
        
        for story_idx, (story, arr) in enumerate(responses.items(), 1):
            print(f"    [{story_idx}/{len(responses)}] {story}...", end=" ")
            
            # arr is (n_TRs, n_voxels)
            story_output = {}
            
            # ==========================================
            # PCA on selected voxels
            # ==========================================
            if arr.shape[1] >= n_pca:
                # Standardize each voxel's timecourse
                scaler = StandardScaler()
                arr_scaled = scaler.fit_transform(arr)
                
                pca = PCA(n_components=n_pca)
                components = pca.fit_transform(arr_scaled)  # (n_TRs, n_pca)
                
                # Store each component
                for i in range(n_pca):
                    story_output[f'PC{i+1}'] = components[:, i].tolist()
                
                # Store variance explained
                var_explained = pca.explained_variance_ratio_
                story_output['explained_variance_ratio'] = var_explained.tolist()
                
                cumsum_var = np.cumsum(var_explained)
                print(f"✓ PC1-{n_pca}: {cumsum_var[-1]*100:.1f}% variance")
            
            elif arr.shape[1] > 0:
                print(f"⚠️  Only {arr.shape[1]} voxel(s)")
                # Compute as many as possible
                n_components_possible = min(n_pca, arr.shape[1])
                if n_components_possible > 0:
                    scaler = StandardScaler()
                    arr_scaled = scaler.fit_transform(arr)
                    
                    pca = PCA(n_components=n_components_possible)
                    components = pca.fit_transform(arr_scaled)
                    
                    for i in range(n_components_possible):
                        story_output[f'PC{i+1}'] = components[:, i].tolist()
                    
                    story_output['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            
            # ==========================================
            # Metadata
            # ==========================================
            story_output['metadata'] = {
                'n_voxels_total': int(arr.shape[1]),
                'n_TRs': int(arr.shape[0]),
                'threshold_r': float(threshold),
                'n_components': len([k for k in story_output.keys() if k.startswith('PC')]),
                'total_variance_explained': float(np.sum(story_output.get('explained_variance_ratio', [])))
            }
            
            output[subject][story] = story_output
        
        # Summary for this subject
        n_stories = len(output.get(subject, {}))
        print(f"\n  ✅ Completed {subject}: {n_stories} stories processed")
    
    # Save
    out_dir = os.path.join(REPO_DIR, "brain_targets_finetuning")
    os.makedirs(out_dir, exist_ok=True)
    
    outfile = os.path.join(out_dir, f"top{percentile}pct_pca{n_pca}_targets.json")
    
    print(f"\n💾 Saving results...")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"📁 Saved: {outfile}")
    print(f"📊 Brain targets per story: {n_pca} PCA components")
    print(f"👥 Subjects processed: {len(output)} / {total_subjects}")
    
    # Calculate total stories
    total_stories = sum(len(stories) for stories in output.values())
    print(f"📖 Total stories processed: {total_stories}")
    print(f"{'='*70}")
    
    return output


if __name__ == "__main__":
    print("="*70)
    print("BRAIN TARGET EXTRACTION FOR WAV2VEC FINETUNING")
    print("="*70)
    print("Configuration:")
    print("  • Percentile: 95th (top 5% of voxels)")
    print("  • PCA components: 3")
    print("="*70)
    
    # Try with 3 components (recommended starting point)
    result = extract_pca_targets(
        percentile=95,
        n_pca=3  # Can adjust to 4 if you want more variance captured
    )
    
    # Detailed validation check
    if result:
        print(f"\n{'='*70}")
        print(f"VALIDATION: EXAMPLE OUTPUT")
        print(f"{'='*70}")
        
        first_sub = next(iter(result))
        first_story = next(iter(result[first_sub]))
        
        print(f"\n📌 Example: {first_sub} – {first_story}")
        print(f"{'-'*70}")
        
        data = result[first_sub][first_story]
        
        print("\n🎯 Available brain targets:")
        for i, key in enumerate([k for k in sorted(data.keys()) if k.startswith('PC')], 1):
            arr = data[key]
            arr_preview = arr[:3] if len(arr) >= 3 else arr
            print(f"   {i}. {key:10s} length: {len(arr):4d}  "
                  f"preview: [{', '.join(f'{x:.4f}' for x in arr_preview)}, ...]")
        
        if 'explained_variance_ratio' in data:
            print(f"\n📊 Variance explained:")
            var_ratios = data['explained_variance_ratio']
            cumsum = np.cumsum(var_ratios)
            for i, (var, cum) in enumerate(zip(var_ratios, cumsum), 1):
                print(f"   PC{i}: {var*100:5.2f}%  (cumulative: {cum*100:5.2f}%)")
        
        print(f"\n📋 Metadata:")
        meta = data['metadata']
        for k, v in sorted(meta.items()):
            print(f"   {k:30s}: {v}")
        
        # Component statistics
        print(f"\n📈 Component statistics:")
        for pc_name in [k for k in sorted(data.keys()) if k.startswith('PC')]:
            pc = np.array(data[pc_name])
            print(f"   {pc_name}: mean={pc.mean():.4f}, std={pc.std():.4f}, "
                  f"range=[{pc.min():.4f}, {pc.max():.4f}]")
        
        print(f"\n{'='*70}")
        print("✅ All done! Ready for wav2vec finetuning.")
        print(f"{'='*70}")