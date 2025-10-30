import os
import numpy as np
import pandas as pd
import argparse
import json
import logging
from os.path import join
from encoding.encoding_utils import load_embeddings, preprocess_features, get_response_mask, get_response, compute_thresholded_mask
from encoding.ridge_utils.ridge import ridge_cv_debug
from encoding.config import REPO_DIR
import time
from nilearn import datasets, image
from nilearn.image import resample_to_img
import nibabel as nib

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encoding model script")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., UTS01)")
    parser.add_argument("--modality", type=str, choices=["text", "audio", "text_audio"], default="text_audio", help="Feature modality")
    parser.add_argument("--trim", type=int, default=5, help="Number of samples to trim from start/end")
    parser.add_argument("--ndelays", type=int, default=4, help="Number of delays for HRF")
    parser.add_argument("--use_pca", action="store_true", help="Apply PCA to reduce dimensionality")
    parser.add_argument("--explained_variance", type=float, default=0.90, help="Target explained variance for PCA")
    parser.add_argument("--use_corr", action="store_true", help="Use correlation instead of R-squared")
    parser.add_argument("--return_wt", action="store_true", help="Return regression weights")
    parser.add_argument("--nboots", type=int, default=20, help="Number of bootstrap iterations")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--chunklen", type=int, default=12, help="Chunk length for bootstrap (unused in ridge_cv)")
    parser.add_argument("--singcutoff", type=float, default=1e-10, help="Singular value cutoff for SVD")
    parser.add_argument("--single_alpha", action="store_true", help="Use single alpha for all voxels (unused in ridge_cv)")
    parser.add_argument("--normalpha", action="store_true", help="Normalize alphas by largest singular value")
    parser.add_argument("--not_use_attention", action="store_true", help="Use GPT-2 attention embeddings if set")
    parser.add_argument("--use_opensmile", action="store_true", help="Use OpenSMILE audio features if set")
    parser.add_argument("--corrmin", type=float, default=0.0, help="Minimum correlation for logging")
    parser.add_argument("--normalize_stim", action="store_true", help="Z-score stimuli")
    parser.add_argument("--normalize_resp", action="store_true", help="Z-score responses")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--with_replacement", action="store_true", help="Sample with replacement in bootstrap")
    parser.add_argument("--optimize_alpha", action="store_true", help="Optimize alpha using bootstrapping")
    parser.add_argument("--json_path", type=str, 
                       default="derivative/common_stories_25.json",
                       help="Path to JSON file with story IDs")
    return parser.parse_args()

def load_session_data(subject, json_path):
    """Load stories for a given subject from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    subject_key = f"sub-{subject}"
    stories = data["participants"][subject_key]["stories"]
    logging.info(f"Loaded {len(stories)} stories for subject {subject}: {stories}")
    return stories

def save_results(save_location, results):
    """Save regression results."""
    os.makedirs(save_location, exist_ok=True)
    for name, data in results.items():
        np.savez(f"{save_location}/{name}", data)

def main():
    """Run the encoding model."""
    args = parse_arguments()
    start_time = time.time()
    setup_logging()
    logging.info(f"Arguments: {vars(args)}")

    # Load features
    logging.info("Loading text and audio features...")
    if args.not_use_attention:
        text_feat = load_embeddings("features/gpt2_attention")
    else:
        text_feat = load_embeddings("features/gpt2")
    
    if args.use_opensmile:
        audio_feat = load_embeddings("features/opensmile")
    else:
        audio_feat = load_embeddings("features/wav2vec")

    # Validate features
    logging.info("Validating feature data...")
    text_values = np.vstack(list(text_feat.values()))
    audio_values = np.vstack(list(audio_feat.values()))
    if np.any(np.isnan(text_values)) or np.any(np.isinf(text_values)):
        logging.warning(f"NaN/Inf in text_feat: {np.sum(np.isnan(text_values))} NaNs, {np.sum(np.isinf(text_values))} Infs")
    if np.any(np.isnan(audio_values)) or np.any(np.isinf(audio_values)):
        logging.warning(f"NaN/Inf in audio_feat: {np.sum(np.isnan(audio_values))} NaNs, {np.sum(np.isinf(audio_values))} Infs")

    # Load and split data
    logging.info("Loading session data...")
    stories = load_session_data(args.subject, args.json_path)[0:5]

    # Preprocess features
    logging.info("Preprocessing features...")
    delRstim, ids_stories = preprocess_features(
        stories, text_feat, audio_feat, args.modality,
        args.trim, args.ndelays, args.use_pca, args.explained_variance
    )
    
    # Validate delRstim and ids_stories
    logging.info("Validating preprocessed features...")
    if np.any(np.isnan(delRstim)) or np.any(np.isinf(delRstim)):
        logging.warning(f"NaN/Inf in delRstim: {np.sum(np.isnan(delRstim))} NaNs, {np.sum(np.isinf(delRstim))} Infs")
    if np.any(np.isnan(ids_stories)) or np.any(np.isinf(ids_stories)):
        logging.warning(f"NaN/Inf in ids_stories: {np.sum(np.isnan(ids_stories))} NaNs, {np.sum(np.isinf(ids_stories))} Infs")
    stim_var = np.var(delRstim, axis=0, ddof=1)
    zero_var_stim = np.sum(stim_var == 0)
    if zero_var_stim > 0:
        logging.warning(f"Found {zero_var_stim} features with zero variance in delRstim")
    logging.info(f"delRstim shape: {delRstim.shape}, ids_stories shape: {ids_stories.shape}")
    logging.info(f"Unique story IDs after preprocessing: {np.unique(ids_stories)}")

    # Load and preprocess fMRI data
    logging.info("Loading and preprocessing fMRI data...")
    icbm = datasets.fetch_icbm152_2009()
    mask_path = icbm['mask']
    mask = image.load_img(mask_path)
    exemple_data = nib.load("ds003020/sub-UTS01/ses-2/func/sub-UTS01_ses-2_task-alternateithicatom_bold.nii.gz")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')
    resampled_mask_data = resampled_mask.get_fdata()
    voxel_indices = np.where(resampled_mask_data.flatten() > 0)[0]  # Limit to 1000 voxels
    logging.info(f"Number of selected voxels: {len(voxel_indices)}")
    
    logging.info("Loading fMRI responses via get_response_mask...")
    zRresp = get_response_mask(stories, f"sub-{args.subject}", voxel_indices)
    
    # Validate zRresp
    logging.info("Validating fMRI response data...")
    if np.any(np.isnan(zRresp)) or np.any(np.isinf(zRresp)):
        logging.warning(f"NaN/Inf in zRresp: {np.sum(np.isnan(zRresp))} NaNs, {np.sum(np.isinf(zRresp))} Infs")
    resp_var = np.var(zRresp, axis=0, ddof=1)
    nan_var_resp = np.sum(np.isnan(resp_var))
    zero_var_resp = np.sum(resp_var == 0)
    if nan_var_resp > 0:
        logging.warning(f"Found {nan_var_resp} voxels with NaN variance in zRresp")
    if zero_var_resp > 0:
        logging.warning(f"Found {zero_var_resp} voxels with zero variance in zRresp")
    logging.info(f"zRresp shape: {zRresp.shape}")

    # Setup save location
    save_location = join(REPO_DIR, "results", args.modality, args.subject)
    logging.info(f"Saving results to: {save_location}")

    # Run ridge regression
    logging.info("Running ridge regression...")
    alphas = np.logspace(2, 4, 10)
    
    # Handle precomputed valphas
    if not args.optimize_alpha:
        valphas_path = join(REPO_DIR, "results", args.modality, args.subject, "valphas_text_audio.npy")
        if not os.path.exists(valphas_path):
            raise ValueError("Must provide a valid --precomputed_valphas path when --optimize_alpha is False.")
        valphas = np.load(valphas_path)
        logging.info(f"Using precomputed valphas at {valphas_path}")
    else:
        valphas = None

    # Set nboots and n_splits
    nboots = args.nboots if args.nboots is not None else len(np.unique(ids_stories))
    n_splits = args.n_splits if args.n_splits is not None else len(np.unique(ids_stories))    
    logging.info(f"Using nboots={nboots}, n_splits={n_splits}")

    # Perform ridge regression with LOO CV
    weights, corrs, valphas, fold_corrs, bootstrap_corrs = ridge_cv_debug(
        stim=delRstim, 
        resp=zRresp, 
        alphas=alphas, 
        story_ids=ids_stories, 
        nboots=nboots, 
        corrmin=args.corrmin,
        n_splits=n_splits, 
        singcutoff=args.singcutoff, 
        normalpha=args.normalpha,
        use_corr=args.use_corr, 
        return_wt=args.return_wt,
        normalize_stim=args.normalize_stim, 
        normalize_resp=args.normalize_resp, 
        n_jobs=args.num_jobs, 
        with_replacement=args.with_replacement,
        optimize_alpha=args.optimize_alpha,
        valphas=valphas,
        logger=logging
    )
    
    # Save results
    logging.info("Saving results...")
    results = {
        "weights": weights,
        "corrs": corrs,
        "valphas": valphas,
        "fold_corrs": fold_corrs,
        "bootstrap_corrs": bootstrap_corrs
    }
    save_results(save_location, results)
    
    r2_score = np.nansum(corrs * np.abs(corrs)) if corrs.size > 0 else np.nan
    logging.info(f"Total R2 score: {r2_score}")
    
    total_time = time.time() - start_time
    logging.info(f"Total analysis completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()