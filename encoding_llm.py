import os
import numpy as np
import argparse
import json
import logging
from os.path import join
from encoding.encoding_utils import load_embeddings, preprocess_features, get_response_mask
from encoding.ridge_utils.ridge import ridge_cv
from encoding.config import REPO_DIR
import time
from nilearn import datasets, image
from nilearn.image import resample_to_img
import nibabel as nib


from encoding.feature_spaces import get_feature_space
from encoding.encoding_utils import apply_zscore_and_hrf


# Default arguments for GUI debugging
DEFAULT_ARGS = {
    "subjects": "UTS01",
    "trim": 5,
    "ndelays": 4,
    "nboots": 5,
    "nsplits": 5,
    "chunklen": 12,  # Note: Not used in ridge_cv but kept for compatibility
    "modality": "text_audio",
    "singcutoff": 1e-10,
    "use_corr": True,
    "single_alpha": False,  # Note: Not used in ridge_cv but kept for compatibility
    "use_pca": True,
    "explained_variance": 0.90,
    "optimize_alpha": True,
    "not_use_attention": True,
    "use_opensmile": False,
    "corrmin": 0.0,
    "normalize_stim": False,
    "normalize_resp": True,
    "num_jobs": 1,
    "with_replacement": False,
    "normalpha": False,
    "return_wt": False ,
    "json_path": "derivative/common_stories_25_for_9_participants.json",
    "optimize_alpha" : False
}

args = argparse.Namespace(**DEFAULT_ARGS)

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encoding model script")
    parser.add_argument("--subjects", type=str, default="UTS01",
                        help="Subject ID(s): single (e.g., UTS01), comma-separated list (e.g., UTS01,UTS02), or 'all' for all subjects in JSON")
    parser.add_argument("--modality", type=str, choices=["text", "audio", "text_audio"], default="text_audio", help="Feature modality")
    parser.add_argument("--trim", type=int, default=5, help="Number of samples to trim from start/end")
    parser.add_argument("--ndelays", type=int, default=4, help="Number of delays for HRF")
    parser.add_argument("--use_pca", action="store_true", help="Apply PCA to reduce dimensionality")
    parser.add_argument("--explained_variance", type=float, default=0.90, help="Target explained variance for PCA")
    parser.add_argument("--use_corr", action="store_true", help="Use correlation instead of R-squared")
    parser.add_argument("--return_wt", action="store_true", help="Return regression weights")
    parser.add_argument("--nboots", type=int, default=25, help="Number of bootstrap iterations")
    parser.add_argument("--nsplits", type=int, default=5, help="Number of CV folds")
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
                        default="derivative/common_stories_25_for_9_participants.json",
                        help="Path to JSON file with story IDs")
    return parser.parse_args()

def load_session_data(subject, json_path):
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Get subject key (e.g., 'sub-subject1')
    subject_key = f"sub-{subject}"
  
    # Extract train and test stories for the subject
    stories = data["participants"][subject_key]["stories"]
        
    return stories

def save_results(save_location, results):
    """Save regression results."""
    os.makedirs(save_location, exist_ok=True)
    for name, data in results.items():
        np.save(f"{save_location}/{name}", data)

def main():
    """Run the encoding model."""
    args = parse_arguments()
    start_time = time.time()
    setup_logging()
    logging.info(f"Arguments: {vars(args)}")
    
    # # Load features (shared across subjects)
    # if args.not_use_attention:
    #     text_feat = load_embeddings("features/gpt2_attention")
    # else:
    #     text_feat = load_embeddings("features/gpt2")
    
    # if args.use_opensmile:
    #     audio_feat = load_embeddings("features/opensmile")
    # else:
    #     audio_feat = load_embeddings("features/wav2vec")
    
    
    # Parse subjects
    if args.subjects == "all":
        with open(args.json_path, "r") as f:
            data = json.load(f)
        subjects = [key[4:] for key in data["participants"] if key.startswith("sub-")]
        logging.info(f"Processing all subjects: {subjects}")
    else:
        subjects = args.subjects.split(",")
        logging.info(f"Processing subjects: {subjects}")
    
    # Load shared mask and example data (outside loop for efficiency)
    icbm = datasets.fetch_icbm152_2009()
    mask_path = icbm['mask']
    mask = image.load_img(mask_path)
    exemple_data = nib.load("derivative/exemple_data/swausub-UTS01_ses-2_task-alternateithicatom_bold.nii")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')
    resampled_mask_data = resampled_mask.get_fdata()
    voxel_indices = np.where(resampled_mask_data.flatten() > 0)[0]
    
    # Alphas (shared)
    alphas = np.logspace(2, 4, 10)
    
    for subject in subjects:
        logging.info(f"Processing subject: {subject}")
        
        # Load and split data
        stories = load_session_data(subject, args.json_path)[0:5]
        
        feature = "eng1000"
        downsampled_feat = get_feature_space(feature, stories)
        delRstim, ids_stories = apply_zscore_and_hrf(stories, downsampled_feat, 5, 4)
            
        # # Preprocess features
        # delRstim, ids_stories = preprocess_features(
        #     stories, text_feat, audio_feat, args.modality,
        #     args.trim, args.ndelays, args.use_pca, args.explained_variance
        # )
        
        logging.info(f"delRstim shape: {delRstim.shape}")
        
        zRresp = get_response_mask(stories, f"sub-{subject}", voxel_indices)
        
        logging.info(f"zRresp shape: {zRresp.shape}")
        
        # Setup save location
        save_location = join(REPO_DIR, "results", args.modality, subject)
        logging.info(f"Saving results to: {save_location}")
        
        # Handle precomputed valphas
        if not args.optimize_alpha:
            valphas_path = join(REPO_DIR, "results", "text_audio", subject, "valphas.npy")
            if not os.path.exists(valphas_path):
                raise ValueError(f"Must provide a valid valphas path for subject {subject} when --optimize_alpha is False.")
            valphas = np.load(valphas_path)
            logging.info(f"Using precomputed valphas at {valphas_path}")
        else:
            valphas = None
        
        # Set nboots and nsplits based on arguments or default to number of participants
        nboots = args.nboots if args.nboots is not None else len(ids_stories)
        nsplits = args.nsplits if args.nsplits is not None else len(ids_stories)
        
        # Perform ridge regression with LOO CV
        _, corrs, valphas, fold_corrs, _ = ridge_cv(
            stim=delRstim,
            resp=zRresp,
            alphas=alphas,
            story_ids=ids_stories,
            nboots=nboots,
            corrmin=args.corrmin,
            nsplits=nsplits,
            singcutoff=1e-10,
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
        results = {
            "corrs": corrs,
            "valphas": valphas,
            "fold_corrs": fold_corrs,
        }
        save_results(save_location, results)
        
        r2_score = sum(corrs * np.abs(corrs))
        logging.info(f"Total R2 score for {subject}: {r2_score}")
    
    total_time = time.time() - start_time
    logging.info(f"Total analysis completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()