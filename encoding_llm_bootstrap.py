import os
import numpy as np
import argparse
import json
import logging
from os.path import join
from encoding.encoding_utils import load_embeddings, preprocess_features, get_response_mask
from encoding.ridge_utils.ridge import bootstrap_ridge
from encoding.config import REPO_DIR
import time
from nilearn import datasets, image
from nilearn.image import resample_to_img
import nibabel as nib


# Default arguments for GUI debugging
DEFAULT_ARGS = {
    "subjects": "UTS01",
    "trim": 5,
    "ndelays": 4,
    "nboots": 15,
    "chunklen": 10,
    "nchunks": None,  # Will be calculated based on data
    "modality": "text_audio",
    "singcutoff": 1e-10,
    "use_corr": True,
    "single_alpha": False,
    "use_pca": True,
    "explained_variance": 0.90,
    "not_use_attention": True,
    "use_opensmile": False,
    "corrmin": 0.0,
    "normalize_stim": False,
    "normalize_resp": True,
    "num_jobs": 1,
    "normalpha": False,
    "return_wt": False,
    "json_path": "derivative/common_stories_25_for_9_participants.json",
    "test_split": 0.2,  # Percentage of stories for test set
}

args = argparse.Namespace(**DEFAULT_ARGS)

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encoding model script with bootstrap ridge")
    parser.add_argument("--subjects", type=str, default="UTS01",
                        help="Subject ID(s): single (e.g., UTS01), comma-separated list (e.g., UTS01,UTS02), or 'all' for all subjects in JSON")
    parser.add_argument("--modality", type=str, choices=["text", "audio", "text_audio"], default="text_audio", help="Feature modality")
    parser.add_argument("--trim", type=int, default=5, help="Number of samples to trim from start/end")
    parser.add_argument("--ndelays", type=int, default=4, help="Number of delays for HRF")
    parser.add_argument("--use_pca", action="store_true", help="Apply PCA to reduce dimensionality")
    parser.add_argument("--explained_variance", type=float, default=0.90, help="Target explained variance for PCA")
    parser.add_argument("--use_corr", action="store_true", help="Use correlation instead of R-squared")
    parser.add_argument("--return_wt", action="store_true", help="Return regression weights")
    parser.add_argument("--nboots", type=int, default=15, help="Number of bootstrap iterations")
    parser.add_argument("--chunklen", type=int, default=10, help="Chunk length for bootstrap")
    parser.add_argument("--nchunks", type=int, default=None, help="Number of chunks held out (default: ~20% of training data)")
    parser.add_argument("--singcutoff", type=float, default=1e-10, help="Singular value cutoff for SVD")
    parser.add_argument("--single_alpha", action="store_true", help="Use single alpha for all voxels")
    parser.add_argument("--normalpha", action="store_true", help="Normalize alphas by largest singular value")
    parser.add_argument("--not_use_attention", action="store_true", help="Use GPT-2 attention embeddings if set")
    parser.add_argument("--use_opensmile", action="store_true", help="Use OpenSMILE audio features if set")
    parser.add_argument("--corrmin", type=float, default=0.0, help="Minimum correlation for logging")
    parser.add_argument("--normalize_stim", action="store_true", help="Z-score stimuli")
    parser.add_argument("--normalize_resp", action="store_true", help="Z-score responses")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of stories to use for testing")
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

def split_train_test(stories, test_split=0.2, seed=42):
    """Split stories into training and test sets."""
    np.random.seed(seed)
    n_stories = len(stories)
    n_test = max(1, int(n_stories * test_split))
    
    # Randomly shuffle story indices
    story_indices = np.random.permutation(n_stories)
    test_indices = story_indices[:n_test]
    train_indices = story_indices[n_test:]
    
    train_stories = [stories[i] for i in train_indices]
    test_stories = [stories[i] for i in test_indices]
    
    logging.info(f"Split {n_stories} stories: {len(train_stories)} train, {len(test_stories)} test")
    
    return train_stories, test_stories

def save_results(save_location, results):
    """Save regression results."""
    os.makedirs(save_location, exist_ok=True)
    for name, data in results.items():
        np.save(f"{save_location}/{name}", data)

def main():
    """Run the encoding model with bootstrap ridge."""
    args = parse_arguments()
    start_time = time.time()
    setup_logging()
    logging.info(f"Arguments: {vars(args)}")
    
    # Load features (shared across subjects)
    if args.not_use_attention:
        text_feat = load_embeddings("features/gpt2_attention")
    else:
        text_feat = load_embeddings("features/gpt2")
    
    if args.use_opensmile:
        audio_feat = load_embeddings("features/opensmile")
    else:
        audio_feat = load_embeddings("features/wav2vec")
    
    
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
    alphas = np.logspace(0, 3, 20)
    
    for subject in subjects:
        logging.info(f"Processing subject: {subject}")
        
        # Load and split data
        stories = load_session_data(subject, args.json_path)[0:5]
        train_stories, test_stories = split_train_test(stories, args.test_split)
        
        # Preprocess features for training set
        logging.info("Preprocessing training features...")
        Rstim, train_ids = preprocess_features(
            train_stories, text_feat, audio_feat, args.modality,
            args.trim, args.ndelays, args.use_pca, args.explained_variance
        )
        
        # Preprocess features for test set
        logging.info("Preprocessing test features...")
        Pstim, test_ids = preprocess_features(
            test_stories, text_feat, audio_feat, args.modality,
            args.trim, args.ndelays, args.use_pca, args.explained_variance
        )
        
        logging.info(f"Training stim shape: {Rstim.shape}, Test stim shape: {Pstim.shape}")
        
        # Get responses for training and test sets
        Rresp = get_response_mask(train_stories, f"sub-{subject}", voxel_indices)
        Presp = get_response_mask(test_stories, f"sub-{subject}", voxel_indices)
        
        logging.info(f"Training resp shape: {Rresp.shape}, Test resp shape: {Presp.shape}")
        
        # Calculate nchunks if not provided (default to ~20% of training data)
        if args.nchunks is None:
            nchunks = max(1, int(0.2 * Rresp.shape[0] / args.chunklen))
            logging.info(f"Calculated nchunks: {nchunks} (20% of {Rresp.shape[0]} TRs)")
        else:
            nchunks = args.nchunks
        
        # Setup save location
        save_location = join(REPO_DIR, "results", args.modality, subject)
        logging.info(f"Saving results to: {save_location}")
        
        # Perform bootstrap ridge regression
        logging.info("Running bootstrap ridge regression...")
        wt, corrs, valphas, bootstrap_corrs, valinds = bootstrap_ridge(
            Rstim=Rstim,
            Rresp=Rresp,
            Pstim=Pstim,
            Presp=Presp,
            alphas=alphas,
            nboots=args.nboots,
            chunklen=args.chunklen,
            nchunks=nchunks,
            corrmin=args.corrmin,
            joined=None,
            singcutoff=args.singcutoff,
            normalpha=args.normalpha,
            single_alpha=args.single_alpha,
            use_corr=args.use_corr,
            return_wt=args.return_wt,
            logger=logging
        )
        
        # Save results
        results = {
            "corrs": corrs,
            "valphas": valphas,
            "bootstrap_corrs": bootstrap_corrs,
            "valinds": valinds,
        }
        
        if args.return_wt and len(wt) > 0:
            results["wt"] = wt
        
        save_results(save_location, results)
        
        # Calculate and log performance metrics
        r2_score = sum(corrs * np.abs(corrs))
        mean_corr = np.mean(corrs)
        logging.info(f"Subject {subject} - Mean correlation: {mean_corr:.4f}, Total R2: {r2_score:.4f}")
        logging.info(f"Voxels with corr > 0.1: {np.sum(corrs > 0.1)}, > 0.2: {np.sum(corrs > 0.2)}")
    
    total_time = time.time() - start_time
    logging.info(f"Total analysis completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()