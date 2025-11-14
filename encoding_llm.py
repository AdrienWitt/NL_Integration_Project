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
    "n_comps": 20,
    "optimize_alpha": True,
    "not_use_attention": False,
    "use_opensmile": True,
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
    parser.add_argument("--n_comps", type=float, default=0.90, help="Target explained variance for PCA")
    parser.add_argument("--use_corr", action="store_true", help="Use correlation instead of R-squared")
    parser.add_argument("--return_wt", action="store_true", help="Return regression weights")
    parser.add_argument("--nboots", type=int, default=25, help="Number of bootstrap iterations")
    parser.add_argument("--nsplits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--chunklen", type=int, default=12, help="Chunk length for bootstrap (unused in ridge_cv)")
    parser.add_argument("--singcutoff", type=float, default=1e-10, help="Singular value cutoff for SVD")
    parser.add_argument("--single_alpha", action="store_true", help="Use single alpha for all voxels (unused in ridge_cv)")
    parser.add_argument("--normalpha", action="store_true", help="Normalize alphas by largest singular value")
    parser.add_argument("--not_use_attention", action="store_true", help="Use simple GPT-2 embeddings if set")
    parser.add_argument("--use_opensmile", action="store_true", help="Use OpenSMILE audio features if set")
    parser.add_argument("--corrmin", type=float, default=0.0, help="Minimum correlation for logging")
    parser.add_argument("--normalize_stim", action="store_true", help="Z-score stimuli")
    parser.add_argument("--normalize_resp", action="store_true", help="Z-score responses")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--with_replacement", action="store_true", help="Sample with replacement in bootstrap")
    parser.add_argument("--optimize_alpha", action="store_true", help="Optimize alpha using bootstrapping")
    parser.add_argument("--alpha_min", type=float, default=-2,
                           help="Minimum exponent for alpha values in logspace (default: -3).")
    parser.add_argument("--alpha_max", type=float, default=3,
                           help="Maximum exponent for alpha values in logspace (default: 3).")
    parser.add_argument("--num_alphas", type=int, default=10,
                           help="Number of alpha values to test in logspace (default: 10).")
    parser.add_argument("--concat_subjects", action="store_true", help="Concatenate all subjects' data and run one joint analysis")
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
    """Run the encoding model (per subject or concatenated)."""
    args = parse_arguments()
    start_time = time.time()
    setup_logging()
    logging.info(f"Arguments: {vars(args)}")
    
    text_model_name = "gpt2_simple" if args.not_use_attention else "gpt2_attention"
    audio_model_name = "opensmile" if args.use_opensmile else "wav2vec"
    
    # Load text features
    if args.modality in ["text", "text_audio"]:
        text_feat_path = f"features/{'gpt2_attention' if args.not_use_attention else 'gpt2'}"
        logging.info(f"Loading text features from: {text_feat_path}")
        text_feat = load_embeddings(text_feat_path)
    
    # Load audio features
    if args.modality in ["audio", "text_audio"]:
        audio_feat_path = f"features/{audio_model_name}"
        logging.info(f"Loading audio features from: {audio_feat_path}")
        audio_feat = load_embeddings(audio_feat_path)
    
    if args.modality == "text":
        feature_str = text_model_name
    elif args.modality == "audio":
        feature_str = audio_model_name
    else:  # text_audio
        feature_str = f"{text_model_name}_{audio_model_name}"
    
    base_result_dir = join(REPO_DIR, "results", feature_str)
    logging.info(f"All results will be saved under: {base_result_dir}")
    os.makedirs(base_result_dir, exist_ok=True)

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

    # Define alphas (grid to search)
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)

    # nboots / nsplits defaults if None (keep behavior from your script)
    nboots = args.nboots if args.nboots is not None else None
    nsplits = args.nsplits if args.nsplits is not None else None

    # ===========================
    # CASE 1: CONCATENATE SUBJECTS
    # ===========================
    if not args.concat_subjects:
        
        for subject in subjects:
            logging.info(f"Processing subject: {subject}")

            # Load and split data
            stories = load_session_data(subject, args.json_path)

            delRstim, ids_stories = preprocess_features(
                stories, text_feat, audio_feat, args.modality,
                args.trim, args.ndelays, args.use_pca, args.n_comps
            )
            logging.info(f"delRstim shape: {delRstim.shape}")

            zRresp = get_response_mask(stories, f"sub-{subject}", voxel_indices)
            logging.info(f"zRresp shape: {zRresp.shape}")

            # Setup save location for this subject
            save_location = join(base_result_dir, subject)
            
            logging.info(f"Saving results to: {save_location}")

            # Handle precomputed valphas per subject (if optimize_alpha is False)
            if not args.optimize_alpha:
                valphas_path = join(base_result_dir, subject, "valphas.npy")
                if not os.path.exists(valphas_path):
                    raise ValueError(f"Must provide a valid valphas path for subject {subject} when --optimize_alpha is False. Expected at: {valphas_path}")
                valphas_subject = np.load(valphas_path)
                logging.info(f"Using precomputed valphas for {subject} at {valphas_path}")
            else:
                valphas_subject = None

            # Run ridge CV for subject
            _, corrs, valphas_used, fold_corrs, _ = ridge_cv(
                stim=delRstim,
                resp=zRresp,
                alphas=alphas,
                story_ids=ids_stories,
                nboots=nboots if nboots is not None else args.nboots,
                corrmin=args.corrmin,
                nsplits=nsplits if nsplits is not None else args.nsplits,
                singcutoff=args.singcutoff,
                normalpha=args.normalpha,
                use_corr=args.use_corr,
                return_wt=args.return_wt,
                normalize_stim=args.normalize_stim,
                normalize_resp=args.normalize_resp,
                n_jobs=args.num_jobs,
                with_replacement=args.with_replacement,
                optimize_alpha=args.optimize_alpha,
                valphas=valphas_subject,
                logger=logging
            )

            # Save subject results
            results = {
                "corrs": corrs,
                "valphas": valphas_used,
                "fold_corrs": fold_corrs,
            }
            save_results(save_location, results)

            r2_score = np.sum(corrs * np.abs(corrs))
            logging.info(f"Total R2 score for {subject}: {r2_score}")
        
    # ===========================
    # CASE 2: PER-SUBJECT LOOP
    # ===========================
    else:
        
        logging.info("Concatenating all subjects for a joint analysis...")

        all_stims = []
        all_resps = []
        all_story_ids = []

        for subject in subjects:
            logging.info(f"Loading and processing data for {subject}")
            stories = load_session_data(subject, args.json_path)

            delRstim, ids_stories = preprocess_features(
                stories, text_feat, audio_feat, args.modality,
                args.trim, args.ndelays, args.use_pca, args.n_comps
            )
            logging.info(f"Subject {subject} stim shape: {delRstim.shape}")

            zRresp = get_response_mask(stories, f"sub-{subject}", voxel_indices)
            logging.info(f"Subject {subject} resp shape: {zRresp.shape}")

            all_stims.append(delRstim)
            all_resps.append(zRresp)
            all_story_ids.extend(ids_stories)

        # Concatenate along time axis (axis=0)
        delRstim_all = np.concatenate(all_stims, axis=0)
        zRresp_all = np.concatenate(all_resps, axis=0)
        story_ids_all = np.array(all_story_ids)

        logging.info(f"Concatenated stim shape: {delRstim_all.shape}, resp shape: {zRresp_all.shape}")

        # Handle precomputed valphas for concatenated run if optimize_alpha is False
        if not args.optimize_alpha:
            valphas_path = join(base_result_dir, "concatenated_subjects", "valphas.npy")
            if not os.path.exists(valphas_path):
                raise ValueError(
                    f"Must provide a valid valphas path for concatenated subjects when --optimize_alpha is False. "
                    f"Expected at: {valphas_path}"
                )
            valphas_concat = np.load(valphas_path)
            logging.info(f"Using precomputed concatenated valphas at {valphas_path}")
        else:
            valphas_concat = None

        # Run ridge CV on concatenated data
        _, corrs, valphas_used, fold_corrs, _ = ridge_cv(
            stim=delRstim_all,
            resp=zRresp_all,
            alphas=alphas,
            story_ids=story_ids_all,
            nboots=nboots if nboots is not None else args.nboots,
            corrmin=args.corrmin,
            nsplits=nsplits if nsplits is not None else args.nsplits,
            singcutoff=args.singcutoff,
            normalpha=args.normalpha,
            use_corr=args.use_corr,
            return_wt=args.return_wt,
            normalize_stim=args.normalize_stim,
            normalize_resp=args.normalize_resp,
            n_jobs=args.num_jobs,
            with_replacement=args.with_replacement,
            optimize_alpha=args.optimize_alpha,
            valphas=valphas_concat,
            logger=logging
        )

        # Save concatenated results
        save_location = join(base_result_dir, "concatenated_subjects")
        results = {"corrs": corrs, "valphas": valphas_used, "fold_corrs": fold_corrs}
        save_results(save_location, results)
        logging.info(f"Saved concatenated results to: {save_location}")

        # Optional summary metric
        total_r2 = np.sum(corrs * np.abs(corrs))
        logging.info(f"Total R2 (concatenated subjects): {total_r2}")
       

    # Final summary
    total_time = time.time() - start_time
    logging.info(f"Total analysis completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    main()