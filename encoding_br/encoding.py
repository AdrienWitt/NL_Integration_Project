import os
import numpy as np
import argparse
import json
import logging
from os.path import join
from utils.config import REPO_DIR, DER_DIR
from utils.loader import load_embeddings, load_session_data, get_response
from utils.io import save_results
from utils.preprocess import preprocess_features
from sklearn.model_selection import check_cv
from utils.ridge_utils import generate_leave_one_run_out, explainable_variance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from himalaya.backend import set_backend
backend = set_backend("torch_cuda", on_error="warn")
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from sklearn import set_config
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.scoring import r2_score_split, correlation_score_split
from scipy.stats import zscore


# Default arguments for GUI debugging
DEFAULT_ARGS = {
    "subjects": "UTS01",
    "text_type": "gpt2_mean",
    "audio_type": "opensmile",
    "trim": 5,
    "ndelays": 4,
    "use_pca": False,
    "n_comps": 20,
    "audio_type": "opensmile",
    "normalize_stim": False,
    "normalize_resp": True,
    "num_jobs": 1,
    "json_name": "train_test_split_25_stories_8_subs.json",
    "alpha_min" : 1,
    "alpha_max": 20,
    "num_alphas" : 20}

args = argparse.Namespace(**DEFAULT_ARGS)

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encoding model script")
    parser.add_argument("--subjects", type=str, default="UTS01",
                        help="Subject ID(s): single (e.g., UTS01), comma-separated list (e.g., UTS01,UTS02), or 'all' for all subjects in JSON")
    parser.add_argument("--text_type", type=str, help="Type of text embeddings", default="gpt2_mean")
    parser.add_argument("--audio_type", type=str, help="Type of audio features", default="opensmile")
    parser.add_argument("--trim", type=int, default=5, help="Number of samples to trim from start/end")
    parser.add_argument("--ndelays", type=int, default=4, help="Number of delays for HRF")
    parser.add_argument("--use_pca", action="store_true", help="Apply PCA to reduce dimensionality")
    parser.add_argument("--n_comps", type=int, default=20, help="Number of PCA components")
    parser.add_argument("--normalize_stim", action="store_true", help="Z-score stimuli")
    parser.add_argument("--normalize_resp", action="store_true", help="Z-score responses")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--json_name", type=str,
                        default="train_test_split_25_stories_8_subs.json",
                        help="Path to JSON file with story IDs")
    parser.add_argument("--alpha_min", type=float, default=1,
                           help="Minimum alpha value for ridge regression")
    parser.add_argument("--alpha_max", type=float, default=20,
                           help="Maximum alpha value for ridge regression")
    parser.add_argument("--num_alphas", type=int, default=20,
                           help="Number of alpha values to test")
    return parser.parse_args()

    
def main():
    
    text_feat_path = join(REPO_DIR, "features", args.text_type)
    logging.info(f"Loading text features from: {text_feat_path}")
    text_feat = load_embeddings(text_feat_path)
    
    audio_feat_path = join(REPO_DIR, "features", args.audio_type)
    logging.info(f"Loading audio features from: {audio_feat_path}")
    audio_feat = load_embeddings(audio_feat_path)
    
    base_result_dir = "results"
    logging.info(f"All results will be saved under: {base_result_dir}")
    os.makedirs(base_result_dir, exist_ok=True)
    
    # Parse subjects
    if args.subjects == "all":
        with open(args.json_name, "r") as f:
            data = json.load(f)
        
        subjects = data["dataset_info"]["participants"]
        logging.info(f"Processing all subjects: {subjects}")
    else:
        subjects = args.subjects.split(",")
        logging.info(f"Processing subjects: {subjects}")
            
        
        
    for subject in subjects:
        logging.info(f"Processing subject: {subject}")
        
        subject = subjects[0]
        
        # Load and split data
        stories = load_session_data(subject, join(DER_DIR, args.json_name))
        train_stories = stories["train"]
        test_stories = stories["test"]
        
        kwargs = dict(trim=args.trim, ndelays=args.ndelays, use_pca=args.use_pca, n_comps=args.n_comps)
        
        # Process train
        X_text_train, onset_train = preprocess_features(train_stories, text_feat, **kwargs)
        X_audio_train, _ = preprocess_features(train_stories, audio_feat, **kwargs)
        
        # Process test
        X_text_test, _ = preprocess_features(test_stories, text_feat, **kwargs)
        X_audio_test, _ = preprocess_features(test_stories, audio_feat, **kwargs)
        
        
        X_train = np.concatenate([X_text_train, X_audio_train], axis=1)
        X_test = np.concatenate([X_text_test, X_audio_test], axis=1)
        
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

    
        Y_train = get_response(train_stories, f"{subject}")
        
        Y_test = get_response(test_stories, f"{subject}")
        
        ev = explainable_variance(Y_test)
        logging.info("(n_voxels,) =", ev.shape)
        
        mask = ev > 0.1
        logging.info("(n_voxels_mask,) =", ev[mask].shape)
        
        Y_test = Y_test.mean(0)
        Y_test = zscore(Y_test, axis=0)

        Y_train = np.nan_to_num(Y_train)
        Y_test = np.nan_to_num(Y_test)
        
        Y_train -= Y_train.mean(0)
        Y_test -= Y_test.mean(0)
        
        n_samples_train = X_train.shape[0]
        cv = generate_leave_one_run_out(n_samples_train, onset_train)
        cv = check_cv(cv)
        
        solver = "random_search"
        n_iter = 20
        alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)
        
        n_targets_batch = 200
        n_alphas_batch = 5
        n_targets_batch_refit = 200
        
        solver_params = dict(n_iter=n_iter, alphas=alphas,
                     n_targets_batch=n_targets_batch,
                     n_alphas_batch=n_alphas_batch,
                     n_targets_batch_refit=n_targets_batch_refit)
    
        mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                          solver_params=solver_params, cv=cv)
            
        set_config(display='diagram')
        
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            Kernelizer(kernel="linear"),
        )
        
        n_text   = X_text_train.shape[1]
        n_audio  = X_audio_train.shape[1]
        
        # Create the slices directly
        slices = [
            slice(0,          n_text),              
            slice(n_text,     n_text + n_audio)]
        
        kernelizers_tuples = [("text",  preprocess_pipeline, slice(0, X_text_train.shape[1])),
                              ("audio", preprocess_pipeline, slice(X_text_train.shape[1], X_train.shape[1])),]
        
        column_kernelizer = ColumnKernelizer(kernelizers_tuples)
        
        pipeline = make_pipeline(
            column_kernelizer,
            mkr_model)
        
        # add mask explainable variance ?
        pipeline.fit(X_train, Y_train[:, mask])
        
        scores_mask = pipeline.score(X_test,  Y_test[:, mask])
        scores_mask = backend.to_numpy(scores_mask)
        print("(n_voxels_mask,) =", scores_mask.shape)
        
        # # Then we extend the scores to all voxels, giving a score of zero to unfitted
        # # voxels.
        n_voxels = Y_train.shape[1]
        scores = np.zeros(n_voxels)
        scores[mask] = scores_mask
        print("(n_voxels,) =", scores.shape)
        
        Y_test_pred_split = pipeline.predict(X_test, split=True)
        split_scores_mask = correlation_score_split(Y_test[:, mask], Y_test_pred_split)
        split_scores = backend.to_numpy(split_scores_mask)
        
        n_kernels = split_scores_mask.shape[0]
        n_voxels = Y_train.shape[1]
        split_scores = np.zeros((n_kernels, n_voxels))
        split_scores[:, mask] = backend.to_numpy(split_scores_mask)
        print("(n_kernels, n_voxels) =", split_scores.shape)



              
          
          



        
        
        
        
    
        
        
    




