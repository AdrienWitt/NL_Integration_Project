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
from utils.ridge_utils import generate_leave_one_run_out, explainable_variance
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.scoring import correlation_score_split
from scipy.stats import zscore
from joblib import Parallel, delayed
import torch  # for explicit GPU check
import time

# Set backend once globally (will be re-set per process if needed)
backend = set_backend("torch_cuda", on_error="warn")

# Default arguments (for debugging / GUI)
DEFAULT_ARGS = {
    "subjects": "UTS01",
    "text_type": "gpt2_mean",
    "audio_type": "opensmile",
    "trim": 5,
    "ndelays": 4,
    "use_pca": False,
    "n_comps": 20,
    "num_jobs": 1,               # -1 = all cores, but careful with GPU memory
    "json_name": "train_test_split_25_stories_8_subs.json",
    "alpha_min": 1.0,
    "alpha_max": 20.0,
    "num_alphas": 20,
    "solver": "random_search",
    "n_iter": 20,
    "n_targets_batch": 200,
    "n_alphas_batch": 5,
    "n_targets_batch_refit": 200,
}
args = argparse.Namespace(**DEFAULT_ARGS)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Encoding model training with text + audio features")
    parser.add_argument("--subjects", type=str, default="UTS01",
                        help="Subject ID(s): single, comma-separated, or 'all'")
    parser.add_argument("--text_type", type=str, default="gpt2_mean")
    parser.add_argument("--audio_type", type=str, default="opensmile")
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--n_comps", type=int, default=20)
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="Number of parallel subject jobs (-1 = all cores). Careful with GPU memory!")
    parser.add_argument("--json_name", type=str, default="train_test_split_25_stories_8_subs.json")
    
    # Ridge / Kernel Ridge hyperparameters
    parser.add_argument("--alpha_min", type=float, default=1.0)
    parser.add_argument("--alpha_max", type=float, default=20.0)
    parser.add_argument("--num_alphas", type=int, default=20)
    parser.add_argument("--solver", type=str, default="random_search",
                        choices=["random_search", "grid_search", "hyper_gradient"])
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--n_targets_batch", type=int, default=200)
    parser.add_argument("--n_alphas_batch", type=int, default=5)
    parser.add_argument("--n_targets_batch_refit", type=int, default=200)

    return parser.parse_args()

def process_subject(subject, args, gpu_id):
    """
    Runs a single subject on a single GPU.
    This function is executed in a separate process.
    """
    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    backend = set_backend("torch_cuda", on_error="warn")
    logging.info(f"[{subject}] Starting on GPU {gpu_id}")

    # -----------------------------------------------------------------
    # Load features
    # -----------------------------------------------------------------
    text_feat_path = join(REPO_DIR, "features", args.text_type)
    audio_feat_path = join(REPO_DIR, "features", args.audio_type)
    logging.info(f"[{subject}] Loading text features from {text_feat_path}")
    text_feat = load_embeddings(text_feat_path)
    logging.info(f"[{subject}] Loading audio features from {audio_feat_path}")
    audio_feat = load_embeddings(audio_feat_path)

    stories = load_session_data(subject, join(DER_DIR, args.json_name))
    train_stories = stories["train"]
    test_stories = stories["test"]

    kwargs = dict(
        trim=args.trim,
        ndelays=args.ndelays,
        use_pca=args.use_pca,
        n_comps=args.n_comps,
    )

    X_text_train, onset_train = preprocess_features(train_stories, text_feat, **kwargs)
    X_audio_train, _ = preprocess_features(train_stories, audio_feat, **kwargs)
    X_text_test, _ = preprocess_features(test_stories, text_feat, **kwargs)
    X_audio_test, _ = preprocess_features(test_stories, audio_feat, **kwargs)

    X_train = np.concatenate([X_text_train, X_audio_train], axis=1).astype("float32")
    X_test  = np.concatenate([X_text_test,  X_audio_test],  axis=1).astype("float32")

    logging.info(f"[{subject}] X_train shape: {X_train.shape}")
    logging.info(f"[{subject}] X_test shape:  {X_test.shape}")
    logging.info(f"[{subject}] Text features dim: {X_text_train.shape[1]}")
    logging.info(f"[{subject}] Audio features dim: {X_audio_train.shape[1]}")

    Y_train = get_response(train_stories, subject)
    Y_test  = get_response(test_stories, subject)

    logging.info(f"[{subject}] Y_train shape: {Y_train.shape}")
    logging.info(f"[{subject}] Y_test shape:  {Y_test.shape}")

    # -----------------------------------------------------------------
    # Mask by explainable variance
    # -----------------------------------------------------------------
    ev = explainable_variance(Y_test)
    mask = ev > 0.1
    logging.info(f"[{subject}] Explainable variance mask: {mask.sum()} / {ev.size} voxels kept ({mask.sum()/ev.size:.1%})")

    Y_test = zscore(Y_test.mean(0), axis=0)
    Y_train = np.nan_to_num(Y_train)
    Y_test = np.nan_to_num(Y_test)
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    # -----------------------------------------------------------------
    # CV
    # -----------------------------------------------------------------
    cv = check_cv(generate_leave_one_run_out(X_train.shape[0], onset_train))

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)
    solver_params = dict(
        n_iter=args.n_iter,
        alphas=alphas,
        n_targets_batch=args.n_targets_batch,
        n_alphas_batch=args.n_alphas_batch,
        n_targets_batch_refit=args.n_targets_batch_refit,
        diagonalize_method='svd'
    )

    mkr = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=args.solver,
        solver_params=solver_params,
        cv=cv
    )

    preprocess = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Kernelizer(kernel="linear")
    )

    n_text = X_text_train.shape[1]
    column_kernelizer = ColumnKernelizer([
        ("text", preprocess, slice(0, n_text)),
        ("audio", preprocess, slice(n_text, X_train.shape[1])),
    ])

    pipeline = make_pipeline(column_kernelizer, mkr)

    # -----------------------------------------------------------------
    # Fit + score
    # -----------------------------------------------------------------
    logging.info(f"[{subject}] Starting model fit...")
    pipeline.fit(X_train, Y_train[:, mask])

    scores_mask = backend.to_numpy(
        pipeline.score(X_test, Y_test[:, mask])
    )
    n_voxels = Y_train.shape[1]
    scores = np.zeros(n_voxels)
    scores[mask] = scores_mask

    Y_pred_split = pipeline.predict(X_test, split=True)
    split_scores_mask = backend.to_numpy(
        correlation_score_split(Y_test[:, mask], Y_pred_split)
    )

    split_scores = np.zeros((split_scores_mask.shape[0], n_voxels))
    split_scores[:, mask] = split_scores_mask
    
    if mask.sum() > 0:
        valid_scores = scores[mask]
        logging.info(f"[{subject}] Joint scores (masked voxels, n={len(valid_scores)}): "
                     f"min={valid_scores.min():.4f}, "
                     f"max={valid_scores.max():.4f}, "
                     f"mean={valid_scores.mean():.4f}, "
                     f"median={np.median(valid_scores):.4f}, "
                     f"std={valid_scores.std():.4f}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    # In process_subject, change the call to:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    result_dir = join(SCRIPT_DIR, "results")
    result_dir = join("results", subject)  # e.g. results/UTS01
    
    save_results(
        result_dir,                            # ← directory path first
        {
            "subject": subject,                # will be saved as subject.npy (but it's a string — see note)
            "scores": scores,
            "split_scores": split_scores,
            "mask": mask,
            "ev": ev,
            "n_voxels": n_voxels,
            "n_masked": mask.sum(),
        }
    )

    duration = time.time() - start_time
    logging.info(f"[{subject}] Finished on GPU {gpu_id} | Duration: {duration:.1f} seconds ({duration/60:.1f} min)")
    return subject

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    setup_logging()
    overall_start = time.time()
    
    args = parse_arguments()

    # Subjects
    if args.subjects == "all":
        json_path = join(DER_DIR, args.json_name)
        with open(json_path) as f:
            subjects = json.load(f)["dataset_info"]["participants"]
    else:
        subjects = args.subjects.split(",")

    # GPUs
    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, "No GPUs detected"
    
    logging.info(f"Detected {n_gpus} GPUs")
    logging.info(f"Processing {len(subjects)} subjects: {', '.join(subjects)}")
    logging.info(f"Parallel jobs: {min(len(subjects), n_gpus)} (limited by # of GPUs)")

    # One job per GPU (round-robin)
    n_jobs = min(len(subjects), n_gpus)

    Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_subject)(
            subject,
            args,
            gpu_id = i % n_gpus
        )
        for i, subject in enumerate(subjects)
    )

    total_duration = time.time() - overall_start
    logging.info(f"All subjects processed | Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} min)")

if __name__ == "__main__":
    main()