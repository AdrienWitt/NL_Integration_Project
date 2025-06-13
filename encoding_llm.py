import os
import numpy as np
import argparse
import json
import logging
from os.path import join
from encoding.encoding_utils import load_embeddings, preprocess_features, get_response_mask, get_response, compute_thresholded_mask
from encoding.ridge_utils.ridge import bootstrap_ridge
from encoding.config import REPO_DIR
import time

# # Default arguments for GUI debugging
# DEFAULT_ARGS = {
#     "subject": "UTS01",
#     "trim": 5,
#     "ndelays": 4,
#     "nboots": 20,
#     "chunklen": 12,
#     "modality": "text_audio",
#     "singcutoff": 1e-10,
#     "use_corr": True,
#     "single_alpha": False,
#     "use_pca": True,
#     "explained_variance": 0.90
# }
# args = argparse.Namespace(**DEFAULT_ARGS)

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encoding model script")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--modality", type=str, default="text_audio")
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--explained_variance", type=float, default=0.95)

    parser.add_argument("--use_corr", action="store_true")
    parser.add_argument("--return_wt", action="store_true")
    parser.add_argument("--nboots", type=int, default=20)
    parser.add_argument("--chunklen", type=int, default=12)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("--single_alpha", action="store_true")
    parser.add_argument("--normalpha", action="store_true")
    
    return parser.parse_args()

def load_session_data(subject, json_path="derivative/stories_split.json"):
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Get subject key (e.g., 'sub-subject1')
    subject_key = f"sub-{subject}"
  
    # Extract train and test stories for the subject
    train_stories = data["participants"][subject_key]["train_stories"]
    test_stories = data["participants"][subject_key]["test_stories"]
    
    assert not set(train_stories) & set(test_stories), "Train-test overlap detected"
    
    return train_stories, test_stories

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
    text_feat = load_embeddings("features/gpt2")
    audio_feat = load_embeddings("features/wav2vec")

    # Load and split data
    train_stories, test_stories = load_session_data(args.subject)
    all_stories = train_stories + test_stories

    # Preprocess features
    delRstim, delPstim = preprocess_features(
        train_stories, test_stories, text_feat, audio_feat, args.modality,
        args.trim, args.ndelays, args.use_pca, args.explained_variance
    )
    
    TR = delRstim.shape[0]
    nchunks = int(0.2 * TR / args.chunklen)
    logging.info(f"Adjusted nchunks: {nchunks} for TR={TR}")
    
    logging.info(f"delRstim shape: {delRstim.shape}")
    logging.info(f"delPstim shape: {delPstim.shape}")
    
    mask, indices = compute_thresholded_mask(all_stories, f"sub-{args.subject}")
    zRresp = get_response_mask(train_stories, f"sub-{args.subject}", indices)
    zPresp = get_response_mask(test_stories, f"sub-{args.subject}", indices)
    
    logging.info(f"zRresp shape: {zRresp.shape}")
    logging.info(f"zPresp shape: {zPresp.shape}")

    # Setup save location
    save_location = join(REPO_DIR, "results", args.modality, args.subject)
    logging.info(f"Saving results to: {save_location}")

    # Run ridge regression
    alphas = np.logspace(2, 4, 10)
    logging.info(f"Ridge parameters: nboots={args.nboots}, chunklen={args.chunklen}, "
                 f"nchunks={nchunks}, single_alpha={args.single_alpha}, "
                 f"use_corr={args.use_corr}")

    _, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim, zRresp, delPstim, zPresp, alphas, args.nboots, args.chunklen,
        nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha,
        use_corr=args.use_corr, return_wt = args.return_wt
    )

    # Save results
    results = {
        "corrs": corrs,
        "valphas": valphas,
        "bscorrs": bscorrs,
        "valinds": np.array(valinds)
    }
    save_results(save_location, results)
    
    r2_score = sum(corrs * np.abs(corrs))
    logging.info(f"Total R2 score: {r2_score}")
    
    total_time = time.time() - start_time
    logging.info(f"Total analysis completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()