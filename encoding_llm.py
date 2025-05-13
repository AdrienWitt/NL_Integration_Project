import os
import numpy as np
import argparse
import json
import logging
from os.path import join
from encoding.encoding_utils import load_embeddings, preprocess_features, get_response_mask, get_response
from encoding.ridge_utils.ridge import bootstrap_ridge
from encoding.config import REPO_DIR

# Default arguments for GUI debugging
DEFAULT_ARGS = {
    "subject": "UTS01",
    "sessions": [1, 2, 3, 4, 5],
    "trim": 5,
    "ndelays": 4,
    "nboots": 50,
    "chunklen": 40,
    "nchunks": 125,
    "modality": "text_audio",
    "singcutoff": 1e-10,
    "use_corr": False,
    "single_alpha": False,
    "use_pca": True,
    "explained_variance": 0.90
}
args = argparse.Namespace(**DEFAULT_ARGS)

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encoding model script")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--modality", type=str, default="text_audio")
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("--explained_variance", type=float, default=0.95)
    parser.add_argument("-use_corr", action="store_true")
    parser.add_argument("-single_alpha", action="store_true")
    parser.add_argument("-use_pca", action="store_true")
    return parser.parse_args()

def load_session_data(subject):
    """Load and split training and test stories."""
    with open("derivative/train_test_split.json", "r") as f:
        sess_to_story = json.load(f)

    train_stories, test_stories = [], []
    sessions = sess_to_story[f"sub-{subject}"]
    
    for sess in sessions:
        stories, tstory = sessions[sess][0], sessions[sess][1]
        train_stories.extend(stories)
        test_stories.extend(tstory)
    
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
    setup_logging()
    logging.info(f"Arguments: {vars(args)}")

    # Load features
    text_feat = load_embeddings("features/gpt2")
    audio_feat = load_embeddings("features/wav2vec")

    # Load and split data
    train_stories, test_stories = load_session_data(args.subject)
    
    train_stories = train_stories[0:3]
    
    # path = 
    # hf = h5py.File(resp_path, "r")
    # sp.extend(hf["data"][:])

    # Preprocess features
    delRstim, delPstim = preprocess_features(
        train_stories, test_stories, text_feat, audio_feat, args.modality,
        args.trim, args.ndelays, args.use_pca, args.explained_variance
    )
    
    logging.info(f"delRstim shape: {delRstim.shape}")
    logging.info(f"delPstim shape: {delPstim.shape}")

    # Get response data
    zRresp = get_response(train_stories, f"sub-{args.subject}")
    zPresp = get_response_mask( ,  f"sub-{args.subject}")
    logging.info(f"zRresp shape: {zRresp.shape}")
    logging.info(f"zPresp shape: {zPresp.shape}")

    # Setup save location
    save_location = join(REPO_DIR, "results", args.modality, args.subject)
    logging.info(f"Saving results to: {save_location}")

    # Run ridge regression
    alphas = np.logspace(1, 3, 10)
    logging.info(f"Ridge parameters: nboots={args.nboots}, chunklen={args.chunklen}, "
                 f"nchunks={args.nchunks}, single_alpha={args.single_alpha}, "
                 f"use_corr={args.use_corr}")

    wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim, zRresp, delPstim, zPresp, alphas, args.nboots, args.chunklen,
        args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha,
        use_corr=args.use_corr
    )

    # Save results
    results = {
        "weights": wt,
        "corrs": corrs,
        "valphas": valphas,
        "bscorrs": bscorrs,
        "valinds": np.array(valinds)
    }
    save_results(save_location, results)
    
    r2_score = sum(corrs * np.abs(corrs))
    logging.info(f"Total R2 score: {r2_score}")

if __name__ == "__main__":
    main()