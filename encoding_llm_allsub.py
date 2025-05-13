import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging
from multiprocessing import Pool, cpu_count

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project")

from encoding.encoding_utils import *
from encoding.feature_spaces import _FEATURE_CONFIG, get_feature_space
from encoding.ridge_utils.ridge import bootstrap_ridge
from encoding.config import REPO_DIR, EM_DATA_DIR, TEXT_EMB, AUDIO_EMB

# Default arguments
args = {
    "subjects": ["UTS01"],
    "feature": "eng1000",
    "sessions": [1, 2, 3, 4, 5],
    "trim": 5,
    "ndelays": 4,
    "nboots": 50,
    "chunklen": 40,
    "nchunks": 125,
    "singcutoff": 1e-10,
    "use_corr": False,
    "single_alpha": False,
    "modality": "text_audio",
    "n_jobs": min(cpu_count(), 4)# New argument: "text", "audio", or "both"
}

globals().update(args)

def load_stim_resp(subjects, modality):
    delRstim_list = []
    delPstim_list = []
    zRresp_subj_list = []
    zPresp_subj_list = []
    voxel_masks = []
    
    text_feat = load_embeddings("features/gpt2")
    audio_feat = load_embeddings("features/wav2vec")
    
    with open("derivative/train_test_split.json", "r") as f:
        sess_to_story = json.load(f)
    
    
    for subject in subjects : 
        train_stories, test_stories = [], []
        sessions = sess_to_story[f"sub-{subject}"]
        for sess in sessions:
            stories, tstory = sessions[sess][0], sessions[sess][1]
            train_stories.extend(stories)
            test_stories.extend(tstory)
        assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"

        # Apply HRF and prepare features based on modality
        if modality == "text":
            delRstim = apply_zscore_and_hrf(train_stories, text_feat, trim, ndelays)
            delPstim = apply_zscore_and_hrf(test_stories, text_feat, trim, ndelays)
        elif modality == "audio":
            delRstim = apply_zscore_and_hrf(train_stories, audio_feat, trim, ndelays)
            delPstim = apply_zscore_and_hrf(test_stories, audio_feat, trim, ndelays)
        else:  # both
            # Apply HRF to each modality separately
            delRstim_text = apply_zscore_and_hrf(train_stories, text_feat, trim, ndelays)
            delPstim_text = apply_zscore_and_hrf(test_stories, text_feat, trim, ndelays)
            delRstim_audio = apply_zscore_and_hrf(train_stories, audio_feat, trim, ndelays)
            delPstim_audio = apply_zscore_and_hrf(test_stories, audio_feat, trim, ndelays)            
            # Concatenate along feature dimension (axis=1)
            delRstim = np.concatenate([delRstim_text, delRstim_audio], axis=1)
            delPstim = np.concatenate([delPstim_text, delPstim_audio], axis=1)
    
        delRstim_list.append(delRstim)
        delPstim_list.append(delPstim_list)
        
        zRresp_subj, mask_subj = get_response_pad(train_stories, subject)
        zPresp_subj, _ = get_response_pad(test_stories, subject)
        voxel_masks.append(mask_subj)
                
        zRresp_subj_list.append(zRresp_subj)
        zPresp_subj_list.append(zPresp_subj)
    
    delRstim_all = np.concatenate(delRstim_list, axis=0)
    delPstim_all = np.concatenate(delPstim_list, axis=0)
    
    zRresp_all =np.concatenate(zRresp_subj_list, axis=0)
    zPresp_all = np.concatenate(zPresp_subj_list, axis=0)
    
    voxel_mask_all = np.concatenate(voxel_masks, axis=0)
    
    return delRstim_all, delPstim_all, zRresp_all, zPresp_all, voxel_mask_all

def process_subject(subject, modality, args):
    """Process a single subject's data"""
    print(f"Processing subject {subject}")
    
    # Load embeddings
    text_feat = load_embeddings("features/gpt2") if modality in ["text", "both"] else {}
    audio_feat = load_embeddings("features/wav2vec") if modality in ["audio", "both"] else {}
    
    # Load train/test split
    with open("derivative/train_test_split.json", "r") as f:
        sess_to_story = json.load(f)
    
    # Get train and test stories for this subject
    train_stories, test_stories = [], []
    sessions = sess_to_story[f"sub-{subject}"]
    for sess in sessions:
        stories, tstory = sessions[sess][0], sessions[sess][1]
        train_stories.extend(stories)
        test_stories.extend(tstory)
    
    # Remove duplicates while preserving order
    train_stories = list(dict.fromkeys(train_stories))
    test_stories = list(dict.fromkeys(test_stories))
    
    assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"

    # Apply HRF and prepare features based on modality
    if modality == "text":
        delRstim = apply_zscore_and_hrf(train_stories, text_feat, args.trim, args.ndelays)
        delPstim = apply_zscore_and_hrf(test_stories, text_feat, args.trim, args.ndelays)
    elif modality == "audio":
        delRstim = apply_zscore_and_hrf(train_stories, audio_feat, args.trim, args.ndelays)
        delPstim = apply_zscore_and_hrf(test_stories, audio_feat, args.trim, args.ndelays)
    else:  # both
        delRstim_text = apply_zscore_and_hrf(train_stories, text_feat, args.trim, args.ndelays)
        delPstim_text = apply_zscore_and_hrf(test_stories, text_feat, args.trim, args.ndelays)
        delRstim_audio = apply_zscore_and_hrf(train_stories, audio_feat, args.trim, args.ndelays)
        delPstim_audio = apply_zscore_and_hrf(test_stories, audio_feat, args.trim, args.ndelays)            
        delRstim = np.concatenate([delRstim_text, delRstim_audio], axis=1)
        delPstim = np.concatenate([delPstim_text, delPstim_audio], axis=1)

    # Get response data
    zRresp, mask = get_response_pad(train_stories, subject)
    zPresp, _ = get_response_pad(test_stories, subject)

    # Run ridge regression for this subject
    alphas = np.logspace(1, 3, 10)
    
    wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim, zRresp, delPstim, zPresp, alphas, 
        args.nboots, args.chunklen, args.nchunks,
        singcutoff=args.singcutoff, 
        single_alpha=args.single_alpha,
        use_corr=args.use_corr
    )

    # Save results for this subject
    save_location = join(REPO_DIR, "results", args.feature, f"subject_{subject}_{modality}")
    os.makedirs(save_location, exist_ok=True)

    np.savez(f"{save_location}/weights", wt)
    np.savez(f"{save_location}/corrs", corrs)
    np.savez(f"{save_location}/valphas", valphas)
    np.savez(f"{save_location}/bscorrs", bscorrs)
    np.savez(f"{save_location}/valinds", np.array(valinds))
    
    print(f"Completed processing subject {subject}")
    return subject, corrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs='+', type=str, default=["UTS01", "UTS02"],
                       help="List of subject IDs to process")
    parser.add_argument("--feature", type=str, default="eng1000")
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("-use_corr", action="store_true")
    parser.add_argument("-single_alpha", action="store_true")
    parser.add_argument("--modality", type=str, choices=["text", "audio", "both"],
                       default="both", help="Modality to use: text, audio, or both")
    parser.add_argument("--n_jobs", type=int, default=min(cpu_count(), 4),
                       help="Number of parallel jobs")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    
    # Process subjects in parallel
    with Pool(processes=args.n_jobs) as pool:
        results = pool.starmap(
            process_subject,
            [(subject, args.modality, args) for subject in args.subjects]
        )
    
    # Print summary of results
    print("\nProcessing complete. Results summary:")
    for subject, corrs in results:
        print(f"Subject {subject} - Total r2: {sum(corrs * np.abs(corrs))}")