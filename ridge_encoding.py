import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import nibabel as nib
from Huth.encoding.ridge_utils.ridge import bootstrap_ridge
from Huth.encoding.ridge_utils.DataSequence import DataSequence
from Huth.encoding.ridge_utils.textgrid import TextGrid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
args = {
    "subject": "default_subject",
    "feature": "gpt2",
    "sessions": [1, 2, 3, 4, 5],
    "trim": 5,
    "ndelays": 4,
    "nboots": 50,
    "chunklen": 40,
    "nchunks": 125,
    "singcutoff": 1e-10,
    "use_corr": False,
    "single_alpha": False,
    "n_workers": multiprocessing.cpu_count()  # Number of parallel workers
}
globals().update(args)

def load_mask(mask_file):
    """Load the brain mask and get the indices of non-zero voxels."""
    mask = nib.load(mask_file).get_fdata()
    voxel_indices = np.nonzero(mask.flatten())[0]
    return mask, voxel_indices

def reconstruct_3d_image(voxel_data, mask, voxel_indices):
    """Reconstruct 3D image from voxel data using the mask."""
    # Create empty 3D array with same shape as mask
    image_3d = np.zeros_like(mask)
    
    # Flatten the 3D array
    flat_image = image_3d.flatten()
    
    # Place voxel data at the correct indices
    flat_image[voxel_indices] = voxel_data
    
    # Reshape back to 3D
    image_3d = flat_image.reshape(mask.shape)
    
    return image_3d

def process_voxel_chunk(chunk_data, feature_data, delays, nboots, chunklen, nchunks, singcutoff, use_corr, single_alpha):
    """Process a chunk of voxels in parallel."""
    voxel_results = []
    for voxel_data in chunk_data:
        # Perform ridge regression for this voxel
        result = bootstrap_ridge(
            feature_data, voxel_data, delays,
            nboots=nboots, chunklen=chunklen, nchunks=nchunks,
            singcutoff=singcutoff, use_corr=use_corr, single_alpha=single_alpha
        )
        voxel_results.append(result)
    return voxel_results

def parallel_ridge_encoding(feature_data, voxel_data, delays, n_workers=None):
    """Perform ridge regression encoding in parallel."""
    if n_workers is None:
        n_workers = args["n_workers"]
    
    # Calculate chunk size for parallel processing
    n_voxels = voxel_data.shape[0]
    chunk_size = max(1, n_voxels // (n_workers * 4))  # Divide into smaller chunks for better load balancing
    
    # Prepare chunks of voxels
    voxel_chunks = [voxel_data[i:i + chunk_size] for i in range(0, n_voxels, chunk_size)]
    
    logger.info(f"Processing {n_voxels} voxels in {len(voxel_chunks)} chunks using {n_workers} workers")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_voxel_chunk,
        feature_data=feature_data,
        delays=delays,
        nboots=args["nboots"],
        chunklen=args["chunklen"],
        nchunks=args["nchunks"],
        singcutoff=args["singcutoff"],
        use_corr=args["use_corr"],
        single_alpha=args["single_alpha"]
    )
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunk_results = list(executor.map(process_func, voxel_chunks))
    
    # Combine results from all chunks
    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)
    
    return np.array(all_results)

def process_subject(subject, feature_data, args):
    """Process a single subject's data."""
    logger.info(f"Processing subject {subject}...")
    
    # Define paths for this subject
    data_dir = join(dirname(__file__), "Text")
    output_dir = join(dirname(__file__), "results", args["feature"], subject)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load brain mask for this subject
    mask_file = join(data_dir, f"sub-{subject}_mask.nii.gz")
    mask, voxel_indices = load_mask(mask_file)
    logger.info(f"Loaded mask for subject {subject} with {len(voxel_indices)} non-zero voxels")
    
    # Load fMRI data for this subject
    logger.info(f"Loading fMRI data for subject {subject}...")
    fmri_data = {}
    for session in args["sessions"]:
        fmri_file = join(data_dir, f"sub-{subject}_ses-{session:02d}_task-{args['feature']}_bold.nii.gz")
        fmri_data[session] = nib.load(fmri_file).get_fdata()
    
    # Create delays for FIR model
    delays = np.arange(args["ndelays"])
    
    # Process each session
    for session in args["sessions"]:
        logger.info(f"Processing session {session} for subject {subject}...")
        
        # Get voxel data for this session
        voxel_data = fmri_data[session]
        
        # Perform parallel ridge regression
        results = parallel_ridge_encoding(
            feature_data,
            voxel_data,
            delays,
            n_workers=args["n_workers"]
        )
        
        # Reconstruct 3D images for each TR
        logger.info(f"Reconstructing 3D images for subject {subject}, session {session}...")
        n_trs = results.shape[0]
        reconstructed_images = []
        
        for tr in range(n_trs):
            # Get voxel data for this TR
            tr_voxel_data = results[tr]
            
            # Reconstruct 3D image
            image_3d = reconstruct_3d_image(tr_voxel_data, mask, voxel_indices)
            reconstructed_images.append(image_3d)
        
        # Convert to 4D array (TR x X x Y x Z)
        reconstructed_images = np.array(reconstructed_images)
        
        # Save results
        output_file = join(output_dir, f"session_{session}_ridge_results.nii.gz")
        nifti_img = nib.Nifti1Image(reconstructed_images, np.eye(4))  # Using identity affine matrix
        nib.save(nifti_img, output_file)
        logger.info(f"Saved results for subject {subject}, session {session}")
        
        # Also save the raw results
        raw_output_file = join(output_dir, f"session_{session}_ridge_results.hf5")
        with h5py.File(raw_output_file, 'w') as f:
            f.create_dataset('results', data=results)
        logger.info(f"Saved raw results for subject {subject}, session {session}")

def main():
    # Define paths
    data_dir = join(dirname(__file__), "Text")
    
    # Get list of subjects from the ds003020 directory
    subjects = [d for d in os.listdir("ds003020") if d.startswith("sub-")]
    subjects = [s.replace("sub-", "") for s in subjects]
    
    logger.info(f"Found {len(subjects)} subjects: {subjects}")
    
    # Load feature data (GPT-2 embeddings) - this is shared across subjects
    logger.info("Loading feature data...")
    feature_data = {}
    output_dir = join(dirname(__file__), "results", args["feature"], subjects[0])  # Use first subject's dir
    for story in os.listdir(join(output_dir)):
        if story.endswith("_gpt2_embeddings.hf5"):
            with h5py.File(join(output_dir, story), 'r') as f:
                feature_data[story] = f['data'][:]
    
    # Process subjects in parallel
    n_workers = min(args["n_workers"], len(subjects))
    logger.info(f"Processing {len(subjects)} subjects using {n_workers} workers")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create partial function with fixed arguments
        process_func = partial(process_subject, feature_data=feature_data, args=args)
        # Process all subjects
        list(executor.map(process_func, subjects))
    
    logger.info("Finished processing all subjects")

if __name__ == "__main__":
    main() 