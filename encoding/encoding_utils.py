import numpy as np
import time
import pathlib
import os
import h5py
from multiprocessing.pool import ThreadPool
from os.path import join, dirname
import nibabel as nib

from encoding.ridge_utils.npp import zscore, mcorr
from encoding.ridge_utils.utils import make_delayed
from encoding.config import DATA_DIR
from sklearn.decomposition import PCA

 
def apply_zscore_and_hrf(stories, downsampled_feat, trim, ndelays):
	"""Get (z-scored and delayed) stimulus for train and test stories.
	The stimulus matrix is delayed (typically by 2,4,6,8 secs) to estimate the
	hemodynamic response function with a Finite Impulse Response model.

	Args:
		stories: List of stimuli stories.

	Variables:
		downsampled_feat (dict): Downsampled feature vectors for all stories.
		trim: Trim downsampled stimulus matrix.
		delays: List of delays for Finite Impulse Response (FIR) model.

	Returns:
		delstim: <float32>[TRs, features * ndelays]
	"""
	stim = [zscore(downsampled_feat[s][5+trim:-trim]) for s in stories]
	stim = np.vstack(stim)
	delays = range(1, ndelays+1)
	delstim = make_delayed(stim, delays)
	return delstim

def preprocess_features(stories, text_feat, audio_feat, modality, trim, ndelays, use_pca=False, explained_variance=0.90):
    """Preprocess features: trim, z-score, PCA, and HRF for a single list of stories.

    Args:
        stories (list): List of story IDs (e.g., 25 or 27 common stories).
        text_feat (dict): Dictionary mapping story IDs to text feature arrays.
        audio_feat (dict): Dictionary mapping story IDs to audio feature arrays.
        modality (str): One of 'text', 'audio', or 'text_audio'.
        trim (int): Number of samples to trim from start/end.
        ndelays (int): Number of delays for HRF.
        use_pca (bool): If True, apply PCA to reduce dimensionality.
        explained_variance (float): Target explained variance for PCA (e.g., 0.90).

    Returns:
        tuple: (delRstim, story_ids)
            - delRstim: Processed feature matrix, shape (T, N) or (T, N_text + N_audio) for text_audio.
            - story_ids: Array of numerical story IDs (0-based indices) for each time point, shape (T,).
    """
    # Validate inputs
    valid_stories = [s for s in stories if s in text_feat and s in audio_feat]
    if not valid_stories:
        raise ValueError("No valid stories found in feature dictionaries.")
    for story in stories:
        if story not in valid_stories:
            print(f"Story {story} not found in feature dictionaries, skipping.")

    # Initialize delays
    delays = range(1, ndelays + 1)

    # Process text features
    text_stim = [text_feat[s][5 + trim:-trim] for s in valid_stories]
    text_sample_counts = [t.shape[0] for t in text_stim]  # Number of samples per story
    text_concat = np.vstack(text_stim)  # Shape: (total_samples, n_features)
    text_concat_z = zscore(text_concat)  # Z-score concatenated features

    if use_pca:
        print(f"Applying PCA to text features with explained variance threshold: {explained_variance}")
        pca_text = PCA(n_components=explained_variance)
        pca_text.fit(text_concat_z)
        n_components_text = pca_text.n_components_
        print(f"Text features: {n_components_text} components selected for {explained_variance:.2f} explained variance")
        text_concat_z = pca_text.transform(text_concat_z)  # Shape: (total_samples, n_components)

    # Process audio features
    audio_stim = [audio_feat[s][5 + trim:-trim] for s in valid_stories]
    audio_sample_counts = [a.shape[0] for a in audio_stim]
    audio_concat = np.vstack(audio_stim)
    audio_concat_z = zscore(audio_concat)

    if use_pca:
        print(f"Applying PCA to audio features with explained variance threshold: {explained_variance}")
        pca_audio = PCA(n_components=explained_variance)
        pca_audio.fit(audio_concat_z)
        n_components_audio = pca_audio.n_components_
        print(f"Audio features: {n_components_audio} components selected for {explained_variance:.2f} explained variance")
        audio_concat_z = pca_audio.transform(audio_concat_z)

    # Generate numerical story_ids
    story_ids = []
    for i, _ in enumerate(valid_stories):
        story_ids.extend([i] * text_sample_counts[i])  # Use numerical index
    story_ids = np.array(story_ids, dtype=int)  # Shape: (total_samples,)

    # Verify sample counts match between text and audio
    if not np.all(np.array(text_sample_counts) == np.array(audio_sample_counts)):
        raise ValueError("Text and audio features have mismatched sample counts after trimming.")

    # Apply HRF based on modality
    if modality == "text":
        delRstim = make_delayed(text_concat_z, delays)
    elif modality == "audio":
        delRstim = make_delayed(audio_concat_z, delays)
    else:  # text_audio
        delRstim_text = make_delayed(text_concat_z, delays)
        delRstim_audio = make_delayed(audio_concat_z, delays)
        delRstim = np.concatenate([delRstim_text, delRstim_audio], axis=1)

    return delRstim, story_ids


# def get_response(stories, subject):
# 	"""Get the subject"s fMRI response for stories."""
# 	main_path = pathlib.Path(__file__).parent.parent.resolve()
# 	subject_dir = join(DATA_DIR, "ds003020/derivative/preprocessed_data/%s" % subject)
# 	base = os.path.join(main_path, subject_dir)
# 	resp = []
# 	for story in stories:
# 		resp_path = os.path.join(base, "%s.hf5" % story)
# 		hf = h5py.File(resp_path, "r")
# 		resp.extend(hf["data"][:])
# 		hf.close()
# 	return np.array(resp)

def get_response(stories, subject):
    """Get the subject"s fMRI response for stories."""
    #subject_dir = join(DATA_DIR, "ds003020/derivative/data_normalized/preprocessed_data/%s" % subject)
    #base_path = "E:/NL/ds003020/derivative/data_normalized"
    subject_dir = os.path.join(DATA_DIR, "preprocessed_data", subject)
    resp = []
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        resp.extend(hf["data"][:])
        hf.close()
    return np.array(resp)

import logging

def get_response_mask(stories, subject, voxel_indices):
    fmri_dir = os.path.join(DATA_DIR, subject)
    resp = []

    for story in stories:
        hdf5_path = os.path.join(fmri_dir, f"{story}.hf5")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        with h5py.File(hdf5_path, "r") as hf:
            data = hf["fmri_data"][:]  # shape: (time, voxels)
            print(data.shape)
            data_masked = data[:, voxel_indices]
            resp.extend(data_masked)
        
        print(f"Loaded and masked fMRI data: {hdf5_path}")

    resp = np.array(resp)
    return resp

def compute_thresholded_mask(stories, subject, threshold=0.85):
    mask_dir = os.path.join(DATA_DIR, "masks", subject)
    mask_data = []
    mask_img_ref = None

    for story in stories:
        mask_path = os.path.join(mask_dir, f"{story}.nii")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        mask_img = nib.load(mask_path)
        if mask_img_ref is None:
            mask_img_ref = mask_img
        
        mask_data.append(mask_img.get_fdata().flatten())
        print(f"Loaded mask: {mask_path}")

    # Combine and threshold masks
    mask_stack = np.stack(mask_data, axis=0)
    mask_mean = mask_stack.mean(axis=0)
    thresholded_mask_flat = (mask_mean >= threshold).astype(np.int16)

    # Reconstruct 3D mask
    mask_shape = mask_img_ref.shape
    thresholded_mask_3d = thresholded_mask_flat.reshape(mask_shape)
    thresholded_mask = nib.Nifti1Image(thresholded_mask_3d, mask_img_ref.affine, mask_img_ref.header)

    voxel_indices = np.where(thresholded_mask_flat > 0)[0]

    return thresholded_mask, voxel_indices

def load_embeddings(folder_path):
    embeddings_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".hf5"):  # Ensure it's an HDF5 file
            story_name = os.path.splitext(file_name)[0]  # Remove .h5 extension
            file_path = os.path.join(folder_path, file_name)

            with h5py.File(file_path, "r") as h5f:
                # Assuming the dataset is stored under a key; update this if necessary
                dataset_name = list(h5f.keys())[0]  # Get the first key (modify if needed)
                embeddings = np.array(h5f[dataset_name])  # Convert to NumPy array
                embeddings_dict[story_name] = embeddings

    return embeddings_dict

def get_permuted_corrs(true, pred, blocklen):
	nblocks = int(true.shape[0] / blocklen)
	true = true[:blocklen*nblocks]
	block_index = np.random.choice(range(nblocks), nblocks)
	index = []
	for i in block_index:
		start, end = i*blocklen, (i+1)*blocklen
		index.extend(range(start, end))
	pred_perm = pred[index]
	nvox = true.shape[1]
	corrs = np.nan_to_num(mcorr(true, pred_perm))
	return corrs

def permutation_test(true, pred, blocklen, nperms):
	start_time = time.time()
	pool = ThreadPool(processes=10)
	perm_rsqs = pool.map(
		lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
	pool.close()
	end_time = time.time()
	print((end_time - start_time) / 60)
	perm_rsqs = np.array(perm_rsqs).astype(np.float32)
	real_rsqs = np.nan_to_num(mcorr(true, pred))
	pvals = (real_rsqs <= perm_rsqs).mean(0)
	return np.array(pvals), perm_rsqs, real_rsqs

def run_permutation_test(zPresp, pred, blocklen, nperms, mode='', thres=0.001):
	assert zPresp.shape == pred.shape, print(zPresp.shape, pred.shape)

	start_time = time.time()
	ntr, nvox = zPresp.shape
	partlen = nvox
	pvals, perm_rsqs, real_rsqs = [[] for _ in range(3)]

	for start in range(0, nvox, partlen):
		print(start, start+partlen)
		pv, pr, rs = permutation_test(zPresp[:, start:start+partlen], pred[:, start:start+partlen],
									  blocklen, nperms)
		pvals.append(pv)
		perm_rsqs.append(pr)
		real_rsqs.append(rs)
	pvals, perm_rsqs, real_rsqs = np.hstack(pvals), np.hstack(perm_rsqs), np.hstack(real_rsqs)

	assert pvals.shape[0] == nvox, (pvals.shape[0], nvox)
	assert perm_rsqs.shape[0] == nperms, (perm_rsqs.shape[0], nperms)
	assert perm_rsqs.shape[1] == nvox, (perm_rsqs.shape[1], nvox)
	assert real_rsqs.shape[0] == nvox, (real_rsqs.shape[0], nvox)

	cci.upload_raw_array(os.path.join(save_location, '%spvals'%mode), pvals)
	cci.upload_raw_array(os.path.join(save_location, '%sperm_rsqs'%mode), perm_rsqs)
	cci.upload_raw_array(os.path.join(save_location, '%sreal_rsqs'%mode), real_rsqs)
	print((time.time() - start_time)/60)
	
	pID, pN = fdr_correct(pvals, thres)
	cci.upload_raw_array(os.path.join(save_location, '%sgood_voxels'%mode), (pvals <= pN))
	cci.upload_raw_array(os.path.join(save_location, '%spN_thres'%mode), np.array([pN, thres], dtype=np.float32))
	return
