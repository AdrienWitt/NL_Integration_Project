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


def preprocess_features(train_stories, test_stories, text_feat, audio_feat, modality, trim, ndelays, use_pca=False, explained_variance=0.70):
    """Preprocess features: trim, z-score, PCA, and HRF for train and test stories.

    Args:
        train_stories (list): List of training story IDs.
        test_stories (list): List of test story IDs.
        text_feat (dict): Dictionary mapping story IDs to text feature arrays.
        audio_feat (dict): Dictionary mapping story IDs to audio feature arrays.
        modality (str): One of 'text', 'audio', or 'text_audio'.
        trim (int): Number of samples to trim from start/end.
        ndelays (int): Number of delays for HRF.
        use_pca (bool): If True, apply PCA to reduce dimensionality.
        explained_variance (float): Target explained variance for PCA (e.g., 0.70).

    Returns:
        tuple: (delRstim, delPstim) - Processed feature matrices for train and test.
    """
    # Validate inputs
    all_stories = train_stories + test_stories
    for story in all_stories:
        if story not in text_feat or story not in audio_feat:
            print(f"Story {story} not found in feature dictionaries, removing.")
            train_stories = [s for s in train_stories if s in text_feat and s in audio_feat]
            test_stories = [s for s in test_stories if s in text_feat and s in audio_feat]
            all_stories = train_stories + test_stories

    if not train_stories or not test_stories:
        raise ValueError("No valid train or test stories after filtering.")

    # Initialize delays
    delays = range(1, ndelays+1)

    # Process text features
    text_stim = [text_feat[s][5+trim:-trim] for s in all_stories]
    text_concat = np.vstack(text_stim)  # Shape: (total_samples, n_features)
    text_concat_z = zscore(text_concat)  # Z-score concatenated features

    if use_pca:
        print(f"Applying PCA to text features with explained variance threshold: {explained_variance}")
        pca_text = PCA(n_components=explained_variance)
        pca_text.fit(text_concat_z)
        n_components_text = pca_text.n_components_
        print(f"Text features: {n_components_text} components selected for {explained_variance:.2f} explained variance")
        text_concat_z = pca_text.transform(text_concat_z)  # Shape: (total_samples, n_components)

    # Reorganize text features into dictionaries
    text_train_stim = {}
    text_test_stim = {}
    start_idx = 0
    for s in all_stories:
        n_samples = text_stim[all_stories.index(s)].shape[0]
        features = text_concat_z[start_idx:start_idx+n_samples]
        if s in train_stories:
            text_train_stim[s] = features
        if s in test_stories:
            text_test_stim[s] = features
        start_idx += n_samples

    # Process audio features
    audio_stim = [audio_feat[s][5+trim:-trim] for s in all_stories]
    audio_concat = np.vstack(audio_stim)
    audio_concat_z = zscore(audio_concat)

    if use_pca:
        print(f"Applying PCA to audio features with explained variance threshold: {explained_variance}")
        pca_audio = PCA(n_components=explained_variance)
        pca_audio.fit(audio_concat_z)
        n_components_audio = pca_audio.n_components_
        print(f"Audio features: {n_components_audio} components selected for {explained_variance:.2f} explained variance")
        audio_concat_z = pca_audio.transform(audio_concat_z)

    # Reorganize audio features into dictionaries
    audio_train_stim = {}
    audio_test_stim = {}
    start_idx = 0
    for s in all_stories:
        n_samples = audio_stim[all_stories.index(s)].shape[0]
        features = audio_concat_z[start_idx:start_idx+n_samples]
        if s in train_stories:
            audio_train_stim[s] = features
        if s in test_stories:
            audio_test_stim[s] = features
        start_idx += n_samples

    # Apply HRF for train and test stories
    if modality == "text":
        # Train
        stim = [text_train_stim[s] for s in train_stories]
        stim = np.vstack(stim)
        delRstim = make_delayed(stim, delays)
        # Test
        stim = [text_test_stim[s] for s in test_stories]
        stim = np.vstack(stim)
        delPstim = make_delayed(stim, delays)
    elif modality == "audio":
        # Train
        stim = [audio_train_stim[s] for s in train_stories]
        stim = np.vstack(stim)
        delRstim = make_delayed(stim, delays)
        # Test
        stim = [audio_test_stim[s] for s in test_stories]
        stim = np.vstack(stim)
        delPstim = make_delayed(stim, delays)
    else:  # text_audio
        # Train
        stim_text = [text_train_stim[s] for s in train_stories]
        stim_text = np.vstack(stim_text)
        stim_audio = [audio_train_stim[s] for s in train_stories]
        stim_audio = np.vstack(stim_audio)
        delRstim_text = make_delayed(stim_text, delays)
        delRstim_audio = make_delayed(stim_audio, delays)
        delRstim = np.concatenate([delRstim_text, delRstim_audio], axis=1)
        # Test
        stim_text = [text_test_stim[s] for s in test_stories]
        stim_text = np.vstack(stim_text)
        stim_audio = [audio_test_stim[s] for s in test_stories]
        stim_audio = np.vstack(stim_audio)
        delPstim_text = make_delayed(stim_text, delays)
        delPstim_audio = make_delayed(stim_audio, delays)
        delPstim = np.concatenate([delPstim_text, delPstim_audio], axis=1)

    return delRstim, delPstim


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
	main_path = pathlib.Path(__file__).parent.parent.resolve()
	subject_dir = join("H:/derivative/data/preprocessed_data/%s" % subject)
	base = os.path.join(main_path, subject_dir)
	resp = []
	for story in stories:
		resp_path = os.path.join(base, "%s.hf5" % story)
		hf = h5py.File(resp_path, "r")
		resp.extend(hf["data"][:])
		hf.close()
	return np.array(resp)

def get_response_mask(stories, subject, threshold=0.85, mask=True, data_dir="H:/derivative/data"):
    """Get the subject's fMRI response and combined thresholded mask for stories.

    Parameters:
    - stories (list): List of story names (e.g., ["story_1", "story_2"]).
    - subject (str): Subject ID (e.g., "sub-UTS02").
    - threshold (float): Threshold for combining masks (default: 0.85).
    - data_dir (str): Root directory containing preprocessed_data and masks (default: "H:/derivative/data").
    - mask (bool): If True, mask the fMRI data; if False, return unmasked data (default: True).

    Returns:
    - resp (np.ndarray): Masked fMRI responses (time_points × masked_voxels) if mask=True,
                        or unmasked responses (time_points × flattened_voxels) if mask=False.
    - thresholded_mask (nib.Nifti1Image): Combined, thresholded 3D mask.
    """
    # Define directories
    fmri_dir = os.path.join(data_dir, "preprocessed_data", subject)
    mask_dir = os.path.join(data_dir, "masks", subject)

    resp = []
    masks = []
    
    # Load fMRI data and masks for each story
    for story in stories:
        # Load HDF5 file
        hdf5_path = os.path.join(fmri_dir, f"{story}.hf5")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as hf:
            resp.append(hf["data"][:])
        print(f"Loaded fMRI data: {hdf5_path}")
        
        # Load mask
        mask_path = os.path.join(mask_dir, f"{story}.nii")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = nib.load(mask_path)
        masks.append(mask)
        print(f"Loaded mask: {mask_path}")
    
    # Concatenate fMRI responses
    resp = np.concatenate(resp, axis=0)  # Shape: (total_time_points, flattened_voxels)
    
    # Combine masks
    if not masks:
        raise ValueError("No valid masks loaded.")
    
    # Flatten masks and sum
    mask_shape = masks[0].shape
    num_voxels = np.prod(mask_shape)
    if resp.shape[1] != num_voxels:
        raise ValueError(f"fMRI data voxel count ({resp.shape[1]}) does not match mask voxels ({num_voxels}).")
    
    mask_sum = np.zeros(num_voxels, dtype=np.float32)
    for mask in masks:
        mask_sum += mask.get_fdata().flatten()
    
    # Apply threshold
    num_masks = len(masks)
    thresholded_mask_flat = (mask_sum >= threshold * num_masks).astype(np.int16)
    
    # Create 3D thresholded mask for output
    thresholded_mask_3d = thresholded_mask_flat.reshape(mask_shape)
    thresholded_mask = nib.Nifti1Image(thresholded_mask_3d, masks[0].affine, masks[0].header)
    
    # Apply mask if requested
    if mask:
        masked_voxel_indices = np.where(thresholded_mask_flat > 0)[0]
        resp = resp[:, masked_voxel_indices]  # Shape: (time_points, masked_voxels)
    
    return resp, thresholded_mask

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
