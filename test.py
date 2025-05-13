from wav2vec_prosody import ProsodyDataset
import os
import json
import librosa
from transformers import Wav2Vec2Processor
from encoding.config import DATA_DIR


audio_dir = os.path.join(DATA_DIR, "stimuli_16k")
prosody_dir = os.path.join(DATA_DIR, "features/prosody/test_opensmile")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
story_names = [f.replace(".wav", "") for f in os.listdir(audio_dir)]


dataset = ProsodyDataset(audio_dir, prosody_dir, processor, story_names)

import soundfile as sf
# Number of samples to visualize
num_samples = 10000  

dataset.feature_names
# Print samples
for i in range(num_samples):
    sample = dataset[i]
    print(sample["labels"])
    
    
for file in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, file)
    audio, sr = librosa.load(audio_path, samplerate=16000)

librosa.o
    
    
#################################
import os
import opensmile
from encoding.ridge_utils.story_utils import get_story_grids
import librosa
from encoding.config import DATA_DIR

textgrid_dir = os.path.join(DATA_DIR, "ds003020/derivative/TextGrids")
stimuli_dir = os.path.join(DATA_DIR, "ds003020/stimuli")
stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]


from encoding.ridge_utils.story_utils import get_story_grids
transcripts = get_story_grids(stories, DATA_DIR)
 

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,  # Contains prosody-related features
    feature_level=opensmile.FeatureLevel.Functionals
)


features_list = []

for story in stories:
    audio_path = os.path.join(stimuli_dir, f"{story}.wav")
    y, sr = librosa.load(audio_path, sr=None)
    transcript = transcripts[story]
    for start_time, end_time, word in transcript:
        print(f"\nProcessing word: '{word}' (start: {start_time}, end: {end_time})")
        start, end = round(float(start_time) * sr), round(float(end_time) * sr)
        word_audio = y[start:end]
        features = smile.process_signal(word_audio, sr)
        features_list.append(features)
        
        
#######################
import os
import nibabel as nib


nii_file = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\sub-UTS01\ses-7\func\sub-UTS01_ses-7_task-treasureisland_bold.nii.gz"

fmri = nib.load(nii_file)

nii_file = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\sub-UTS04\ses-2\func\sub-UTS04_ses-2_task-souls_bold.nii.gz"

fmri = nib.load(nii_file)


from encoding.ridge_utils.stimulus_utils import load_simulated_trfiles

dic = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\derivative\respdict.json"

with open(dic) as f:
    spdict = json.load(f)

yo = load_simulated_trfiles(spdict)

####################################################

import nibabel as nib
import h5py
import os

story = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\sub-UTS01\ses-2\func\sub-UTS01_ses-2_task-alternateithicatom_bold.nii.gz")

mask = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\derivative\pycortex-db\UTS01\transforms\UTS01_auto\mask_thick.nii.gz")

# sub1
with h5py.File(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\derivative\preprocessed_data\UTS01\alternateithicatom.hf5", "r") as hf:
    # List all datasets in the file
    print("Keys in the HDF5 file:", list(hf.keys()))

    # Access a specific dataset
    dataset_name = list(hf.keys())[0]  # Get the first dataset name
    data = hf[dataset_name][:]  # Load the dataset as a NumPy array

    print("Dataset shape:", data.shape)
    




    
import nibabel as nib
lh_flat = nib.load('ds003020/derivative/pycortex-db/UTS01/surfaces/flat_lh.gii')
rh_flat = nib.load('ds003020/derivative/pycortex-db/UTS01/surfaces/flat_rh.gii')
lh_vertices = lh_flat.darrays[0].data  # Vertex coordinates
rh_vertices = rh_flat.darrays[0].data
print(f"Left hemisphere vertices: {lh_vertices.shape[0]}")
print(f"Right hemisphere vertices: {rh_vertices.shape[0]}")
total_vertices = lh_vertices.shape[0] + rh_vertices.shape[0]
print(f"Total cortical vertices: {total_vertices}")

import numpy as np
flatverts = np.load('ds003020/derivative/pycortex-db/UTS01/cache/flatverts_1024.npz')
for key in flatverts.keys():
    print(key, flatverts[key].shape)

mask = nib.load('ds003020/derivative/pycortex-db/UTS01/transforms/UTS01_auto/mask_thick.nii.gz')
print(f"3D cortical mask shape: {mask.shape}")
cortical_voxels = np.sum(mask.get_fdata() == 1)
print(f"Number of cortical voxels: {cortical_voxels}")


import numpy as np
flatmask = np.load('ds003020/derivative/pycortex-db/UTS01/cache/flatmask_1024.npz')
for key in flatmask.keys():
    print(f"{key}: {flatmask[key].shape}")
    if flatmask[key].size == 81126 or np.sum(flatmask[key]) == 81126:
        print(f"Possible match: {key}")

flatverts = np.load('ds003020/derivative/pycortex-db/UTS01/cache/flatverts_1024.npz')
for key in flatverts.keys():
    print(f"{key}: {flatverts[key].shape}")

from scipy.sparse import csr_matrix
matrix = csr_matrix((flatverts['data'], flatverts['indices'], flatverts['indptr']))
print("Matrix shape:", matrix.shape)
print("Non-zero elements:", matrix.nnz)


folder_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\derivative\pycortex-db\UTS01"  # Replace with your actual path

for root, dirs, files in os.walk(folder_path):
    for file in files:
        print(os.path.join(root, file)) 

import numpy as np
import nibabel as nib
import h5py
 
subjects = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"]  # Your subject list
max_voxels = 0
voxel_counts = {}

for subject in subjects:
    mask_path = f"ds003020/derivative/pycortex-db/{subject}/transforms/{subject}_auto/mask_thick.nii.gz"
    mask = nib.load(mask_path)
    cortical_voxels = np.sum(mask.get_fdata() == 1)
    voxel_counts[subject] = cortical_voxels
    max_voxels = max(max_voxels, cortical_voxels)

print("Voxel counts:", voxel_counts)
print("Maximum voxel count:", max_voxels)




    
     