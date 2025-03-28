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


nii_file = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\sub-UTS01\ses-2\func\sub-UTS01_ses-2_task-alternateithicatom_bold.nii.gz"
        
        
fmri = nib.load(nii_file)

nii_file = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\sub-UTS04\ses-2\func\sub-UTS04_ses-2_task-souls_bold.nii.gz"
    
fmri = nib.load(nii_file)


    
     