import numpy as np
import os
import h5py
from .config import DATA_DIR
import json

def load_embeddings(folder_path):
    embeddings_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".hf5"):  # Ensure it's an HDF5 file
            story_name = os.path.splitext(file_name)[0]  # Remove .h5 extension
            file_path = os.path.join(folder_path, file_name)

            with h5py.File(file_path, "r") as h5f:
                dataset_name = list(h5f.keys())[0]  # Get the first key (modify if needed)
                embeddings = np.array(h5f[dataset_name])  # Convert to NumPy array
                embeddings_dict[story_name] = embeddings

    return embeddings_dict

def load_session_data(subject, json_path):
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Check if subject is in the participants list
    participants = data["dataset_info"]["participants"]
    if subject not in participants:
        raise ValueError(f"Subject {subject} not found in participants list: {participants}")
    
    # Get train and test stories (same for all participants in this group)
    train_stories = data["train"]["stories"]
    test_stories = data["test"]["stories"]
    
    # Combine or return separately based on your needs
    stories = {
        "train": train_stories,
        "test": test_stories,
        "all": train_stories + test_stories  
    }
    
    return stories

def get_response(stories, subject):
    fmri_dir = os.path.join(DATA_DIR, subject)
    resp = []  # for training stories (will be concatenated)
    
    for story in stories:
        hdf5_path = os.path.join(fmri_dir, f"{story}.hf5")  # .hf5 instead of .h5
        
        with h5py.File(hdf5_path, "r") as hf:
            
            if story == "wheretheressmoke":
                data = hf["individual_repeats"][:]      # shape: (10, time, voxels)
                print(f"{story} (test): loaded individual repeats {data.shape}")
                print(f"Loaded fMRI data: {hdf5_path}")
                return data                              # ← return repeats immediately
            
            else:

                data = hf["data"][:]                     # shape: (time, voxels)
                print(f"{story}: {data.shape}")
                resp.extend(data)
        
        print(f"Loaded fMRI data: {hdf5_path}")
    
    resp = np.array(resp)
    return resp



class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        
        if trfilename is not None:
            self.load_from_file(trfilename)

def load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5):
    trdict = dict()
    for story, resps in respdict.items():
        trf = TRFile(None, tr)
        trf.soundstarttime = start_time
        trf.simulate(resps - pad)
        trdict[story] = [trf]
    return trdict


