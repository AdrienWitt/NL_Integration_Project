# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:19:09 2025

@author: adywi
"""

import pandas as pd
import os
import re
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from Huth.encoding.ridge_utils.textgrid import TextGrid



os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project")

def transcript_df(dir_textgrid):
    textgrid_dfs = []
    for file in os.listdir(dir_textgrid):
        if file.endswith(".txt"):
            path = os.path.join(dir_textgrid, file)
            df = pd.read_csv(path, delimiter="\t")            
            df["TR"] = (df["tmin"] // 2) + 1 
            word_df = df[df["tier"] == "word"]
            grouped_words = word_df.groupby("TR")["text"].apply(lambda x: " ".join(x)).reset_index()
            grouped_words["story"] = file[:-4]  
            textgrid_dfs.append(grouped_words)
    return textgrid_dfs

def clean_text(text):
    text = text.lower()  # Convert the text to lowercase
    text = re.sub(r'\bsp\b', '', text)  # Remove "sp" as a word
    text = re.sub(r'\s+', ' ', text)  
    text = text.replace('?', '')  # Remove question marks
    text = text.strip()  # Remove leading and trailing spaces
    return text


def transcript_df(dir_textgrid):
    textgrid_dfs = []
    for file in os.listdir(dir_textgrid):
        path = os.path.join(dir_textgrid, file)
        tg = TextGrid.load(path)
        transcripts = []
        for tier in tg:
            transcripts.append(tier.simple_transcript)
        df = pd.DataFrame(transcripts[1], columns=["tmin", "tmax", "text"])
        df["TR"] = (df["tmin"] // 2) + 1
        grouped_words = df.groupby("TR")["text"].apply(lambda x: " ".join(x)).reset_index()
        grouped_words["story"] = file[:-4]  
        textgrid_dfs.append(grouped_words)
    return textgrid_dfs

def clean_text(text):
    text = text.lower()  # Convert the text to lowercase
    text = re.sub(r'\bsp\b', '', text)  # Remove "sp" as a word
    text = re.sub(r'\s+', ' ', text)  
    text = text.replace('?', '')  # Remove question marks
    text = text.strip()  # Remove leading and trailing spaces
    return text


def get_embeddings_text(text, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    last_layer_embeddings = outputs.last_hidden_state.cpu().numpy()
    summed_embeddings = last_layer_embeddings.sum(axis=1)
    return summed_embeddings


dir_textgrid = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\Text"
output_dir = r'C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\embeddings\text'

df_list = transcript_df(dir_textgrid)

for df in df_list:
    for _, row in df.iterrows():  
        text = row['text']
        story = row['story']
        tr = row['TR']        
        cleaned_text = clean_text(text)        
        if cleaned_text.strip() == "":
            embeddings = np.zeros(768)
            print("replace with zeros")
        else:
            embeddings = get_embeddings_text(cleaned_text)           
        np.save(os.path.join(output_dir, f"{story}_{int(tr)}_lastlayer.npy"), embeddings)
    


import h5py
import numpy as np
import nibabel as nib
from nilearn import plotting


file_path = r"C:\Users\adywi\Downloads\adollshouse.hf5"

with h5py.File(file_path, 'r') as file:
    dataset = file['data']
    data = dataset[:]
    
file = r"C:\Users\adywi\Downloads\sub-UTS01_ses-11_task-againstthewind_bold.nii.gz"
fmri = nib.load(file)
    