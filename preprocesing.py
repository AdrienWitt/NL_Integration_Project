# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:12:33 2025

@author: adywi
"""

import subprocess
import os

# Define paths
bids_dir = "/path/to/project/data"  # Replace with your BIDS data directory
output_dir = "/path/to/project/output"  # Where preprocessed data will go
work_dir = "/path/to/project/work"  # Temporary working directory

# Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(work_dir, exist_ok=True)

# Dynamically detect sessions (optional, for flexibility)
subject_dir = os.path.join(bids_dir, "sub-01")
ses_dirs = [d for d in os.listdir(subject_dir) if d.startswith("ses-")]
ses_labels = " ".join([s.replace("ses-", "") for s in ses_dirs])  # e.g., "01 02 03"

# fMRIPrep Docker command
fmriprep_cmd = [
    "docker", "run", "--rm",
    "-v", f"{bids_dir}:/data:ro",  # Mount BIDS directory as read-only
    "-v", f"{output_dir}:/out",    # Mount output directory
    "-v", f"{work_dir}:/work",     # Mount working directory
    "nipreps/fmriprep:latest",     # Official fMRIPrep Docker image
    "/data", "/out", "participant",  # Input, output, and mode
    "--participant-label", "01",   # Process subject 'sub-01'
    "--session-label", ses_labels, # Process all detected sessions (e.g., "01 02 03")
    "--fs-license-file", "/path/to/freesurfer/license.txt",  # FreeSurfer license
    "--work-dir", "/work",         # Working directory inside container
    "--nthreads", "4",             # Number of threads (adjust based on your CPU)
    "--verbose"                    # Verbose output for debugging
]

# Run fMRIPrep
try:
    print(f"Running fMRIPrep for sessions: {ses_labels}...")
    result = subprocess.run(fmriprep_cmd, check=True, text=True, capture_output=True)
    print("fMRIPrep completed successfully!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("fMRIPrep failed.")
    print(e.stderr)
    raise

# Check output for all sessions
output_base = os.path.join(output_dir, "fmriprep", "sub-01")
for ses in ses_dirs:
    ses_path = os.path.join(output_base, ses)
    if os.path.exists(ses_path):
        print(f"Output for {ses}:", os.listdir(ses_path))
    else:
        print(f"No output found for {ses}!")