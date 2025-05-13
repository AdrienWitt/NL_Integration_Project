import os
import nibabel as nib
from nipype.interfaces import fsl, spm
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nilearn import plotting

# Set environment variables
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'
if 'FSLDIR' not in os.environ:
    raise EnvironmentError("FSLDIR not set. Please install FSL and set the environment variable.")

def convert_to_windows_path(wsl_path):
    """Convert a WSL path (/mnt/c/...) to a Windows path (C:\...) for SPM."""
    if wsl_path.startswith('/mnt/c/'):
        return 'C:\\' + wsl_path[6:].replace('/', '\\')
    return wsl_path

def fmri_preprocessing_pipeline(func_file, anat_file, output_dir):
    # Keep WSL paths for Python/FSL, convert for SPM if needed
    wsl_func_file = func_file
    wsl_anat_file = anat_file
    wsl_output_dir = output_dir
    
    # Verify files exist in WSL
    if not os.path.exists(wsl_func_file):
        raise FileNotFoundError(f"Functional file not found: {wsl_func_file}")
    if not os.path.exists(wsl_anat_file):
        raise FileNotFoundError(f"Anatomical file not found: {wsl_anat_file}")
    print(f"WSL check - Func: {wsl_func_file}, Anat: {wsl_anat_file}")
    
    # Convert paths for SPM (assuming SPM might need Windows paths)
    spm_func_file = convert_to_windows_path(wsl_func_file)
    spm_anat_file = convert_to_windows_path(wsl_anat_file)
    spm_output_dir = convert_to_windows_path(wsl_output_dir)
    
    os.makedirs(wsl_output_dir, exist_ok=True)  # Use WSL path for directory creation
    
    preprocess = pe.Workflow(name='fmri_preprocess')
    preprocess.base_dir = spm_output_dir  # SPM might need Windows path here
    
    inputnode = pe.Node(niu.IdentityInterface(fields=['func', 'anat']), name='inputnode')
    inputnode.inputs.func = spm_func_file  # Windows path for SPM
    inputnode.inputs.anat = spm_anat_file  # Windows path for SPM
    
    slicetiming = pe.Node(fsl.SliceTimer(), name='slicetiming')
    slicetiming.inputs.time_repetition = 2.0
    
    realign = pe.Node(spm.Realign(), name='realign')
    realign.inputs.register_to_mean = True
    
    coreg = pe.Node(spm.Coregister(), name='coregister')
    coreg.inputs.jobtype = 'estwrite'
    
    normalize = pe.Node(spm.Normalize(), name='normalize')
    template_path = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"MNI template not found at {template_path}")
    normalize.inputs.template = template_path  # FSL in WSL, keep Linux path
    
    outputnode = pe.Node(niu.IdentityInterface(fields=['preprocessed_func']), name='outputnode')
    
    preprocess.connect([
        (inputnode, slicetiming, [('func', 'in_file')]),
        (slicetiming, realign, [('slice_time_corrected_file', 'in_files')]),
        (inputnode, coreg, [('anat', 'target')]),
        (realign, coreg, [('mean_image', 'source')]),
        (realign, coreg, [('realigned_files', 'apply_to_files')]),
        (coreg, normalize, [('coregistered_files', 'source')]),
        (normalize, outputnode, [('normalized_files', 'preprocessed_func')])
    ])
    
    preprocess.run()
    
    func_base = os.path.basename(wsl_func_file).replace('.nii.gz', '').replace('.nii', '')
    output_file = os.path.join(wsl_output_dir, 'fmri_preprocess', 'normalize', f"w{func_base}.nii")
    return output_file

def preprocess_all_subjects(base_dir):
    dataset_dir = os.path.join(base_dir, "ds003020")
    subject_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('sub-')]
    
    for sub_dir in subject_dirs:
        subject_id = sub_dir.replace('sub-', '')
        
        anat_file = os.path.join(dataset_dir, sub_dir, "ses-1", "anat", 
                                f"sub-{subject_id}_ses-1_T1w.nii.gz")
        if not os.path.exists(anat_file):
            print(f"Error: Anatomical file not found for {sub_dir} at {anat_file}")
            continue
        print(f"Found anatomical file: {anat_file}")
        
        session_dirs = [d for d in os.listdir(os.path.join(dataset_dir, sub_dir)) 
                       if d.startswith('ses-')]
        
        for ses_dir in session_dirs:
            session_id = ses_dir.replace('ses-', '')
            func_dir = os.path.join(dataset_dir, sub_dir, ses_dir, "func")
            if not os.path.exists(func_dir):
                print(f"Warning: Functional directory not found for {sub_dir}/{ses_dir}")
                continue
            
            func_files = os.listdir(func_dir)
            for func_file in func_files:
                func = os.path.join(func_dir, func_file)
                if not os.path.exists(func):
                    func += ".gz"
                if not os.path.exists(func):
                    print(f"Warning: Functional file not found for {sub_dir}/{ses_dir}: {func}")
                    continue
                
                out_dir = os.path.join(dataset_dir, sub_dir, ses_dir, "derivative", "preprocessed")
                
                print(f"Processing: sub-{subject_id}, ses-{session_id}")
                try:
                    preprocessed_file = fmri_preprocessing_pipeline(func, anat_file, out_dir)
                    print(f"Completed: {preprocessed_file}")
                except Exception as e:
                    print(f"Error processing sub-{subject_id}, ses-{session_id}: {str(e)}")

if __name__ == '__main__':
    base_dir = "/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/NL_Project"
    preprocess_all_subjects(base_dir)