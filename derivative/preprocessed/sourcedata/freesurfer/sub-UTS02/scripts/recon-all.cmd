

#---------------------------------
# New invocation of recon-all Wed Apr  2 09:32:48 UTC 2025 

 mri_convert /work/fmriprep_25_0_wf/sub_UTS02_wf/anat_fit_wf/anat_validate/sub-UTS02_ses-1_T1w_noise_corrected_ras_valid.nii.gz /out/sourcedata/freesurfer/sub-UTS02/mri/orig/001.mgz 

#--------------------------------------------
#@# MotionCor Wed Apr  2 09:32:56 UTC 2025

 cp /out/sourcedata/freesurfer/sub-UTS02/mri/orig/001.mgz /out/sourcedata/freesurfer/sub-UTS02/mri/rawavg.mgz 


 mri_info /out/sourcedata/freesurfer/sub-UTS02/mri/rawavg.mgz 


 mri_convert /out/sourcedata/freesurfer/sub-UTS02/mri/rawavg.mgz /out/sourcedata/freesurfer/sub-UTS02/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /out/sourcedata/freesurfer/sub-UTS02/mri/transforms/talairach.xfm /out/sourcedata/freesurfer/sub-UTS02/mri/orig.mgz /out/sourcedata/freesurfer/sub-UTS02/mri/orig.mgz 


 mri_info /out/sourcedata/freesurfer/sub-UTS02/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Wed Apr  2 09:33:10 UTC 2025

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --ants-n4 --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

lta_convert --src orig.mgz --trg /opt/freesurfer/average/mni305.cor.mgz --inxfm transforms/talairach.xfm --outlta transforms/talairach.xfm.lta --subject fsaverage --ltavox2vox
#--------------------------------------------
#@# Talairach Failure Detection Wed Apr  2 09:36:00 UTC 2025

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /opt/freesurfer/bin/extract_talairach_avi_QA.awk /out/sourcedata/freesurfer/sub-UTS02/mri/transforms/talairach_avi.log 


 tal_QC_AZS /out/sourcedata/freesurfer/sub-UTS02/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Wed Apr  2 09:36:00 UTC 2025

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --n 2 --ants-n4 


 mri_add_xform_to_header -c /out/sourcedata/freesurfer/sub-UTS02/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Wed Apr  2 09:38:39 UTC 2025

 mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz 

