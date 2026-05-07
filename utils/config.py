import matplotlib as mpl
from pathlib import Path
from datetime import datetime


currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year


#########
# Font for plots
FPATH = Path(mpl.get_data_path(), "/hpc/group/youlab/ks723/miniconda3/Lingchong/fonts/ARIAL.TTF")  # ARIAL.TTF
#########

#########################################################################

# INVERSE PROBLEM 

# folders from Fig 2, simulations used as the base to generate 3 replicate experimental-like patterns for Figure 6 
SIM_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3'  # Sim_050924_intermediate_Tp3.tar
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output'  # Sim_050924_seed.tar

# Data augmentation using rotation (created using SeedplusSim_rotation_3reps.py script), for training the sim to seed mapping model with same input data size as the exp to seed model(90k)
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3ROTREPS'
SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input_32x32_3ROTREPS/intermediate/Tp3' 

# generated images using ControlNet pipeline, synthetic images are named as exp in the folder variable cause they look like exps
# used as the base dataset to train the exp to seed model in Figure 6-- exp folder acts as output, seed as input. 
SPECIFIC_FOLDER_EXP_DIFFUSION_MORE = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202642_19_SIMTOEXP'  # synthetic generated patterns 30k*3 replicates # selected used in Fig 6 are in Generated_patterns_selected.tar
SPECIFIC_FOLDER_SEED_DIFFUSION_MORE= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3reps' # created using SeedDataset_MORE_Diffusion.py, just triplicates patterns from SEED_FOLDER_TEST_INFOENCODING_MORESEEDS for creating matched copies to train the downstream model 

# Model checkpoints
CKPT_PATH_MORESEEDS_DIFFUSION= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run7_MORE_seeds_no_augmentation/checkpoints/best-epoch=169-val_loss=0.3398.ckpt'  # synthetic experimental-like patterns to seed mapping, checkpoint_exptoseed.tar
CKPT_PATH_SEEDTOSIM_MORESEEDS_3ROTREPS= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_sim_vae_nonrandom/lambda0.5_pos2.0_GPUsv3RotReps/checkpoints/best-epoch=290-val_loss=0.0080.ckpt'  # sim to seed mapping, checkpoint_simtoseed.tar

# Exp folder test, similar as the ones used in Fig 5, input seed images first converted to 32x32, no extra cropping in Figure 6
EXP_FOLDER_TEST   = "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3/"   # Exp_testset.tar in sim_to_exp_diffusion hf datasets folder
SEED_FOLDER_EXPTOSIM_TEST_32X32= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3_32x32_coordmethod"  # Exp_testset_seed.tar

# New Exp folder test, has fixed seeding configurations, seed folder are the corresponding seeding configurations
SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input_preprocessed_v3'  # Exp_testset_fixed_seed.tar
EXP_FOLDER_EXPTOSIM_TEST_FIXED= '/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_Fixedseeds_v4_preprocessed/'  # Exp_testset_fixed.tar

# Simulation test set consisting of fixed patterns, used to test simulation-seed mappping 
SEED_FOLDER_SEEDTOSIM_TEST="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_output"  #Sim_050924_ModelTesting_seed.tar
SIM_FOLDER_SEEDTOSIM_TEST= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_input/intermediate/Tp3" #Sim_050924_ModelTesting_intermediate_Tp3.tar


# Testing metrics on the full test set 

# exp to seed mapping
SEED_FOLDER_EXPTOSIM_TEST_FIXED_NONFIXED_32X32= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input_FULLTESTSET'  # combining fixed and non-fixed test sets in the same folder 
OUTPUT_DIR_EXPTOSEED_COLOR_SAVED= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v202648_1529_EXPTOSEED_COLOR"    # saved prediction folder for exp to seed color mapping 

# sim to seed mapping
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_SAVED= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026420_1618_SIMTOSEED_MORESEEDS_3ROTREPS"  # saved prediction folder for sim to seed mapping 

# exp to seed mapping for model trained on sim to seed mapping 
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET_SAVED= "/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026420_1812_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET"  