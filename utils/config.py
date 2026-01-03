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

# For creating prompt.json files 
BASE_FOLDER= '/hpc/group/youlab/ks723/storage/'
SPECIFIC_FOLDER_SIM='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_100AUG' # Extract SimcorrtoExp.tar, run the augmentation script and use those images here
SPECIFIC_FOLDER_EXP='/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_100AUG' # Extract Exp.tar, run the augmentation script and use those images here
SPECIFIC_FOLDER_SEED='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG' # Extract Exp_SimcorrtoExp_seed.tar, run Seed_DataAugmentation.py script and use those images here

# For Sim to Exp dataset, model training
EXP_FOLDER_TRAIN_NONAUG='/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed/'  # Exp.tar

# For Seed to Exp dataset, model training

SEED_FOLDER_TRAIN_NONAUG="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input/"  # Exp_SimcorrtoExp_seed.tar

# For running model inference # test images folder, v3 is the folder with 96 images- same number of images in experiment and simulation folders, used in the final analysis
SIM_FOLDER_TEST   = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_v3/"  # SimcorrtoExp_testset.tar
SEED_FOLDER_TEST  = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3/"  # Exp_SimcorrtoExp_testset_seed.tar
EXP_FOLDER_TEST   = "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3/"   # Exp_testset.tar 


# Creation of different folders that will be later used for inference outputs
OUTPUT_DIR_SEEDTOEXP = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SEEDTOEXP"
OUTPUT_DIR_SIMTOEXP = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SIMTOEXP"

OUTPUT_DIR_RANDOMSEEDSWEEP = f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_random_seed_sweep/"
OUTPUT_DIR_ABLATION_BASE= f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}"
OUTPUT_DIR_CONTRASTIVE_LEARNING=f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/contrastive_learning_v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}/"

# For ablation study 

MAIN_FOLDER= '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v2025926_1251_simtoexp_v3/'   # default ControlNet, OUTPUT_DIR_SIMTOEXP
ABLATION_1=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1458_no_guess/'  # Guess_mode= TRUE , Run ablation script and modify the part accordingly
ABLATION_2=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1753_no_negative/' # n_prompt=""
ABLATION_3=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1756_plus_positive/' # a_prompt="best quality, extremely detailed" # default ControlNet
ABLATION_4=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1758_low_strength_point85/' # strength=0.85
ABLATION_5=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1758_high_strength_1point25/' # strength=1.25
ABLATION_6=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_1759_higher_DDIM_steps_100/'  # ddim_steps=100
ABLATION_7=  '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251023_181_lower_guidance_9point0/'  # guidance_scale=9.0 # default ControlNet

# For plotting Fig 5 (SUPP FIG 15)
PRED_FOLDER_SEEDTOEXP = '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251011_841_seedtoexp_swapped_v3/'  # OUTPUT_DIR_SEEDTOEXP

# Checkpoint path for inference 

CKPT_PATH='/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_37312560/checkpoints/epoch=4-step=51124.ckpt'   # checkpoint_simtoexp.tar
CKPT_PATH_SEEDTOEXP="/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_37726282/checkpoints/epoch=4-step=51124.ckpt" # checkpoint_seedtoexp.tar

#########################################################################

# INVERSE PROBLEM 

# for Seed to Sim dataset creation

SEED_FOLDER_SEEDTOSIM= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output"
SIM_FOLDER_SEEDTOSIM= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3"

# saved model checkpoint 

CKPT_PATH_SEEDTOSIM= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/version_41225691/checkpoints/epoch=307-step=129975.ckpt'

# output folder

OUTPUT_DIR_SEEDTOSIM = f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SEEDTOSIM"

# test set 

SEED_FOLDER_SEEDTOSIM_TEST="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_output"
SIM_FOLDER_SEEDTOSIM_TEST= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_input/intermediate/Tp3"

# output folder containing saved inference images

OUTPUT_DIR_SEEDTOSIM_SAVED= '/hpc/dctrl/ks723/Information_encoding_decoding/inference/v202612_1641_SEEDTOSIM'