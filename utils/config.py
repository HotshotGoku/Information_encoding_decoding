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
SPECIFIC_FOLDER_SEED_OLD='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG' # Extract Exp_SimcorrtoExp_seed.tar, run Seed_DataAugmentation.py script and use those images here
SPECIFIC_FOLDER_SEED= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG_32x32'  # preprocessed with left_crop=2, right_crop=3

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


# For ablation study 

MAIN_FOLDER= '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v2025926_1251_simtoexp_v3/'   # default ControlNet, OUTPUT_DIR_SIMTOEXP


# Checkpoint path for inference 

CKPT_PATH='/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_37312560/checkpoints/epoch=4-step=51124.ckpt'   # checkpoint_simtoexp.tar
CKPT_PATH_SEEDTOEXP="/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/lightning_logs/version_37726282/checkpoints/epoch=4-step=51124.ckpt" # checkpoint_seedtoexp.tar

#########################################################################

# INVERSE PROBLEM 

# for Seed to Sim dataset creation, original folders without augmentations

SEED_FOLDER_SEEDTOSIM= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output"
SEED_FOLDER_SEEDTOSIM_TEST_2= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_output_dup_2'  # replacing Output prefix with Input 

# test set 

SEED_FOLDER_SEEDTOSIM_TEST="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_output_dup"
SIM_FOLDER_SEEDTOSIM_TEST= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924_ModelTesting/Sim_input/intermediate/Tp3"


SEED_FOLDER_EXPTOSIM_TEST="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3"
EXP_FOLDER_EXPTOSIM_TEST= "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3"
SEED_FOLDER_EXPTOSIM_TEST_256x256= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3_256x256/"  # for seed to exp color model


# for seed to exp datast, color version
# CKPT_PATH_EXPTOSEED_COLOR= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color/checkpoints/best-epoch=117-val_loss=0.0033.ckpt'
# CKPT_PATH_EXPTOSEED_COLOR= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_nonrandom_v2/checkpoints/last.ckpt'
CKPT_PATH_EXPTOSEED_COLOR= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_nonrandom_v3/checkpoints/last.ckpt'

OUTPUT_DIR_EXPTOSEED_COLOR=f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_EXPTOSEED_COLOR"
OUTPUT_DIR_EXPTOSEED_COLOR_SAVED= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v202648_1529_EXPTOSEED_COLOR"    # saved prediction folder for exp to seed color mapping 


CKPT_PATH_EXPTOSEED_COLOR_VAE_ResNet= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/ResNet_exps_color_vae/old/checkpoints/last.ckpt'
SEED_FOLDER_EXPTOSIM_TEST_32x32= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3_32x32/"  # for seed to exp color model
SEED_FOLDER_EXPTOSIM_TEST_32X32= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input_v3_32x32_coordmethod"


# CKPT_PATH_EXPTOSEED_COLOR_VAE_Res_UNet= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae/lambda0.2_pos5.0/checkpoints/best-epoch=033-val_loss=0.1214.ckpt'
CKPT_PATH_EXPTOSEED_COLOR_VAE_Res_UNet= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae/lambda0.2_pos10.0/checkpoints/best-epoch=027-val_loss=0.1330.ckpt'
CKPT_PATH_EXPTOSEED_COLOR_VAE_RES_UNET_BNW = '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_BNW_vae_nonrandom_v2/checkpoints/best-epoch=449-val_loss=0.2338.ckpt'
CKPT_PATH_EXPTOSEED_COLOR_VAE_RES_UNET_BNW_2= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_BNW_vae_nonrandom_v3/checkpoints/last.ckpt'
# CKPT_PATH_EXPTOSEED_COLOR_VAE_RES_UNET= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_v2/lambda0.2_pos2.0/checkpoints/last.ckpt' # best-epoch=021-val_loss=0.2275
CKPT_PATH_EXPTOSEED_COLOR_VAE_RES_UNET= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_v2/lambda0.2_pos5.0/checkpoints/last.ckpt' # best-epoch=021-val_loss=0.2275


# using the photorealistic patterns, we need to generate a new augmented dataset for simulations 

SPECIFIC_FOLDER_SEED_DIFFUSION= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_input_100AUG_diffusion_32x32"    
SPECIFIC_FOLDER_EXP_DIFFUSION= "/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v2026325_13_SIMTOEXP"


SPECIFIC_FOLDER_EXP_DIFFUSION_MORE = '/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202642_19_SIMTOEXP'  # synthetic generated patterns 30k*3 replicates 
SPECIFIC_FOLDER_SEED_DIFFUSION_MORE= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3reps' # same as SPECIFIC_FOLDER_SEED_DIFFUSION_MORESEEDS_EXTRAREPLICATES

CKPT_PATH_EXPTOSEED_COLOR_VAE_RES_UNET_DIFFUSION= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run3/checkpoints/last.ckpt' 
CKPT_PATH_EXPTOSEED_COLOR_VAE_RES_UNET_DIFFUSION_CONTROL= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run4control/checkpoints/last.ckpt'



CKPT_PATH_MORESEEDS_DIFFUSION= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run7_MORE_seeds_no_augmentation/checkpoints/best-epoch=169-val_loss=0.3398.ckpt'



SIM_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3'  # same as the file in Sim_050924_intermediate_Tp3.tar
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output'  # same as the file in seed to sim deterministic, Sim_050924_seed.tar
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_EXTRAREPLICATES= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3reps'  # dont know about this for now

# sim to seed model mapping, 30000 x 3 augmentations
SEED_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3ROTREPS'
SIM_FOLDER_TEST_INFOENCODING_MORESEEDS_3ROTREPS= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input_32x32_3ROTREPS/intermediate/Tp3' 

CKPT_PATH_SEEDTOSIM_MORESEEDS_3ROTREPS= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_sim_vae_nonrandom/lambda0.5_pos2.0_GPUsv3RotReps/checkpoints/best-epoch=290-val_loss=0.0080.ckpt'  # checkpoint used in the paper. 
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SIMTOSEED_MORESEEDS_3ROTREPS"
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_SAVED= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026420_1618_SIMTOSEED_MORESEEDS_3ROTREPS"  # saved prediction folder for sim to seed mapping 

OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET= f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET"
OUTPUT_DIR_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET_SAVED= "/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026420_1812_SIMTOSEED_MORESEEDS_3ROTREPS_EXPDATASET"  # saved prediction folder for exp to seed mapping



# new test set 

SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input_preprocessed_v3'
EXP_FOLDER_EXPTOSIM_TEST_FIXED= '/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_Fixedseeds_v4_preprocessed/'
EXP_FOLDER_EXPTOSIM_TEST_FIXED_RAW= '/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_Fixedseeds'

SEED_FOLDER_EXPTOSIM_TEST_FIXED_NONFIXED_32X32= '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_041326/Sim_input_FULLTESTSET'  # combining fixed and non-fixed test sets in the same folder 


#####################
# extra stuff not used in paper, keeping here for future reference if needed 
#####################

SIM_FOLDER_TEST_INFOENCODING= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL"  # folder without the 100 augmentations

# for Seed to Sim dataset creation, original folders without augmentations

SEED_FOLDER_SEEDTOSIM= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output"
SIM_FOLDER_SEEDTOSIM= "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_input/intermediate/Tp3"

SPECIFIC_FOLDER_EXP_DIFFUSION_ROTATED= "/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v2026325_13_SIMTOEXP_AUG100rotation"

# CKPT_PATH_SEEDTOSIM= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/version_41225691/checkpoints/epoch=307-step=129975.ckpt'
# CKPT_PATH_SEEDTOSIM= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/version_41363178/checkpoints/epoch=499-step=117499.ckpt'
CKPT_PATH_SEEDTOSIM= '/hpc/dctrl/ks723/Information_encoding_decoding/Res_UNet_Res_UNet/0_1/checkpoints/last.ckpt'
CKPT_PATH_SEEDTOSIM_VAE= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_sim_vae/checkpoints/best-epoch=145-val_loss=0.0061.ckpt'

OUTPUT_DIR_SEEDTOSIM = f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SEEDTOSIM"
# output folder

OUTPUT_DIR_SEEDTOSIM = f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SEEDTOSIM"


# output folder containing saved inference images

OUTPUT_DIR_SEEDTOSIM_SAVED= '/hpc/dctrl/ks723/Information_encoding_decoding/inference/v202612_1641_SEEDTOSIM'
OUTPUT_DIR_SEEDTOSIM_SAVED_V2= '/hpc/dctrl/ks723/Information_encoding_decoding/inference/v202616_1450_SEEDTOSIM'


OUTPUT_DIR_SEEDTOSIM_SAVED_VAE= '/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026212_1734_SEEDTOSIM' #'/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026210_1314_SEEDTOSIM'
OUTPUT_DIR_SEEDTOSIM_SAVED_VAE_V2= '/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026212_1725_SEEDTOSIM'

OUTPUT_DIR_EXPTOSEED_SAVED_VAE= '/hpc/dctrl/ks723/Information_encoding_decoding/inference/v2026212_1939_SEEDTOEXP'


# k indicates number of experimental rotational augmentations  
CKPT_PATH_CHOOSEAUGMENTATION_K1 = '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_k1_choseaugmentation/checkpoints/best-epoch=065-val_loss=0.4891.ckpt'
CKPT_PATH_CHOOSEAUGMENTATION_K20 = '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_k20_choseaugmentation/checkpoints/best-epoch=057-val_loss=0.4818.ckpt'
CKPT_PATH_CHOOSEAUGMENTATION_K40= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_k40_choseaugmentation/checkpoints/best-epoch=063-val_loss=0.4748.ckpt'
CKPT_PATH_CHOOSEAUGMENTATION_K80 = '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_k80_choseaugmentation/checkpoints/best-epoch=058-val_loss=0.4679.ckpt'

CKPT_PATH_CHOOSEAUGMENTATION_K100= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run4control/checkpoints/best-epoch=034-val_loss=0.4538.ckpt'
CKPT_PATH_CHOOSEAUGMENTATION_K0='/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_color_vae_nonrandom_vDiffusion/lambda0.5_pos2.0_GPUs_run6_rotated_diffusion/checkpoints/best-epoch=072-val_loss=0.4893.ckpt'

# for seed to exp dataset

# CKPT_PATH_SEEDTOEXP= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_run2/checkpoints/best-epoch=098-val_loss=0.0046.ckpt'
# CKPT_PATH_SEEDTOEXP= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_nonrandom/checkpoints/best-epoch=001-val_loss=0.0689.ckpt'
CKPT_PATH_SEEDTOEXP= '/hpc/dctrl/ks723/Information_encoding_decoding/lightning_logs/Res_UNet_exps_nonrandom/checkpoints/last.ckpt'
OUTPUT_DIR_SEEDTOEXP=f"/hpc/dctrl/ks723/Information_encoding_decoding/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_SEEDTOEXP"