#!/bin/bash


################################################################################
# Experiment Parameters
################################################################################
lm_name=$"gpt2"
seed_list=$"1318" #"84 1318 735" 
ablation_list=$"TEST_no_expression_targetkg" # no_expression_targetkg -- see configs folder for more
project_name=$"test_cleaning"
top_k_layers=6
gpu_setting="single_gpu" # "single_gpu" "multi_gpu" -- see configs folder for more

kg_type=$"wordnet"
kg_list=$"statement.n.01" # "location.n.01 building.n.01 statement.n.01 change.n.01 magnitude.n.01 representation.n.02 communication.n.02"
# kg_type=$"conceptnet"
# kg_list=$"sun" #"sun swimming fruit"

NFS_MOUNT="..." # TODO: absolute path to dir where you cloned the repo
WORK_DIR="${NFS_MOUNT}/know-subnet-clean/know_subnet"

################################################################################
# Train & test example
################################################################################
for ablation_name in $ablation_list;
    do
    for kg_name in $kg_list;
        do
        for seed in $seed_list;
            do
            exper_name=$"lm=$lm_name-kg=$kg_name-seed=$seed-ablation=$ablation_name"
            exper_date=$(date +"%Y_%m_%d_%H:%M:%S")

            echo "Exper name: $exper_name"
            echo "Date: $exper_date"
            
            ####################################################################
            # Training experiment launch
            ####################################################################
            accelerate launch \
                --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
                subnet_train.py  \
                --kg_type="$kg_type" \
                --targetkg_name="$kg_name" \
                --seed="$seed" \
                --exper_name="$exper_name" \
                --project_name="$project_name" \
                --subnet_config_file="$WORK_DIR/configs/ablation_$ablation_name.json" \
                --date="$exper_date" \
                --lm="$lm_name" \
                --top_k_layers=$top_k_layers \
                --nfs_dir="$WORK_DIR"

            ####################################################################
            # Post-training evaluation
            ####################################################################
            accelerate launch \
                --config_file="$WORK_DIR/configs/accelerate_$gpu_setting.yaml" \
                subnet_test.py  \
                --exper_name="$exper_name" \
                --project_name="$project_name" \
                --date="$exper_date" \
                --nfs_dir="$WORK_DIR"
            done
        done
    done
