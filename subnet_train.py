import os
import time
import wandb
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
# os.environ["WANDB_MODE"] = "offline"

# LM IMPORTS
from know_subnet.lm.lm_utils import load_lm

# DATASET IMPORTS
# from know_subnet.data.wordnet_dataloader import (
#     load_wordnet_targetkg,
#     load_wordnet_controlkg
# )
# from know_subnet.data.conceptnet_dataloader import load_conceptnet
# from know_subnet.data.wikitext_dataloader import load_wikitext2_dataloader
from know_subnet.data.ours_dataloader import load_ours

# EXPERIMENT CONFIG IMPORTS
from know_subnet.args import get_args
from know_subnet.utils import (
    write_lines,
    set_seed_and_device,
    build_main_log_paths,
    create_experiment_folder,
)

# EXPERIMENT RUNNER IMPORTS
from know_subnet.subnet_train_utils import (
    train_mask,
    zeroshot_log_loop
)

import warnings
warnings.simplefilter("always")
warnings.filterwarnings("error")

def data_loading(args, accelerator=None):
    if args.verbose:
        accelerator.print("Loading data...")

    with accelerator.main_process_first():
        targetkg_train_loader = None
        targetkg_val_loader = None
        controlkg_train_loader = None
        controlkg_val_loader = None
        controllm_train_loader = None
        controllm_val_loader = None

        ########################################################################
        # 1) Ours loading
        ########################################################################
        accelerator.print("-" * 50)
        accelerator.print("Loading Our dataset...")
        our_train_loader, our_val_loader = load_ours(
            lm=args.lm,
            reasoning_train=args.reasoning_train,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size
        )
        accelerator.print("Loading Our dataset done.")

        ########################################################################
        # 4) Log datasets
        ########################################################################
        accelerator.print("-" * 50)
        accelerator.print("Logging dataset for quality check...")
        for dataset_name, dataloader in [
            ("our-train", our_train_loader),
            ("our-val", our_val_loader),
        ]:
            dataset = dataloader.dataset
            data = {
                'inputs_str': dataset.inputs_str,
                'labels_str': dataset.labels_str
            }
            
            df = pd.DataFrame.from_dict(data)
            dataset_table = wandb.Table(dataframe=df)
            if accelerator is not None:
                accelerator.log({dataset_name: dataset_table})
            else:
                dataset_table_artifact = wandb.Artifact(dataset_name, type="dataset")
                dataset_table_artifact.add(dataset_table, dataset_name + "-table")
                wandb.log({dataset_name: dataset_table})
                wandb.log_artifact(dataset_table_artifact)
            
    if args.verbose:
        if accelerator is None:
            print("Dataset loading and saving done.")
        else:
            accelerator.print("Dataset loading and saving done.")
    
    return (
        our_train_loader,
        our_val_loader,
    )

def main():
    # 1) Loading
    args = get_args()
    args = set_seed_and_device(args)
    args = build_main_log_paths(args)
    # accelerator = None
    accelerator = Accelerator( log_with="wandb")
    if accelerator.is_main_process:
        wandb.login(key="bf5686948224a019c77ae421247f858cd53ddcbe")
    
    # 2) Running experiments
    # [0] setting up logging + saving config
    time_start = time.time()
    args = create_experiment_folder(args)

    wandb_dir = None
    if args.nfs_dir != '':
        wandb_dir = os.path.join(args.nfs_dir, 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
    if accelerator is not None: 
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={"wandb": {
                "reinit": True, 
                "dir": wandb_dir, 
                "name": args.exper_name + "-date=" + args.date
            }}
        )
    else:
        run = wandb.init(reinit=True, project=args.project_name, dir=wandb_dir)
        wandb.run.name = args.exper_name + "-date=" + args.date
        wandb.config.update(args)

    print(f'args.train_batch_size = {args.train_batch_size}')
    print(f'args.eval_batch_size = {args.eval_batch_size}')
    # [1] load lms and save if randomly masked
    our_train_loader, our_val_loader = data_loading(args, accelerator)
    
    # [2] loading lang model
    model = load_lm(args)

    # [3] train/test if needed
    if not args.test_full_model:
        csv_name = f'metrics.csv'
        
        last_log_dict, model = train_mask(
            args=args, 
            model=model, 
            our_train_loader=our_train_loader, 
            our_val_loader=our_val_loader,
            accelerator=accelerator,
            csv_name=csv_name
        )
    else:
        last_log_dict = zeroshot_log_loop(
            model=model, 
            lm_name=args.lm,
            targetkg_val_loader=targetkg_val_loader,
            controllm_val_loader=controllm_val_loader,
            controlkg_val_loader=controlkg_val_loader,
            accelerator=accelerator,
            verbose=args.verbose
        )
    
    time_passed = (time.time() - time_start)
    write_lines(["Time passed in minutes: {}".format(time_passed / 60.0)], os.path.join(args.exper_dir, 'time.txt'))
    if accelerator is not None:
        accelerator.end_training()

if __name__ == '__main__':
    main()
