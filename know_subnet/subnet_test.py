import os
import glob
import argparse
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from know_subnet.utils import *
from know_subnet.metrics import *
from know_subnet.constants import *
from know_subnet.lm.lm_utils import *
from know_subnet.lm.mask import *
from know_subnet.data.wordnet_dataloader import load_wordnet_targetkg, load_wordnet_controlkg
from know_subnet.data.conceptnet_dataloader import load_conceptnet
from know_subnet.data.wikitext_dataloader import (
    gpt2_infer_wiki_ppl, 
    load_wikitext2_test_dataloader
)
from know_subnet.subnet_train_utils import test_mask


def get_config(exper_dir):
    if not os.path.exists(exper_dir):
        print("Could not find experiment directory!")
        exit()
    config_paths = list(glob.iglob('{}/config.json'.format(exper_dir), recursive=True))
    config = None
    for config_path in config_paths:
        config = parse_json_to_dict(config_path)
    print("Found matching config!")
    return config

def get_model_ckpnt_list(exper_dir):
    if not os.path.exists(exper_dir):
        print("Could not find experiment directory!")
        exit()
    checkpoint_paths = list(glob.iglob('{}/checkpoints/ckpt-step=*-*'.format(exper_dir), recursive=True))
    checkpoint_steps = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_title = checkpoint_path.split("/")[-1].split("-")
        checkpoint_step = checkpoint_title[1].split("=")[1]
        checkpoint_steps.append(checkpoint_step)
    return checkpoint_steps

def get_model_and_config(
        exper_dir,
        checkpoint_step,
    ):
    config_paths = list(glob.iglob('{}/config*'.format(exper_dir), recursive=True))
    checkpoint_paths = list(glob.iglob('{}/checkpoints/ckpt-step={}-*'.format(exper_dir, checkpoint_step), recursive=True))
    assert (len(checkpoint_paths) == 1)
    for config_path in config_paths:
        config = parse_json_to_dict(config_path)
    for checkpoint_path in checkpoint_paths:
        checkpoint_title = checkpoint_path.split("/")[-1].split("-")
        checkpoint_step = checkpoint_title[1].split("=")[1]
        checkpoint_sparsity = checkpoint_title[2].split("=")[1]
        model = load_from_checkpoint(config, checkpoint_path, torch.cuda.is_available(), used_accelerator=True)
        model.eval()
    return model, checkpoint_sparsity, config

def data_loading(config, accelerator): 
    accelerator.print("Loading data...")

    with accelerator.main_process_first():
        targetkg_train_loader = None
        targetkg_val_loader = None
        controlkg_train_loader = None
        controlkg_val_loader = None

        ########################################################################
        # 1) TargetKG loading
        ########################################################################
        accelerator.print("-" * 50)
        if config["kg_type"] == 'wordnet':
            accelerator.print("Loading WordNet TargetKG...")
            targetkg_train_loader, targetkg_val_loader =  load_wordnet_targetkg(
                targetkg_name=config["targetkg_name"], 
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_worse_format=False
            )
            accelerator.print("Loading WordNet TargetKG done.")
        elif config["kg_type"] == 'conceptnet':
            accelerator.print("Loading ConceptNet TargetKG...")
            targetkg_train_loader, targetkg_val_loader = load_conceptnet(
                targetkg_name=config["targetkg_name"], 
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_controlkg=False,
                is_csqa=config["is_csqa"]
            )
            accelerator.print("Loading ConceptNet TargetKG done.")
        
        ########################################################################
        # 2) ControlKG loading
        ########################################################################
        accelerator.print("-" * 50)
        if config["kg_type"] == "wordnet":
            accelerator.print("Loading WordNet ControlKG...")
            controlkg_train_loader, controlkg_val_loader = load_wordnet_controlkg(
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_worse_format=False
            )
            accelerator.print("Loading WordNet ControlKG done.")
        elif config["kg_type"] == "conceptnet":
            accelerator.print("Loading ConceptNet ControlKG...")
            controlkg_train_loader, controlkg_val_loader = load_conceptnet(
                targetkg_name=config["targetkg_name"], 
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_controlkg=True,
                is_csqa=config["is_csqa"]
            )
            accelerator.print("Loading ConceptNet ControlKG done.")
            
    accelerator.print("Dataset loading and saving done.")
    
    return (
        targetkg_train_loader,
        targetkg_val_loader,
        controlkg_train_loader,
        controlkg_val_loader
    )


@torch.no_grad()
def zeroshot_log_loop(
    model,
    targetkg_val_loader,
    controllm_val_loader,
    controlkg_val_loader,
    lm_name="gpt2",
    epoch=0,
    step=0,
    processed=0,
    accelerator=None,
    verbose=True
):
    model.eval()
    
    log_dict = {
        'epoch': epoch,
        'step': step,
        'processed': processed
    }
    
    loaders_and_prefixes = [
        (targetkg_val_loader, 'targetkg'),
        (controlkg_val_loader, 'controlkg')
    ]
    
    for loader, prefix in loaders_and_prefixes:
        metric_dict = test_mask(
            model=model,
            dataset_loader=loader,
            lm_name=lm_name,
            do_sparsity_calc=False,
            do_rank_calc=True,
            accelerator=accelerator,
            verbose=verbose
        )
        
        # Prefix the keys and update the log dictionary.
        prefixed_metrics = {f"{prefix}-{k}": v for k, v in metric_dict.items()}
        log_dict.update(prefixed_metrics)
        
    log_dict["controllm-ppl"] = gpt2_infer_wiki_ppl(model, controllm_val_loader, is_accelerator=True)
    
    return log_dict

@torch.no_grad()
def validation_log_loop(
    config,
    model,
    log_dict,
    targetkg_val_loader,
    controllm_val_loader,
    controlkg_val_loader,
    accelerator
):
    model.eval()

    # 1) Define the three validation passes.
    loaders_and_prefixes = [
        ("targetkg-",    targetkg_val_loader, True),
        # ("controllm-",   controllm_val_loader, False),
        # ("controlkg-",   controlkg_val_loader, False),
    ]
    # 2) Run the normal passes
    for prefix, loader, do_sparsity in loaders_and_prefixes:
        prefix = f"masked-{prefix}"
        metrics = test_mask(
            model=model,
            dataset_loader=loader,
            do_sparsity_calc=do_sparsity,
            do_rank_calc=False,
            accelerator=accelerator
        )
        metrics = {prefix + k: v for k, v in metrics.items()}
        log_dict.update(metrics)

    # 3) If any of the include_* flags is set, run inverse-mask passes
    if (
        config["include_controllm_loss"]
        or config["include_controlkg_loss"]
        or config["include_targetkg_suppression_loss"]
    ):
        # flip on inverse mask
        if accelerator is None:
            model.set_is_inverse_mask(is_inverse_mask=True)
        else:
            accelerator.unwrap_model(model).set_is_inverse_mask(is_inverse_mask=True)
            
        loaders_and_prefixes = [
            ("targetkg-",    targetkg_val_loader, True),
            ("controlkg-",   controlkg_val_loader, False),
        ]
        # run the same three jobs, but with an "inverse-" prefix on each
        for prefix, loader, do_sparsity in loaders_and_prefixes:
            inv_prefix = f"inverse-{prefix}"
            metrics = test_mask(
                model=model,
                dataset_loader=loader,
                do_sparsity_calc=do_sparsity,
                do_rank_calc=True,
                accelerator=accelerator
            )
            metrics = {inv_prefix + k: v for k, v in metrics.items()}
            log_dict.update(metrics)
            
        log_dict["inverse-controllm-ppl"] = gpt2_infer_wiki_ppl(model, controllm_val_loader, is_accelerator=True)

        # flip back
        if accelerator is None:
            model.set_is_inverse_mask(is_inverse_mask=False)
        else:
            accelerator.unwrap_model(model).set_is_inverse_mask(is_inverse_mask=False)

    return log_dict

def generate_column_order():
    columns = ['checkpoint_step', 'checkpoint_sparsity']
    
    for d in ["targetkg", "controlkg", "controllm"]:
        columns += [f"{d}-(inverse - full)-ppl"]
    
    columns += ['targetkg-(inverse - masked)-ppl']
    columns += ['targetkg-(full - masked)-ppl']
        
    for mask_mode in ["full", "inverse"]:
        for d in ["targetkg", "controlkg", "controllm"]:
            columns += [f"{mask_mode}-{d}-ppl"]
    
    columns += ['masked-targetkg-ppl']

    for d in ["targetkg", "controlkg"]:
        for m in ["rank", "prb"]:
            for stat in ["mean", "min", "max"]:
                columns += [f"{d}-(inverse - full)-{m}-{stat}"]

    return columns

def test_subnet_post_train():
    set_seed(42)
    parser = argparse.ArgumentParser(description="args for subnet_test.py main func")
    parser.add_argument(
        "--exper_name",
        type=str,
        default="",
        help="experiment name ID/name to separate it from other experiments")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="folder to store training logs")
    parser.add_argument(
        "--nfs_dir",
        type=str,
        default="",
        help="nfs root dir")
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="date of the experiment, will be filled if nothing is passed")
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="log directory")
    args, unknown = parser.parse_known_args()

    accelerator = Accelerator(
        log_with="wandb"
    )

    ############################################################################
    # 1) Load model config for dataloading
    args.exper_dir = os.path.join(args.nfs_dir, "logs")
    args.exper_dir = os.path.join(
        args.exper_dir, 
        args.project_name, 
        args.log_dir,
        f"{args.exper_name}-date={args.date}"
    )
    accelerator.print("Requested exper dir is: ", args.exper_dir)
    accelerator.print("Retrieving config of experiment...")
    config = get_config(args.exper_dir)
    accelerator.print("Retrieval done.")
    
    config["train_batch_size"] = 250
    config["eval_batch_size"] = 250
    config["controllm_eval_batch_size"] = 8

    ############################################################################
    # 2) Data loading
    _, targetkg_val_loader, _, controlkg_val_loader = data_loading(config, accelerator)
    
    accelerator.print("-" * 50)
    accelerator.print("Loading wikitext2 as ControlLM...")
    controllm_val_loader = load_wikitext2_test_dataloader(
        lm_name=config["lm"],
        block_size=512,
        controllm_eval_batch_size=config["controllm_eval_batch_size"]
    ) 
    accelerator.print("Loading ControlLM done.")

    targetkg_val_loader, controlkg_val_loader, controllm_val_loader = accelerator.prepare(targetkg_val_loader, controlkg_val_loader, controllm_val_loader)

    ############################################################################
    # 3) Setting up the full model for delta calculations
    if not config["lm"].startswith("gpt"):
        raise NotImplementedError("Only GPT2 model family is supported for now.")
    full_model = GPT2LM(
        use_dropout=False,
        lm_name=config["lm"]
    )
    full_model.freeze_params(exclude_name_list=[], verbose=False)
    full_model.eval()
    full_model = accelerator.prepare_model(full_model, evaluation_mode=True)
    
    ############################################################################
    # 4) Get full model results
    full_dict = zeroshot_log_loop(
        model=full_model,
        targetkg_val_loader=targetkg_val_loader,
        controllm_val_loader=controllm_val_loader,
        controlkg_val_loader=controlkg_val_loader,
        lm_name=config["lm"],
        epoch=0,
        step=0,
        processed=0,
        accelerator=accelerator,
        verbose=True
    )
    # add "full" prefix to all entries
    full_dict = {f"full-{k}": v for k, v in full_dict.items()}

    ############################################################################
    # 5) Get all checkpoints to be evaluated
    checkpoint_steps = get_model_ckpnt_list(args.exper_dir)
    accelerator.print("Retrieved {} checkpoints in total.".format(len(checkpoint_steps)))
    res_dictlist = []
    ckpnt_loop = tqdm(checkpoint_steps, desc='ckpnt step: ', leave=False, position=1)
    
    ############################################################################
    # 6) Start wandb logging into a new project
    wandb_dir = None
    wandb_proj_name = args.project_name + "_test"
    accelerator.init_trackers(
        project_name=wandb_proj_name,
        config=vars(args),
        init_kwargs={"wandb": {
            "reinit": True, 
            "dir": wandb_dir, 
            "name": args.exper_name + "-date=" + args.date
        }}
    )
    keep_columns = generate_column_order()
    res_table = wandb.Table(columns=keep_columns)

    ############################################################################
    # 7) Define some general metric vars
    PPL_DATASETS = ["targetkg", "controlkg", "controllm"]
    RANK_PRB_DATASETS = ["targetkg", "controlkg"]
    TARGETKG_DELTA_PPL_PAIRS = {
        # name           (minuend, subtrahend)
        "(inverse - full)":    ("inverse", "full"),
        "(inverse - masked)":  ("inverse", "masked"),
        "(full - masked)":     ("full",    "masked"),
    }
    OTHERS_DELTA_PPL_PAIRS = {
        # name           (minuend, subtrahend)
        "(inverse - full)":    ("inverse", "full"),
    }
    
    ############################################################################
    # 8) Loop through checkpoints and evaluate
    model = None
    for checkpoint_step in ckpnt_loop:
        del model
        accelerator.free_memory() 
        model, checkpoint_sparsity, config = get_model_and_config(args.exper_dir, checkpoint_step)
        model.freeze_params(exclude_name_list=[], verbose=False)
        model.eval()
        model = accelerator.prepare(model)
        
        log_dict = {
            'epoch': 0,
            'step': 0,
            'processed': 0,
        }
        
        log_dict = validation_log_loop(
            config=config,
            model=model, 
            log_dict=log_dict,
            targetkg_val_loader=targetkg_val_loader,
            controllm_val_loader=controllm_val_loader,
            controlkg_val_loader=controlkg_val_loader,
            accelerator=accelerator
        )

        # Gather PPLs per mask type / dataset
        ppl_dict = {
            "full":   { d: full_dict[f"full-{d}-ppl"]   for d in PPL_DATASETS },
            "inverse":{ d: log_dict[f"inverse-{d}-ppl"] for d in PPL_DATASETS },
            "masked": { "targetkg": log_dict["masked-targetkg-ppl"] }  # masked only has targetkg
        }
        # Probably repetitive but doing it to be safe + to add to step_results
        step_results = {}
        for mask_mode, data_dict in ppl_dict.items():
            for d, value in data_dict.items():
                step_results[f"{mask_mode}-{d}-ppl"] = value

        # Gather rank and prb metrics per mask type / dataset
        for d in RANK_PRB_DATASETS:
            step_results[f"full-{d}-rank"] = full_dict[f"full-{d}-gold_rank"]
            step_results[f"full-{d}-prb"]  = full_dict[f"full-{d}-gold_prb"]
            step_results[f"inverse-{d}-rank"] = log_dict[f"inverse-{d}-gold_rank"]
            step_results[f"inverse-{d}-prb"]  = log_dict[f"inverse-{d}-gold_prb"]

        # Compute deltas for PPLs
        for d in ["targetkg"]:
            for key, (minuend, subtrahend) in TARGETKG_DELTA_PPL_PAIRS.items():
                step_results[f"{d}-{key}-ppl"] = ppl_dict[minuend][d] - ppl_dict[subtrahend][d]
        for d in ["controlkg", "controllm"]:
            for key, (minuend, subtrahend) in OTHERS_DELTA_PPL_PAIRS.items():
                step_results[f"{d}-{key}-ppl"] = ppl_dict[minuend][d] - ppl_dict[subtrahend][d]

        # Compute deltas for ranks/prbs
        for d in RANK_PRB_DATASETS:
            for metric in ["rank", "prb"]:
                full_value = step_results[f"full-{d}-{metric}"]
                inv_value  = step_results[f"inverse-{d}-{metric}"]
                diff_value = inv_value - full_value
                step_results.update({
                    f"{d}-(inverse - full)-{metric}-mean": diff_value.mean().item(),
                    f"{d}-(inverse - full)-{metric}-min":  diff_value.min().item(),
                    f"{d}-(inverse - full)-{metric}-max":  diff_value.max().item(),
                })

        # Add checkpoint metadata
        step_results["checkpoint_step"]     = int(checkpoint_step)
        step_results["checkpoint_sparsity"] = float(checkpoint_sparsity)

        # Log to wandb
        accelerator.log(step_results)
        myrowlist = [step_results[c] for c in keep_columns]
        res_table.add_data(*myrowlist)
        
        # Save to CSV locally as the checkpoint loop progresses
        res_dictlist.append(step_results)
        new_res_dictlist = []
        for mydict in res_dictlist:
            new_mydict = {k: mydict[k] for k in keep_columns}
            new_res_dictlist.append(new_mydict)
        save_dictlist_to_csv(
            mydictlist=new_res_dictlist, 
            filename=os.path.join(args.exper_dir, f"res-{args.exper_name}-date={args.date}.csv"))
    
    accelerator.log({"results": res_table})
    accelerator.end_training()


if __name__ == '__main__':
    test_subnet_post_train()
