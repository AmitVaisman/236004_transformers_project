import os
import torch
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from know_subnet.constants import PATH_DICT, DEEP_SEEK_MODEL

from know_subnet.utils import (
    seed_worker,
    parse_json_to_dict
)
from know_subnet.lm.lm_utils import create_uniform_dist

torch.set_printoptions(precision=32)

kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
cse_loss = torch.nn.CrossEntropyLoss(reduction='none')

class OursDataset(Dataset):
    def __init__(
            self, 
            inputs: List[str], 
            labels: List[str], 
            lm,
            is_controlkg: bool = False,
        ):
        if len(labels) != len(inputs):
            raise ValueError("Mismatch in inputs and labels length.")

        self.inputs_str = inputs
        self.labels_str = labels
        
        self.tokenizer = AutoTokenizer.from_pretrained(lm, fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        self.labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)["input_ids"]

        # set all non-masked locs in labels ids to -100 so the cross entropy loss ignores it:
        # https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
        final_ids = [len(ids) for ids in self.tokenizer(inputs)["input_ids"]]
        self.mask = torch.ones(self.labels.shape[0], self.labels.shape[1])
        for row, idx in zip(self.mask, final_ids):
            row[idx - 1] = 0
        # self.mask = self.mask == 1
        # self.labels[self.mask] = -100

        self.is_controlkg = is_controlkg
        if not self.is_controlkg:
            self.uniform_kl_dist = create_uniform_dist(self.labels, lm)
        
        inp_ids = self.inputs['input_ids']


    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.inputs.items()}
        item['inputs_str'] = self.inputs_str[idx]
        item['mask'] = self.mask[idx]
        
        if not(self.is_controlkg):
            item['uniform_kl'] = self.uniform_kl_dist[idx]
        
        if not(self.labels_str is None):
            item['labels'] = self.labels[idx]
            item['labels_str'] = self.labels_str[idx]

        
        return item

    def __len__(self):
        return len(self.inputs_str)


def load_ours(
        targetkg_name: str, 
        lm: str,
        train_batch_size: int,
        eval_batch_size: int,
        use_cuda: bool = True,
        is_worse_format: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
    fam_dict_list = []
    fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["ours_targetkg"], f"filtered_data.json"))
                
    data_len = len(fam_dict_list)

    print("Train TargetKG len: ", data_len)
    
    inputs = []
    labels = []
    for entry in fam_dict_list:
        inputs.append(entry[f"question"])
        labels.append(entry[f"only_answer_part"])
    
    train_dataset = OursDataset(
        inputs=inputs, 
        labels=labels,
        lm=lm,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        pin_memory=use_cuda, 
        num_workers=0,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        train_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        pin_memory=use_cuda, 
        num_workers=0,
        worker_init_fn=seed_worker)

    return train_loader, val_loader

def load_ours_controlkg(
        lm: str,
        train_batch_size: int,
        eval_batch_size: int,
        use_cuda: bool = True,
        val_num: int = 50,
        is_worse_format: bool = False,
        do_train_shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:

    train_fam_dict_list = None
    val_fam_dict_list = None
    if not is_worse_format:
        train_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_controlkg"], f"controlkg_train_best_format.json"))
        val_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_controlkg"], f"controlkg_val_best_format.json"))[:val_num]
        train_fam_dict_list = [mydict for mydict in train_fam_dict_list if mydict not in val_fam_dict_list]
    else:
        train_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_controlkg"], f"controlkg_train_worse_format.json"))
        val_limit = 50 * 21
        val_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_controlkg"], f"controlkg_val_worse_format.json"))[:val_limit]
    
    print("Train ControlKG len: ", len(train_fam_dict_list))
    print("Val ControlKG len: ", len(val_fam_dict_list))

    inputs = []
    labels = []
    for entry in train_fam_dict_list:
        inputs.append(entry[f"best_gpt2_input"])
        labels.append(entry[f"best_gpt2_label"])
    
    train_dataset = WordNetDataset(
        inputs=inputs, 
        labels=labels,
        lm=lm,
        is_controlkg=True
    )
    if is_worse_format:
        do_train_shuffle = False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=do_train_shuffle, 
        pin_memory=use_cuda, 
        num_workers=0,
        worker_init_fn=seed_worker
    )

    inputs = []
    labels = []
    for entry in val_fam_dict_list:
        inputs.append(entry[f"best_gpt2_input"])
        labels.append(entry[f"best_gpt2_label"])

    val_dataset = WordNetDataset(
        inputs=inputs, 
        labels=labels,
        lm=lm,
        is_controlkg=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        pin_memory=use_cuda, 
        num_workers=0,
        worker_init_fn=seed_worker)

    return train_loader, val_loader


def test():
    import pprint 
    pp = pprint.PrettyPrinter(indent=4)
    def print_batches(input_dataloader):
        cnt = 0
        for idx, batch in enumerate(input_dataloader):
            print("_" * 60)
            print(f"idx={idx}")
            print(f"cnt={cnt}")
            pp.pprint(batch)
            temp_labels = batch["labels"]
            temp_labels[temp_labels == -100] = input_dataloader.dataset.tokenizer.pad_token_id
            print("Decoded labels:")
            print(input_dataloader.dataset.tokenizer.batch_decode(temp_labels, skip_special_tokens=True))
            print(batch["labels_str"][:2])
            cnt += 1
            if  torch.all(batch["labels"] == 1):
                raise ValueError("This dataloader has 1s as labels!")
            if cnt == 2:
                break
            
    targetkg_name = "Synset('statement.n.01')"
    lm = DEEPSEEK_MODEL

    targetkg_train_loader, targetkg_val_loader = load_wordnet_targetkg(
        targetkg_name=targetkg_name, 
        lm=lm,
        train_batch_size=250,
        eval_batch_size=250,
        use_cuda=True,
        is_worse_format=False,
    )
    print_batches(targetkg_train_loader)
    print_batches(targetkg_val_loader)
    
    controlkg_train_loader, controlkg_val_loader = load_wordnet_controlkg(
        lm=lm,
        train_batch_size=50,
        eval_batch_size=50,
        use_cuda=True,
    )
    print_batches(controlkg_train_loader)
    print_batches(controlkg_val_loader)


if __name__ == '__main__':
    test()