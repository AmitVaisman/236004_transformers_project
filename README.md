# Discovering Knowledge-Critical Subnetworks in Pretrained Language Models

This codebase is the implementation of the EMNLP 2024 paper *Discovering Knowledge-Critical Subnetworks in Pretrained Language Models*. 

**Paper:** [https://arxiv.org/abs/2310.03084](https://arxiv.org/abs/2310.03084).


![image](assets/know-subnet-overview.png)

## Table of Contents

* [Overview](#overview)
* [Setup](#setup)
* [Usage](#usage)
    * [Mask training & testing]()
    * [Downstream task finetuning]()
    * [Structural analysis]()
* [Bugs or questions](#bugs-or-questions)
* [Citation](#citation)

## Overview

Pretrained language models (LMs) encode implicit representations of knowledge in their parameters. However, localizing these representations and disentangling them from each other remains an open problem. In this work, we investigate whether pretrained language models contain various knowledge-critical subnetworks: particular sparse computational subgraphs that can, if removed, precisely suppress specific knowledge the model has memorized. We propose a multi-objective differentiable masking scheme that can be applied to both weights and neurons to discover such subnetworks and show that we can use them to precisely remove specific knowledge from models while minimizing adverse effects on the behavior of the original model. We demonstrate our method on multiple GPT2 variants, uncovering highly sparse subnetworks (98%+ sparsity) that are critical for expressing specific collections of relational knowledge. When these subnetworks are removed, the remaining network maintains most of its initial abilities but struggles to represent the suppressed knowledge.

## Setup

### Requirements

To run our code, you will need Python 3.10. While a portion of our code was originally run in python 3.7 and 3.8, we provide a 3.10 version to help future users integrate new models with it. You can download the requirements as follows:

```shell
conda create --name subnetvenv python=3.10
conda activate subnetvenv
pip install -r requirements.txt
```

Then you will need to download the codebase as a package at the root of the repository:

```shell
cd know-subnet # TODO: right now it's know-subnet-clean, but it will be changed at final push
pip install -e .
```

Our code uses `wandb` to log the experiment metrics. So make sure to login with your API key or have the key in your OS environment. 

### Data

Due to its small size, the data is directly uploaded to the repository in the `data` folder.

## Usage

### Mask training & testing

The primary files needed to run the experiments are `subnet_train.py` and `subnet_test.py`. A demonstration on how to run them is in the following bash file:

```shell
bash scripts/train_test_subnet_example.sh
```

For more information about the arguments, please refer to `args.py`.

### Downstream task finetuning

For the downstream task finetuning experiments you would need to (1) train a mask for the TargetKG and (2) finetune the model where that mask is removed with a specific finetuning style.

For step (1):

```
TODO
```


For step (2):
```
TODO
```

### Structural analysis

```
TODO
```


## Bugs or questions

Note that this codebase is purely for the purpose of research and scientific experiments. We expect unknown bugs or issues caused by different package versions. If you encounter any problems when using the code or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker! If you have any questions related to the code or the paper, feel free to email the corresponding authors.

## Related Codebases

This work was built on top of prior work and open-sourced code, which we are very grateful for. We particularly thank the contributors of [stevenxcao/subnetwork-probing](https://github.com/stevenxcao/subnetwork-probing) and [RobertCsordas/modules](https://github.com/RobertCsordas/modules) as majority of the mask modules were borrowed from them. We also highly suggest looking at more generic masking code from [mlepori1/NeuroSurgeon](https://github.com/mlepori1/NeuroSurgeon) if you are interested in trying different types of model pruning with our approach.

## Citation

If you use our method in your work, please cite our paper:

```bibtex
@inproceedings{bayazit-etal-2024-discovering,
    title = "Discovering Knowledge-Critical Subnetworks in Pretrained Language Models",
    author = "Bayazit, Deniz  and
      Foroutan, Negar  and
      Chen, Zeming  and
      Weiss, Gail  and
      Bosselut, Antoine",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.376/",
    doi = "10.18653/v1/2024.emnlp-main.376"
}
```